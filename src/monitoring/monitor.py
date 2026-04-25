"""
monitoring/monitor.py
──────────────────────
Production monitoring for data drift and prediction distribution shifts.

Two types of drift to detect:
  1. Data drift (covariate shift): Input feature distributions change.
     E.g., product mix changes → avg_session_duration distribution shifts.
     Detected with: Population Stability Index (PSI) + Kolmogorov-Smirnov test.

  2. Concept drift: The relationship between features and target changes.
     E.g., a new competitor launches → previously loyal customers become at risk.
     Detected with: prediction distribution shift + performance degradation.

Retraining triggers:
  - PSI > 0.2 for any key feature (significant drift)
  - Mean predicted probability shifts > 10pp from baseline
  - If labeled data available: ROC-AUC drops > 5pp
  - Time-based: retrain every 30 days regardless

PSI interpretation:
  < 0.10 : No significant change
  0.10–0.20 : Moderate change — monitor closely
  > 0.20 : Significant change — trigger retraining
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    DRIFT_PSI_THRESHOLD, DRIFT_KS_PVALUE_THRESHOLD,
    PREDICTION_SHIFT_THRESHOLD, MONITORING_WINDOW_DAYS,
    LOGS_DIR, ARTIFACTS_DIR
)

logger = logging.getLogger(__name__)


# ── Population Stability Index ────────────────────────────────────────────────

def compute_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Compute Population Stability Index between reference and current distributions.

    PSI = Σ (actual% - expected%) × ln(actual% / expected%)

    Args:
        expected: Reference (training) distribution
        actual: Current (production) distribution
        n_bins: Number of histogram bins
        eps: Small constant to avoid log(0)

    Returns:
        PSI score
    """
    # Use quantile-based bins (more robust than equal-width)
    breakpoints = np.nanpercentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # remove duplicates for degenerate distributions

    if len(breakpoints) < 3:
        # Not enough unique values for binning (e.g., binary feature)
        return 0.0

    expected_counts = np.histogram(expected, bins=breakpoints)[0] + eps
    actual_counts = np.histogram(actual, bins=breakpoints)[0] + eps

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


def compute_ks_test(
    reference: np.ndarray,
    current: np.ndarray,
) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov test for distributional difference.
    Returns (statistic, p_value). Low p_value → drift detected.
    """
    stat, p_value = stats.ks_2samp(reference, current)
    return float(stat), float(p_value)


# ── Feature drift analysis ────────────────────────────────────────────────────

class DriftDetector:
    """
    Computes drift metrics for all monitored features.
    Maintains a reference distribution (from training) and compares to current.
    """

    def __init__(self):
        self.reference_stats: Dict = {}
        self.drift_history: List[Dict] = []

    def fit_reference(self, X_train: pd.DataFrame, numeric_cols: List[str]):
        """Store reference distribution statistics from training data."""
        for col in numeric_cols:
            if col in X_train.columns:
                vals = X_train[col].dropna().values
                self.reference_stats[col] = {
                    "values": vals,
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "p25": float(np.percentile(vals, 25)),
                    "p50": float(np.percentile(vals, 50)),
                    "p75": float(np.percentile(vals, 75)),
                }
        logger.info(f"Reference distribution fitted for {len(self.reference_stats)} features")

    def detect_drift(
        self,
        X_current: pd.DataFrame,
        window_label: str = "current",
    ) -> Dict:
        """
        Compare current data distribution to reference.
        Returns drift report with feature-level PSI and KS test results.
        """
        if not self.reference_stats:
            raise RuntimeError("Call fit_reference() before detect_drift()")

        results = {
            "window": window_label,
            "timestamp": datetime.utcnow().isoformat(),
            "features": {},
            "overall_alert": False,
            "alerted_features": [],
        }

        for col, ref_stats in self.reference_stats.items():
            if col not in X_current.columns:
                continue

            current_vals = X_current[col].dropna().values
            if len(current_vals) < 30:
                logger.warning(f"Too few samples ({len(current_vals)}) for drift check on {col}")
                continue

            psi = compute_psi(ref_stats["values"], current_vals)
            ks_stat, ks_pvalue = compute_ks_test(ref_stats["values"], current_vals)

            drift_detected = psi > DRIFT_PSI_THRESHOLD or ks_pvalue < DRIFT_KS_PVALUE_THRESHOLD

            results["features"][col] = {
                "psi": round(psi, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_pvalue, 6),
                "drift_detected": drift_detected,
                "current_mean": round(float(current_vals.mean()), 4),
                "reference_mean": round(ref_stats["mean"], 4),
                "mean_shift_pct": round(
                    abs(current_vals.mean() - ref_stats["mean"]) / (ref_stats["mean"] + 1e-10) * 100, 2
                ),
            }

            if drift_detected:
                results["overall_alert"] = True
                results["alerted_features"].append(col)
                logger.warning(
                    f"DRIFT DETECTED — {col}: PSI={psi:.3f}, KS p-value={ks_pvalue:.4f}"
                )

        self.drift_history.append(results)

        # Summary
        n_features = len(results["features"])
        n_drifted = len(results["alerted_features"])
        logger.info(
            f"Drift check [{window_label}]: {n_drifted}/{n_features} features drifted | "
            f"Alert: {results['overall_alert']}"
        )

        return results


# ── Prediction distribution monitoring ────────────────────────────────────────

class PredictionMonitor:
    """
    Tracks the distribution of model predictions over time.
    Alerts when the mean predicted probability shifts significantly from baseline.

    This is a proxy for concept drift when labels aren't available yet.
    """

    def __init__(self):
        self.baseline_mean_prob: Optional[float] = None
        self.prediction_log: List[Dict] = []

    def set_baseline(self, train_probs: np.ndarray):
        self.baseline_mean_prob = float(train_probs.mean())
        logger.info(f"Baseline mean churn probability: {self.baseline_mean_prob:.4f}")

    def log_predictions(self, probs: np.ndarray, window_label: str = "current"):
        current_mean = float(probs.mean())
        shift = abs(current_mean - self.baseline_mean_prob) if self.baseline_mean_prob else 0

        entry = {
            "window": window_label,
            "timestamp": datetime.utcnow().isoformat(),
            "mean_prob": round(current_mean, 4),
            "baseline_mean_prob": self.baseline_mean_prob,
            "shift": round(shift, 4),
            "alert": shift > PREDICTION_SHIFT_THRESHOLD,
            "n_predictions": len(probs),
            "pct_high_risk": round(float((probs > 0.6).mean()), 4),
        }
        self.prediction_log.append(entry)

        if entry["alert"]:
            logger.warning(
                f"PREDICTION SHIFT ALERT: baseline={self.baseline_mean_prob:.3f} "
                f"current={current_mean:.3f} shift={shift:.3f} (threshold={PREDICTION_SHIFT_THRESHOLD})"
            )
        else:
            logger.info(
                f"Prediction distribution stable: mean={current_mean:.3f} shift={shift:.3f}"
            )

        return entry


# ── Retraining trigger logic ──────────────────────────────────────────────────

def should_retrain(
    drift_report: Dict,
    pred_monitor_entry: Dict,
    days_since_last_training: int,
    roc_auc_drop: Optional[float] = None,
) -> Tuple[bool, str]:
    """
    Aggregate all signals to make a retraining decision.

    Returns:
        (should_retrain: bool, reason: str)
    """
    reasons = []

    # Signal 1: Feature drift
    if drift_report.get("overall_alert"):
        drifted = drift_report.get("alerted_features", [])
        reasons.append(f"Feature drift in {len(drifted)} features: {drifted}")

    # Signal 2: Prediction distribution shift
    if pred_monitor_entry.get("alert"):
        shift = pred_monitor_entry.get("shift", 0)
        reasons.append(f"Prediction distribution shifted by {shift:.1%}")

    # Signal 3: Scheduled retraining (time-based fallback)
    MAX_DAYS_WITHOUT_RETRAIN = 30
    if days_since_last_training >= MAX_DAYS_WITHOUT_RETRAIN:
        reasons.append(f"Scheduled retrain: {days_since_last_training} days since last training")

    # Signal 4: Performance degradation (when labels become available)
    if roc_auc_drop is not None and roc_auc_drop > 0.05:
        reasons.append(f"ROC-AUC dropped by {roc_auc_drop:.3f}")

    trigger = len(reasons) > 0
    reason_str = " | ".join(reasons) if reasons else "No retraining needed"

    if trigger:
        logger.warning(f"🔄 RETRAINING TRIGGERED: {reason_str}")
    else:
        logger.info(f"✅ No retraining needed. All signals stable.")

    return trigger, reason_str


# ── Monitoring dashboard plot ─────────────────────────────────────────────────

def plot_monitoring_dashboard(
    prediction_log: List[Dict],
    drift_history: List[Dict],
    key_features: List[str] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Simple monitoring dashboard showing:
      - Prediction probability distribution over time
      - PSI trends for top features
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Production Monitoring Dashboard", fontsize=14, fontweight="bold")

    # Panel 1: Prediction distribution over windows
    if prediction_log:
        windows = [p["window"] for p in prediction_log]
        means = [p["mean_prob"] for p in prediction_log]
        high_risk = [p.get("pct_high_risk", 0) for p in prediction_log]

        ax = axes[0]
        ax.plot(windows, means, "o-", color="#2563EB", lw=2, label="Mean Churn Prob")
        ax.axhline(
            prediction_log[0].get("baseline_mean_prob", means[0]),
            color="gray", linestyle="--", label="Baseline"
        )
        ax.axhline(
            (prediction_log[0].get("baseline_mean_prob", means[0]) or 0) + PREDICTION_SHIFT_THRESHOLD,
            color="#DC2626", linestyle=":", alpha=0.7, label="Alert threshold"
        )
        ax.fill_between(range(len(means)),
                        [m - 0.02 for m in means],
                        [m + 0.02 for m in means],
                        alpha=0.15, color="#2563EB")
        ax.set_xlabel("Monitoring Window")
        ax.set_ylabel("Mean Predicted Churn Probability")
        ax.set_title("Prediction Distribution Stability")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(axis='x', rotation=30)

    # Panel 2: PSI heatmap across features and windows
    if drift_history and key_features:
        psi_data = {}
        for entry in drift_history:
            window = entry["window"]
            psi_data[window] = {}
            for feat in key_features:
                feat_data = entry.get("features", {}).get(feat, {})
                psi_data[window][feat] = feat_data.get("psi", 0)

        psi_df = pd.DataFrame(psi_data).T  # windows × features
        ax = axes[1]
        im = ax.imshow(psi_df.values.T, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.3)
        ax.set_xticks(range(len(psi_df.index)))
        ax.set_xticklabels(psi_df.index, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(psi_df.columns)))
        ax.set_yticklabels(psi_df.columns, fontsize=8)
        plt.colorbar(im, ax=ax, label="PSI")
        ax.set_title("Feature Drift (PSI)\nGreen=stable, Red=drifted")
        ax.axhline(-0.5, color="#DC2626", linestyle="-", lw=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Monitoring dashboard saved to {save_path}")

    return fig
