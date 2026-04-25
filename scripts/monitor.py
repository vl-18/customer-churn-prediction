"""
scripts/monitor.py
───────────────────
Simulate production monitoring over multiple time windows.

In real production this runs as:
  - A cron job (every hour/day)
  - An Airflow DAG
  - A Kafka consumer processing real-time prediction logs

This script simulates data drift by progressively modifying the
data distribution across 5 monitoring windows.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import generate_churn_dataset
from src.data.preprocessor import prepare_data
from src.models.model_registry import ModelRegistry
from src.monitoring.monitor import DriftDetector, PredictionMonitor, should_retrain, plot_monitoring_dashboard
from configs.config import ARTIFACTS_DIR, NUMERIC_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def simulate_drifted_data(base_df: pd.DataFrame, drift_factor: float, rng) -> pd.DataFrame:
    """
    Simulate gradual data drift:
      - monthly_charges increases (price hike scenario)
      - days_since_last_login increases (disengagement)
      - num_support_calls increases (service degradation)
    """
    df = base_df.copy()
    df["monthly_charges"] = (df["monthly_charges"] * (1 + drift_factor * 0.15)).clip(18, 120)
    df["days_since_last_login"] = (df["days_since_last_login"] * (1 + drift_factor * 0.3)).clip(0, 90).astype(int)
    df["num_support_calls"] = (df["num_support_calls"] * (1 + drift_factor * 0.4)).clip(0, 10).astype(int)
    # Add some noise
    df["avg_session_duration_mins"] = (
        df["avg_session_duration_mins"].fillna(30) * (1 - drift_factor * 0.2) + rng.normal(0, 2, len(df))
    ).clip(1, 180)
    return df


def run_monitoring():
    logger.info("=" * 60)
    logger.info("Production Monitoring Simulation")
    logger.info("=" * 60)

    # ── Load champion model ───────────────────────────────────────────────────
    registry = ModelRegistry()
    try:
        pipeline, winsorizer, metadata = registry.load_champion()
    except FileNotFoundError:
        logger.error("No champion model found. Run training first: python scripts/train.py")
        sys.exit(1)

    # ── Generate reference training data ──────────────────────────────────────
    logger.info("Loading reference (training) data distribution")
    reference_df = generate_churn_dataset(n_samples=5000, random_state=42)
    X_ref, y_ref, _ = prepare_data(reference_df, fit_winsorizer=False, winsorizer=winsorizer)

    # Compute baseline prediction distribution
    ref_probs = pipeline.predict_proba(X_ref)[:, 1]

    # ── Initialize monitors ───────────────────────────────────────────────────
    drift_detector = DriftDetector()
    numeric_cols = [c for c in NUMERIC_FEATURES if c in X_ref.columns
                    and not c.startswith("usage_score")
                    and not c.startswith("charge_per_month")
                    and not c.startswith("support_call_rate")
                    and not c.startswith("tenure_x_products")]

    # Use raw numeric cols that exist in reference
    monitor_cols = [c for c in ["monthly_charges", "days_since_last_login",
                                  "num_support_calls", "tenure_months",
                                  "avg_session_duration_mins"] if c in X_ref.columns]
    drift_detector.fit_reference(X_ref, monitor_cols)

    pred_monitor = PredictionMonitor()
    pred_monitor.set_baseline(ref_probs)

    # ── Simulate 5 monitoring windows ────────────────────────────────────────
    rng = np.random.RandomState(99)
    windows = ["Week 1", "Week 2", "Week 3", "Week 4 (drift)", "Week 5 (severe drift)"]
    drift_factors = [0.0, 0.05, 0.10, 0.25, 0.45]  # gradually increasing

    all_drift_reports = []
    all_pred_entries = []
    retrain_recommendations = []

    for window, drift_factor in zip(windows, drift_factors):
        logger.info(f"\n{'─'*40}\nMonitoring Window: {window} (drift={drift_factor:.0%})")

        # Simulate current production data
        current_raw = generate_churn_dataset(n_samples=1000, random_state=rng.randint(0, 9999))
        if drift_factor > 0:
            current_raw = simulate_drifted_data(current_raw, drift_factor, rng)

        X_current, _, _ = prepare_data(current_raw, fit_winsorizer=False, winsorizer=winsorizer)
        current_probs = pipeline.predict_proba(X_current)[:, 1]

        # Detect feature drift
        drift_report = drift_detector.detect_drift(X_current, window_label=window)
        all_drift_reports.append(drift_report)

        # Track prediction distribution
        pred_entry = pred_monitor.log_predictions(current_probs, window_label=window)
        all_pred_entries.append(pred_entry)

        # Retraining decision
        days_since_training = (windows.index(window) + 1) * 7
        retrain, reason = should_retrain(
            drift_report=drift_report,
            pred_monitor_entry=pred_entry,
            days_since_last_training=days_since_training,
        )
        retrain_recommendations.append({
            "window": window,
            "retrain": retrain,
            "reason": reason,
        })

        # Summary for this window
        n_drifted = len(drift_report.get("alerted_features", []))
        logger.info(
            f"  Features drifted: {n_drifted}/{len(monitor_cols)} | "
            f"Pred mean: {pred_entry['mean_prob']:.3f} | "
            f"Pred shift: {pred_entry['shift']:.3f} | "
            f"Retrain: {'YES ⚠️' if retrain else 'NO ✅'}"
        )
        if drift_report.get("alerted_features"):
            logger.info(f"  Drifted features: {drift_report['alerted_features']}")

    # ── Plot monitoring dashboard ─────────────────────────────────────────────
    logger.info("\nGenerating monitoring dashboard")
    plot_monitoring_dashboard(
        prediction_log=all_pred_entries,
        drift_history=all_drift_reports,
        key_features=monitor_cols[:5],
        save_path=ARTIFACTS_DIR / "monitoring_dashboard.png"
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("MONITORING SUMMARY")
    logger.info("=" * 60)
    for rec in retrain_recommendations:
        status = "🔄 RETRAIN" if rec["retrain"] else "✅ STABLE"
        logger.info(f"{rec['window']:20s} | {status} | {rec['reason'][:80]}")

    logger.info(f"\nMonitoring dashboard saved to {ARTIFACTS_DIR / 'monitoring_dashboard.png'}")
    logger.info("\nRetrain Trigger Policy:")
    logger.info("  - PSI > 0.20 for any key feature → retrain")
    logger.info("  - Prediction distribution shift > 10pp → retrain")
    logger.info("  - 30+ days without retraining → scheduled retrain")
    logger.info("  - ROC-AUC drops > 5pp (when labels available) → emergency retrain")


if __name__ == "__main__":
    run_monitoring()
