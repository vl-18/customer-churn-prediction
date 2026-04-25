"""
evaluation/evaluator.py
────────────────────────
Comprehensive model evaluation with business-context metrics.

Key design choices:
  - ROC-AUC: threshold-agnostic ranking metric. Primary metric for model selection.
  - PR-AUC (Average Precision): better for imbalanced data. ROC-AUC can look
    optimistic when negatives dominate; PR-AUC is more honest.
  - Threshold tuning: default 0.5 is wrong for business problems. We sweep
    thresholds to maximize F1, then optionally shift toward recall if
    FN cost > FP cost (which is true for churn: missing a churner costs 3–5×
    more than a false alarm that wastes a retention offer).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import DEFAULT_THRESHOLD, ARTIFACTS_DIR

logger = logging.getLogger(__name__)


def evaluate_model(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
    split_name: str = "Test",
) -> Dict[str, float]:
    """
    Full evaluation of a fitted pipeline on a dataset.
    Returns dict of metrics.
    """
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "split": split_name,
        "threshold": threshold,
        "roc_auc": roc_auc_score(y, y_prob),
        "pr_auc": average_precision_score(y, y_prob),
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "n_samples": len(y),
        "n_positives": y.sum(),
        "pred_positive_rate": y_pred.mean(),
    }

    logger.info(
        f"[{split_name}] ROC-AUC={metrics['roc_auc']:.4f} | "
        f"PR-AUC={metrics['pr_auc']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"Precision={metrics['precision']:.4f} | "
        f"Recall={metrics['recall']:.4f} | "
        f"Threshold={threshold}"
    )

    return metrics


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    strategy: str = "f1",
    fn_cost: float = 3.0,
    fp_cost: float = 1.0,
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.

    Strategies:
      'f1'   : maximize F1 score (balanced precision/recall)
      'cost' : minimize business cost (FN cost × FN + FP cost × FP)
               fn_cost=3 means missing a churner costs 3× more than false alarm.
               Business rationale: retention offer costs ~$20; lost customer LTV ~$60+.

    Returns:
        (optimal_threshold, metric_at_threshold)
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = DEFAULT_THRESHOLD
    best_score = -np.inf

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        if strategy == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif strategy == "cost":
            # Lower cost = better; negate for argmax
            score = -(fn_cost * fn + fp_cost * fp)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        if score > best_score:
            best_score = score
            best_threshold = t

    logger.info(
        f"Optimal threshold ({strategy}): {best_threshold:.2f} | "
        f"Score: {best_score:.4f}"
    )
    return best_threshold, best_score


def bias_variance_report(
    pipeline,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    model_name: str = "Model",
) -> Dict:
    """
    Compare train vs val performance to diagnose bias/variance.

    Interpretation guide:
      Train ROC-AUC >> Val ROC-AUC (gap > 0.05) → overfitting (high variance)
        Fix: increase regularization, reduce features, add dropout (NN)
      Both low (< 0.70) → underfitting (high bias)
        Fix: add features, reduce regularization, use more complex model
      Both high, small gap → well-calibrated model ✓
    """
    train_metrics = evaluate_model(pipeline, X_train, y_train, split_name="Train")
    val_metrics = evaluate_model(pipeline, X_val, y_val, split_name="Val")

    gap = train_metrics["roc_auc"] - val_metrics["roc_auc"]
    diagnosis = "Well-calibrated"
    recommendation = "No action needed"

    if gap > 0.08:
        diagnosis = "Overfitting (High Variance)"
        recommendation = (
            "Increase regularization (C for LR, max_depth for trees), "
            "reduce n_estimators, apply dropout, or collect more training data"
        )
    elif val_metrics["roc_auc"] < 0.72:
        diagnosis = "Underfitting (High Bias)"
        recommendation = (
            "Add more features, reduce regularization, "
            "use a more complex model, or tune hyperparameters"
        )
    elif gap < 0.02:
        diagnosis = "Well-calibrated"
        recommendation = "Model generalizes well. Consider threshold tuning for business KPIs."

    report = {
        "model": model_name,
        "train_roc_auc": train_metrics["roc_auc"],
        "val_roc_auc": val_metrics["roc_auc"],
        "gap": gap,
        "diagnosis": diagnosis,
        "recommendation": recommendation,
    }

    logger.info(
        f"\n{'='*60}\n"
        f"Bias-Variance Report: {model_name}\n"
        f"  Train ROC-AUC : {train_metrics['roc_auc']:.4f}\n"
        f"  Val   ROC-AUC : {val_metrics['roc_auc']:.4f}\n"
        f"  Gap           : {gap:.4f}\n"
        f"  Diagnosis     : {diagnosis}\n"
        f"  Recommendation: {recommendation}\n"
        f"{'='*60}"
    )

    return report


def plot_evaluation_dashboard(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Champion Model",
    threshold: float = DEFAULT_THRESHOLD,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    4-panel evaluation dashboard:
      1. ROC Curve
      2. Precision-Recall Curve
      3. Confusion Matrix (heatmap)
      4. Threshold sweep (F1 vs threshold)
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Evaluation Dashboard — {model_name}", fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # ── Panel 1: ROC Curve ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax1.plot(fpr, tpr, color="#2563EB", lw=2.5, label=f"ROC (AUC = {auc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier")
    ax1.fill_between(fpr, tpr, alpha=0.08, color="#2563EB")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.02])
    ax1.grid(alpha=0.3)

    # ── Panel 2: Precision-Recall Curve ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    baseline = y_test.mean()
    ax2.plot(recall, precision, color="#16A34A", lw=2.5, label=f"PR Curve (AUC = {pr_auc:.4f})")
    ax2.axhline(baseline, color="gray", linestyle="--", lw=1.5,
                label=f"Baseline (prevalence = {baseline:.2f})")
    ax2.fill_between(recall, precision, baseline, alpha=0.08, color="#16A34A")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="upper right")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.02])
    ax2.grid(alpha=0.3)

    # ── Panel 3: Confusion Matrix ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum()
    im = ax3.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    labels = ["Not Churned", "Churned"]
    tick_marks = np.arange(2)
    ax3.set_xticks(tick_marks)
    ax3.set_xticklabels(labels)
    ax3.set_yticks(tick_marks)
    ax3.set_yticklabels(labels)
    ax3.set_xlabel("Predicted Label")
    ax3.set_ylabel("True Label")
    ax3.set_title(f"Confusion Matrix (threshold={threshold})")
    for i in range(2):
        for j in range(2):
            ax3.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1%})",
                     ha="center", va="center",
                     color="white" if cm_pct[i, j] > 0.5 else "black",
                     fontsize=13, fontweight="bold")

    # ── Panel 4: Threshold Sweep ──────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    thresholds_sweep = np.linspace(0.1, 0.9, 81)
    f1s, precisions, recalls = [], [], []
    for t in thresholds_sweep:
        yp = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_test, yp, zero_division=0))
        precisions.append(precision_score(y_test, yp, zero_division=0))
        recalls.append(recall_score(y_test, yp))

    ax4.plot(thresholds_sweep, f1s, color="#DC2626", lw=2.5, label="F1")
    ax4.plot(thresholds_sweep, precisions, color="#2563EB", lw=2, linestyle="--", label="Precision")
    ax4.plot(thresholds_sweep, recalls, color="#16A34A", lw=2, linestyle=":", label="Recall")
    ax4.axvline(threshold, color="black", linestyle="-.", lw=1.5, label=f"Current threshold ({threshold})")
    ax4.set_xlabel("Classification Threshold")
    ax4.set_ylabel("Score")
    ax4.set_title("Threshold Tuning\n(↓ threshold → ↑ recall, ↓ precision)")
    ax4.legend(loc="center left")
    ax4.set_xlim([0.1, 0.9])
    ax4.set_ylim([0, 1])
    ax4.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Evaluation dashboard saved to {save_path}")

    return fig


def print_classification_report(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
    model_name: str = "Champion Model",
):
    """Print full sklearn classification report."""
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{'='*60}")
    print(f"Classification Report — {model_name}")
    print(f"Threshold: {threshold} | Test samples: {len(y_test)}")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"]))
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
    print(f"PR-AUC:   {average_precision_score(y_test, y_prob):.4f}")
    print("="*60)
