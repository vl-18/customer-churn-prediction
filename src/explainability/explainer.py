"""
explainability/explainer.py
────────────────────────────
SHAP-based model explainability.

Why SHAP?
  - Model-agnostic: works for LR, RF, XGBoost without rewriting
  - Theoretically grounded: Shapley values have fairness guarantees
  - Both global (feature importance) and local (single prediction) explanations
  - Consistent: a feature's total SHAP contribution = prediction - baseline

TreeExplainer is used for RF/XGBoost (fast, exact).
LinearExplainer for Logistic Regression.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import logging
from pathlib import Path
from typing import Optional, Union

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import ARTIFACTS_DIR
from src.features.feature_engineer import get_feature_importance_names

logger = logging.getLogger(__name__)


def get_shap_explainer(pipeline, X_background: pd.DataFrame):
    """
    Build a SHAP explainer appropriate for the classifier type.

    TreeExplainer: exact values for tree-based models, much faster than KernelExplainer.
    LinearExplainer: exact values for linear models.
    KernelExplainer: fallback, model-agnostic but slow (O(n_features^2)).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    # Transform data for the explainer
    X_transformed = preprocessor.transform(X_background)

    clf_type = type(classifier).__name__

    if clf_type in ("RandomForestClassifier", "XGBClassifier", "GradientBoostingClassifier"):
        explainer = shap.TreeExplainer(
            classifier,
            data=X_transformed,
            feature_perturbation="interventional",
        )
        logger.info(f"Using TreeExplainer for {clf_type}")

    elif clf_type == "LogisticRegression":
        explainer = shap.LinearExplainer(
            classifier,
            X_transformed,
            feature_perturbation="interventional",
        )
        logger.info(f"Using LinearExplainer for {clf_type}")

    else:
        # Fallback — slow but universal
        background = shap.kmeans(X_transformed, 50)
        explainer = shap.KernelExplainer(
            classifier.predict_proba,
            background,
        )
        logger.warning(f"Using KernelExplainer for {clf_type} — this will be slow")

    return explainer, preprocessor


def compute_shap_values(
    explainer,
    preprocessor,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> np.ndarray:
    """
    Compute SHAP values for a dataset.
    Subsample for speed if dataset is large.
    Returns shap_values array of shape (n_samples, n_features).
    """
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)
        logger.info(f"Subsampled to {max_samples} rows for SHAP computation")

    X_transformed = preprocessor.transform(X)
    shap_values = explainer.shap_values(X_transformed)

    # For binary classifiers, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # use class 1 (churn) values

    return shap_values, X_transformed


def plot_global_importance(
    shap_values: np.ndarray,
    feature_names: list,
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart of mean |SHAP| per feature = global feature importance.
    More honest than impurity-based importance (no bias toward high-cardinality).
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=feature_names)
    top_features = feature_importance.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#2563EB" if i < 3 else "#93C5FD" for i in range(len(top_features))]
    top_features[::-1].plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value| (average impact on model output)")
    ax.set_title(f"Global Feature Importance (Top {top_n})\nvia SHAP — {top_n} most influential features")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)

    for i, (val, name) in enumerate(zip(top_features[::-1].values, top_features[::-1].index)):
        ax.text(val + 0.001, i, f"{val:.4f}", va="center", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Global importance plot saved to {save_path}")

    return fig


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X_transformed: np.ndarray,
    feature_names: list,
    top_n: int = 15,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Beeswarm / summary plot: shows BOTH importance AND direction of effect.
    Red = high feature value, Blue = low feature value.
    Right of 0 = pushes toward churn, Left = pushes away from churn.
    """
    fig, ax = plt.subplots(figsize=(11, 8))

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        max_display=top_n,
        show=False,
        plot_type="dot",
        color_bar=True,
        axis_color="#374151",
    )
    ax = plt.gca()
    ax.set_title("SHAP Beeswarm — Feature Impact Direction & Magnitude", fontsize=13)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"SHAP beeswarm saved to {save_path}")

    return plt.gcf()


def explain_single_prediction(
    pipeline,
    X_single: pd.DataFrame,
    explainer,
    feature_names: list,
    customer_id: str = "CUST_UNKNOWN",
    save_path: Optional[Path] = None,
) -> dict:
    """
    Local explanation for a single customer prediction.

    Returns:
        dict with churn_probability, top_factors (pro/con churn), waterfall plot
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_single)

    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    churn_prob = pipeline.predict_proba(X_single)[0, 1]

    # Build explanation dict
    contributions = pd.Series(sv, index=feature_names).sort_values(key=abs, ascending=False)
    top_churn_drivers = contributions[contributions > 0].head(5)
    top_retention_factors = contributions[contributions < 0].head(5)

    # Force plot saved as static bar chart (waterfall-style)
    fig, ax = plt.subplots(figsize=(10, 6))
    top15 = contributions.head(15).sort_values()
    colors = ["#DC2626" if v > 0 else "#16A34A" for v in top15.values]
    top15.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("SHAP Value (contribution to churn probability)")
    ax.set_title(
        f"Local Explanation — Customer {customer_id}\n"
        f"Churn Probability: {churn_prob:.1%} | "
        f"{'HIGH RISK 🔴' if churn_prob > 0.5 else 'LOW RISK 🟢'}"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Local explanation saved to {save_path}")

    logger.info(
        f"\nCustomer {customer_id} — Churn Prob: {churn_prob:.1%}\n"
        f"Top churn drivers: {top_churn_drivers.to_dict()}\n"
        f"Top retention factors: {top_retention_factors.to_dict()}"
    )

    return {
        "customer_id": customer_id,
        "churn_probability": float(churn_prob),
        "churn_prediction": churn_prob >= 0.40,
        "top_churn_drivers": top_churn_drivers.to_dict(),
        "top_retention_factors": top_retention_factors.to_dict(),
        "figure": fig,
    }
