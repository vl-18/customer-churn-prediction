"""
models/trainer.py
──────────────────
Multi-model training with:
  - Cross-validation (stratified)
  - Class imbalance handling (class_weight vs SMOTE — compared)
  - Consistent Pipeline wrapping (preprocessor + classifier)

Why wrap classifier in Pipeline?
  → Prevents data leakage during CV: preprocessing is fit only on training
    folds, not the validation fold. Without this, scaling/imputation fitted
    on all data leaks val statistics into training.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Any, Tuple

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import CV_FOLDS, SCORING_METRIC, RANDOM_STATE, IMBALANCE_STRATEGY

logger = logging.getLogger(__name__)


def get_baseline_models(imbalance_strategy: str = IMBALANCE_STRATEGY) -> Dict[str, Any]:
    """
    Returns baseline (un-tuned) estimators.

    Imbalance strategies:
    ─────────────────────
    class_weight="balanced":
      → Tells the loss function to weight minority class inversely proportional
        to frequency. Fast, no resampling, works for all models.
        Trade-off: doesn't add new information; may underfit minority class.

    SMOTE (Synthetic Minority Over-sampling Technique):
      → Generates synthetic minority samples by interpolating between neighbors.
        Can improve recall significantly but risks overfitting on synthetic points.
        Must be applied INSIDE the CV loop to avoid leakage.
        Trade-off: slower, adds complexity, may hurt precision.

    Verdict for this project: class_weight preferred.
      - Our churn rate (~27%) is moderate; SMOTE shines at extreme imbalance (5%).
      - class_weight is simpler and more robust for deployment.
      - We offer both via config toggle.
    """
    if imbalance_strategy == "class_weight":
        models = {
            "LogisticRegression": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=1.0,
                solver="lbfgs",
                random_state=RANDOM_STATE,
            ),
            "RandomForest": RandomForestClassifier(
                class_weight="balanced",
                n_estimators=100,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "XGBoost": xgb.XGBClassifier(
                # scale_pos_weight = n_neg / n_pos (set dynamically in train())
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                verbosity=0,
            ),
        }

    elif imbalance_strategy == "smote":
        # SMOTE must use imblearn Pipeline to be applied inside each CV fold
        lr = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE)),
        ])
        rf = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=10,
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ])
        xgb_model = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("clf", xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                       use_label_encoder=False, eval_metric="logloss",
                                       random_state=RANDOM_STATE, verbosity=0)),
        ])
        models = {
            "LogisticRegression_SMOTE": lr,
            "RandomForest_SMOTE": rf,
            "XGBoost_SMOTE": xgb_model,
        }
    else:
        raise ValueError(f"Unknown imbalance_strategy: {imbalance_strategy}")

    return models


def cross_validate_model(
    estimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    preprocessor=None,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Run stratified K-fold CV and return mean ± std for key metrics.

    If preprocessor provided, wraps estimator in a sklearn Pipeline
    so that preprocessing happens inside each fold (no leakage).
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Wrap with preprocessor if provided and estimator is not already a Pipeline
    if preprocessor is not None and not isinstance(estimator, (Pipeline, ImbPipeline)):
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])
    elif preprocessor is not None and isinstance(estimator, ImbPipeline):
        # SMOTE pipeline — prepend preprocessor
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("smote_clf", estimator),
        ])
    else:
        pipeline = estimator

    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    start = time.time()
    results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,  # For bias-variance analysis
        n_jobs=-1,
    )
    elapsed = time.time() - start

    metrics = {}
    for metric in scoring:
        val_scores = results[f"test_{metric}"]
        train_scores = results[f"train_{metric}"]
        metrics[f"val_{metric}_mean"] = val_scores.mean()
        metrics[f"val_{metric}_std"] = val_scores.std()
        metrics[f"train_{metric}_mean"] = train_scores.mean()

    metrics["cv_time_seconds"] = elapsed

    logger.info(
        f"{model_name} CV Results | "
        f"ROC-AUC: {metrics['val_roc_auc_mean']:.4f} ± {metrics['val_roc_auc_std']:.4f} | "
        f"PR-AUC: {metrics['val_average_precision_mean']:.4f} | "
        f"F1: {metrics['val_f1_mean']:.4f} | "
        f"Time: {elapsed:.1f}s"
    )

    # Bias-variance check
    roc_gap = metrics["train_roc_auc_mean"] - metrics["val_roc_auc_mean"]
    if roc_gap > 0.10:
        logger.warning(
            f"{model_name}: Train-Val ROC-AUC gap = {roc_gap:.3f} → potential overfitting"
        )
    elif metrics["val_roc_auc_mean"] < 0.70:
        logger.warning(
            f"{model_name}: Val ROC-AUC = {metrics['val_roc_auc_mean']:.3f} → potential underfitting"
        )

    return metrics


def train_final_model(
    estimator,
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "Model",
) -> Pipeline:
    """
    Train the final model on full training data (after CV selects best model).
    Returns a fitted sklearn Pipeline (preprocessor + classifier).
    """
    # For XGBoost with class_weight strategy, set scale_pos_weight dynamically
    if hasattr(estimator, "set_params") and not isinstance(estimator, ImbPipeline):
        if isinstance(estimator, xgb.XGBClassifier):
            n_neg = (y_train == 0).sum()
            n_pos = (y_train == 1).sum()
            scale_pos_weight = n_neg / n_pos
            estimator.set_params(scale_pos_weight=scale_pos_weight)
            logger.info(f"XGBoost scale_pos_weight set to {scale_pos_weight:.2f}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])

    logger.info(f"Training final {model_name} on {len(X_train)} samples...")
    start = time.time()
    pipeline.fit(X_train, y_train)
    logger.info(f"{model_name} trained in {time.time() - start:.1f}s")

    return pipeline


def compare_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
) -> pd.DataFrame:
    """
    Side-by-side comparison of class_weight vs SMOTE for all models.
    Used for the bias-variance / strategy selection section of the report.
    """
    results = []

    for strategy in ["class_weight", "smote"]:
        models = get_baseline_models(imbalance_strategy=strategy)
        for name, estimator in models.items():
            metrics = cross_validate_model(
                estimator, X_train, y_train, preprocessor, model_name=name
            )
            results.append({
                "strategy": strategy,
                "model": name.replace("_SMOTE", ""),
                "roc_auc": metrics["val_roc_auc_mean"],
                "pr_auc": metrics["val_average_precision_mean"],
                "f1": metrics["val_f1_mean"],
                "precision": metrics["val_precision_mean"],
                "recall": metrics["val_recall_mean"],
                "train_roc_auc": metrics["train_roc_auc_mean"],
            })

    return pd.DataFrame(results).sort_values("roc_auc", ascending=False)
