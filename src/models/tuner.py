"""
models/tuner.py
────────────────
Bayesian hyperparameter optimization using Optuna.

Why Optuna over GridSearchCV?
  - GridSearch: exhaustive, scales as O(n^k) where k = num params.
    Works well for small search spaces (<100 combinations).
  - Optuna: uses TPE (Tree-structured Parzen Estimator) to sample
    promising regions first. Finds good solutions in 10-20% of the
    evaluations a full grid search would need.
  - Optuna also supports pruning: stops unpromising trials early.

Trade-off: Optuna is non-deterministic; GridSearch is reproducible.
We fix the Optuna seed for reproducibility.
"""

import numpy as np
import logging
import warnings
warnings.filterwarnings("ignore")

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import CV_FOLDS, RANDOM_STATE, OPTUNA_TRIALS, OPTUNA_TIMEOUT

logger = logging.getLogger(__name__)


def _make_cv(n_folds: int = CV_FOLDS):
    return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)


# ── Objective functions per model ─────────────────────────────────────────────

def _lr_objective(trial, X_train, y_train, preprocessor):
    """
    Search space for Logistic Regression.
    C: inverse regularization strength. Low C = strong regularization = less overfit.
    solver: lbfgs efficient for multiclass; saga for large datasets + L1.
    """
    C = trial.suggest_float("C", 1e-3, 10.0, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
    solver = "saga" if penalty == "l1" else "lbfgs"

    clf = LogisticRegression(
        C=C, penalty=penalty, solver=solver,
        max_iter=1000, class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=_make_cv(),
                              scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def _rf_objective(trial, X_train, y_train, preprocessor):
    """
    Search space for Random Forest.
    max_features: fraction of features per split — controls variance.
    min_samples_leaf: min samples at leaf — controls overfit.
    """
    n_estimators = trial.suggest_int("n_estimators", 50, 400)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_float("max_features", 0.3, 1.0)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=_make_cv(),
                              scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def _xgb_objective(trial, X_train, y_train, preprocessor):
    """
    Search space for XGBoost.
    subsample + colsample_bytree: row/column sampling — key regularization.
    reg_alpha (L1) + reg_lambda (L2): explicit regularization weights.
    min_child_weight: min sum of instance weights in a child — anti-overfit.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "scale_pos_weight": scale_pos_weight,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "verbosity": 0,
    }

    clf = xgb.XGBClassifier(**params)
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
    scores = cross_val_score(pipeline, X_train, y_train, cv=_make_cv(),
                              scoring="roc_auc", n_jobs=1)  # XGBoost already parallel
    return scores.mean()


# ── Public API ────────────────────────────────────────────────────────────────

OBJECTIVE_MAP = {
    "LogisticRegression": _lr_objective,
    "RandomForest": _rf_objective,
    "XGBoost": _xgb_objective,
}


def tune_model(
    model_name: str,
    X_train,
    y_train,
    preprocessor,
    n_trials: int = OPTUNA_TRIALS,
    timeout: int = OPTUNA_TIMEOUT,
) -> dict:
    """
    Run Optuna study for a given model.

    Returns:
        best_params: dict of best hyperparameters
        best_value: best CV ROC-AUC score
        study: Optuna study object (for visualization)
    """
    if model_name not in OBJECTIVE_MAP:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(OBJECTIVE_MAP)}")

    objective_fn = OBJECTIVE_MAP[model_name]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    logger.info(f"Starting Optuna tuning for {model_name} ({n_trials} trials, timeout={timeout}s)")

    study.optimize(
        lambda trial: objective_fn(trial, X_train, y_train, preprocessor),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    logger.info(
        f"{model_name} tuning complete | "
        f"Best ROC-AUC: {study.best_value:.4f} | "
        f"Best params: {study.best_params}"
    )

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "study": study,
        "n_trials_completed": len(study.trials),
    }


def build_tuned_model(model_name: str, best_params: dict, y_train=None):
    """
    Instantiate a model with the best parameters found by Optuna.
    """
    if model_name == "LogisticRegression":
        solver = "saga" if best_params.get("penalty") == "l1" else "lbfgs"
        return LogisticRegression(
            **{k: v for k, v in best_params.items()},
            solver=solver,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    elif model_name == "RandomForest":
        return RandomForestClassifier(
            **best_params,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    elif model_name == "XGBoost":
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        return xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=n_neg / n_pos,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
