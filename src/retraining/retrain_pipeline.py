"""
retraining/retrain_pipeline.py
────────────────────────────────
Automated retraining pipeline.

Triggered by:
  - Monitoring system (drift alerts)
  - Scheduled job (cron / Airflow DAG)
  - Manual trigger (data scientist)

Safety checks:
  - New model must beat champion by a minimum margin (0.5% ROC-AUC)
    before promotion. This prevents "accidental degradation" from noisy
    new data from silently replacing a good model.
  - All validation runs on held-out test set that neither the champion
    nor challenger has seen during training.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.data_loader import load_raw_data
from src.data.data_validator import validate_dataframe
from src.data.preprocessor import prepare_data, build_preprocessing_pipeline, train_val_test_split
from src.models.trainer import get_baseline_models, train_final_model, cross_validate_model
from src.models.tuner import tune_model, build_tuned_model
from src.evaluation.evaluator import evaluate_model, find_optimal_threshold, bias_variance_report
from src.models.model_registry import ModelRegistry
from configs.config import RANDOM_STATE, ARTIFACTS_DIR, LOGS_DIR

logger = logging.getLogger(__name__)

# Minimum improvement required to promote challenger over champion
MIN_IMPROVEMENT_THRESHOLD = 0.005  # 0.5% ROC-AUC improvement


def run_retraining_pipeline(
    new_data_path: Optional[Path] = None,
    tune_hyperparameters: bool = True,
    force_promote: bool = False,
    notes: str = "",
) -> Dict:
    """
    Full retraining pipeline:
      1. Load + validate new data
      2. Preprocess + feature engineer
      3. Train challenger model (with optional tuning)
      4. Evaluate challenger vs champion
      5. Promote if challenger is better
      6. Register new version

    Args:
        new_data_path: Path to new CSV data. If None, regenerates synthetic data.
        tune_hyperparameters: Whether to run Optuna tuning (slower but better)
        force_promote: Skip improvement check and always promote (use with caution)
        notes: Human-readable notes about this retraining run

    Returns:
        Report dict with metrics and promotion decision
    """
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    logger.info(f"\n{'='*60}\nRetraining Run: {run_id}\n{'='*60}")

    report = {
        "run_id": run_id,
        "started_at": datetime.utcnow().isoformat(),
        "notes": notes,
    }

    # ── Step 1: Load and validate data ────────────────────────────────────────
    logger.info("Step 1: Loading and validating data")
    df = load_raw_data(filepath=new_data_path)
    validation = validate_dataframe(df, require_target=True)

    if not validation.passed:
        logger.error(f"Data validation failed: {validation.errors}")
        report["status"] = "FAILED"
        report["failure_reason"] = f"Data validation: {validation.errors}"
        return report

    # ── Step 2: Prepare data ──────────────────────────────────────────────────
    logger.info("Step 2: Feature engineering and preprocessing")
    X, y, winsorizer = prepare_data(df, fit_winsorizer=True)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    preprocessor = build_preprocessing_pipeline()

    # ── Step 3: Train challenger ──────────────────────────────────────────────
    logger.info("Step 3: Training challenger model")

    # We use XGBoost as primary model (best performer in baseline)
    TARGET_MODEL = "XGBoost"

    if tune_hyperparameters:
        logger.info(f"Running Optuna hyperparameter tuning for {TARGET_MODEL}")
        tune_result = tune_model(
            model_name=TARGET_MODEL,
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
        )
        best_params = tune_result["best_params"]
        challenger_clf = build_tuned_model(TARGET_MODEL, best_params, y_train=y_train)
        tuning_roc_auc = tune_result["best_value"]
        logger.info(f"Tuning complete. Best CV ROC-AUC: {tuning_roc_auc:.4f}")
    else:
        # Use default params — faster for testing
        from xgboost import XGBClassifier
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        challenger_clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=n_neg/n_pos,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        )

    challenger_pipeline = train_final_model(
        challenger_clf, preprocessor, X_train, y_train, model_name="Challenger_XGBoost"
    )

    # ── Step 4: Threshold tuning ──────────────────────────────────────────────
    logger.info("Step 4: Threshold tuning on validation set")
    val_probs = challenger_pipeline.predict_proba(X_val)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(
        y_val.values, val_probs, strategy="cost", fn_cost=3.0, fp_cost=1.0
    )

    # ── Step 5: Evaluate challenger on test set ───────────────────────────────
    logger.info("Step 5: Evaluating challenger on held-out test set")
    challenger_metrics = evaluate_model(
        challenger_pipeline, X_test, y_test,
        threshold=optimal_threshold, split_name="Challenger_Test"
    )

    # Bias-variance analysis
    bv_report = bias_variance_report(
        challenger_pipeline, X_train, y_train, X_val, y_val,
        model_name="Challenger_XGBoost"
    )

    # ── Step 6: Compare with champion ─────────────────────────────────────────
    logger.info("Step 6: Comparing challenger vs champion")
    registry = ModelRegistry()
    promote = False
    champion_roc_auc = None

    try:
        champion_pipeline, champion_winsorizer, champion_metadata = registry.load_champion()
        champion_metrics = evaluate_model(
            champion_pipeline, X_test, y_test,
            threshold=champion_metadata.get("threshold", 0.40),
            split_name="Champion_Test"
        )
        champion_roc_auc = champion_metrics["roc_auc"]
        improvement = challenger_metrics["roc_auc"] - champion_roc_auc

        logger.info(
            f"Champion ROC-AUC: {champion_roc_auc:.4f} | "
            f"Challenger ROC-AUC: {challenger_metrics['roc_auc']:.4f} | "
            f"Improvement: {improvement:+.4f}"
        )

        if force_promote or improvement >= MIN_IMPROVEMENT_THRESHOLD:
            promote = True
            promote_reason = (
                "Force promote" if force_promote
                else f"Improvement {improvement:+.4f} exceeds threshold {MIN_IMPROVEMENT_THRESHOLD}"
            )
        else:
            promote_reason = (
                f"Improvement {improvement:+.4f} below threshold {MIN_IMPROVEMENT_THRESHOLD}. "
                f"Champion retained."
            )

    except FileNotFoundError:
        # No champion exists — first deployment
        logger.info("No existing champion. Challenger will be promoted automatically.")
        promote = True
        promote_reason = "First model deployment"
        improvement = None

    # ── Step 7: Register and (optionally) promote ─────────────────────────────
    logger.info("Step 7: Registering new model version")
    from src.features.feature_engineer import get_feature_importance_names
    preprocessor.fit(X_train, y_train)  # ensure fitted
    try:
        feature_names = get_feature_importance_names(preprocessor)
    except Exception:
        feature_names = []

    version = registry.register_model(
        pipeline=challenger_pipeline,
        preprocessor_winsorizer=winsorizer,
        metrics=challenger_metrics,
        model_name="XGBoost_Challenger",
        threshold=optimal_threshold,
        feature_names=feature_names,
        notes=f"Run {run_id}: {notes}",
    )

    if promote:
        registry.promote_to_champion(version)
        logger.info(f"✅ {version} promoted to champion. Reason: {promote_reason}")
    else:
        logger.info(f"⏸️  {version} registered as candidate. Reason: {promote_reason}")

    # ── Final report ──────────────────────────────────────────────────────────
    report.update({
        "status": "SUCCESS",
        "new_version": version,
        "promoted": promote,
        "promote_reason": promote_reason,
        "challenger_metrics": challenger_metrics,
        "champion_roc_auc": champion_roc_auc,
        "improvement": improvement,
        "optimal_threshold": optimal_threshold,
        "bias_variance": bv_report,
        "completed_at": datetime.utcnow().isoformat(),
    })

    # Save run report
    report_path = LOGS_DIR / f"retrain_report_{run_id}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Retraining report saved to {report_path}")
    logger.info(f"\n{'='*60}\nRetraining Complete: {run_id}\n{'='*60}")

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = run_retraining_pipeline(
        tune_hyperparameters=False,  # Fast mode for testing
        notes="Scheduled weekly retrain",
    )
    print(f"\nStatus: {result['status']}")
    print(f"Promoted: {result['promoted']}")
    if result.get("challenger_metrics"):
        print(f"Test ROC-AUC: {result['challenger_metrics']['roc_auc']:.4f}")
