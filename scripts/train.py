"""
scripts/train.py
─────────────────
Main training orchestration script.
Run this to train all models, compare them, and deploy the best one.

Usage:
    python scripts/train.py
    python scripts/train.py --tune          # Run Optuna hyperparameter tuning
    python scripts/train.py --strategy smote  # Use SMOTE instead of class_weight
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Setup path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import load_raw_data
from src.data.data_validator import validate_dataframe
from src.data.preprocessor import prepare_data, build_preprocessing_pipeline, train_val_test_split
from src.models.trainer import (
    get_baseline_models, cross_validate_model, train_final_model, compare_imbalance_strategies
)
from src.models.tuner import tune_model, build_tuned_model
from src.evaluation.evaluator import (
    evaluate_model, find_optimal_threshold, bias_variance_report,
    plot_evaluation_dashboard, print_classification_report
)
from src.explainability.explainer import (
    get_shap_explainer, compute_shap_values,
    plot_global_importance, plot_shap_beeswarm, explain_single_prediction
)
from src.models.model_registry import ModelRegistry
from src.features.feature_engineer import get_feature_importance_names
from configs.config import ARTIFACTS_DIR, NUMERIC_FEATURES, IMBALANCE_STRATEGY

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/training.log"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train churn prediction models")
    parser.add_argument("--tune", action="store_true", help="Run Optuna hyperparameter tuning")
    parser.add_argument("--strategy", default=IMBALANCE_STRATEGY,
                        choices=["class_weight", "smote"],
                        help="Imbalance handling strategy")
    parser.add_argument("--data", default=None, help="Path to CSV data file")
    parser.add_argument("--compare-imbalance", action="store_true",
                        help="Compare class_weight vs SMOTE strategies")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Customer Churn Prediction — Training Pipeline")
    logger.info("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    # 1. DATA LOADING & VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[1/8] Loading and validating data")
    df = load_raw_data(filepath=args.data)

    validation_report = validate_dataframe(df, require_target=True)
    if not validation_report.passed:
        logger.error(f"Data validation FAILED: {validation_report.errors}")
        sys.exit(1)

    logger.info(f"Dataset: {len(df)} rows | Churn rate: {df['churn'].mean():.1%}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. FEATURE ENGINEERING & PREPROCESSING
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[2/8] Feature engineering and preprocessing")
    X, y, winsorizer = prepare_data(df, fit_winsorizer=True)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    logger.info(f"Feature matrix: {X_train.shape[1]} features | {X_train.shape[0]} training samples")

    # ─────────────────────────────────────────────────────────────────────────
    # 3. COMPARE IMBALANCE STRATEGIES (optional)
    # ─────────────────────────────────────────────────────────────────────────
    if args.compare_imbalance:
        logger.info("\n[3/8] Comparing imbalance strategies (class_weight vs SMOTE)")
        preprocessor = build_preprocessing_pipeline()
        comparison_df = compare_imbalance_strategies(X_train, y_train, preprocessor)
        logger.info(f"\nImbalance Strategy Comparison:\n{comparison_df.to_string(index=False)}")
    else:
        logger.info(f"\n[3/8] Using imbalance strategy: {args.strategy}")

    # ─────────────────────────────────────────────────────────────────────────
    # 4. BASELINE MODEL TRAINING + CROSS-VALIDATION
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"\n[4/8] Baseline cross-validation (strategy: {args.strategy})")
    preprocessor = build_preprocessing_pipeline()
    baseline_models = get_baseline_models(imbalance_strategy=args.strategy)

    cv_results = {}
    for name, estimator in baseline_models.items():
        cv_results[name] = cross_validate_model(
            estimator, X_train, y_train, preprocessor, model_name=name
        )

    # Print baseline comparison
    logger.info("\nBaseline Model Comparison:")
    header = f"{'Model':<25} {'Val ROC-AUC':>12} {'Val PR-AUC':>10} {'Val F1':>8} {'Train-Val Gap':>14}"
    logger.info(header)
    logger.info("-" * 75)
    for name, metrics in cv_results.items():
        gap = metrics['train_roc_auc_mean'] - metrics['val_roc_auc_mean']
        logger.info(
            f"{name:<25} "
            f"{metrics['val_roc_auc_mean']:>12.4f} "
            f"{metrics['val_average_precision_mean']:>10.4f} "
            f"{metrics['val_f1_mean']:>8.4f} "
            f"{gap:>14.4f}"
        )

    # Select best baseline model
    best_model_name = max(cv_results, key=lambda k: cv_results[k]["val_roc_auc_mean"])
    logger.info(f"\nBest baseline model: {best_model_name} "
                f"(ROC-AUC: {cv_results[best_model_name]['val_roc_auc_mean']:.4f})")

    # ─────────────────────────────────────────────────────────────────────────
    # 5. HYPERPARAMETER TUNING (optional)
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"\n[5/8] Hyperparameter tuning: {'Enabled' if args.tune else 'Skipped'}")

    # Always tune XGBoost (best performer for churn tasks)
    CHAMPION_MODEL = "XGBoost"

    if args.tune:
        tune_result = tune_model(
            model_name=CHAMPION_MODEL,
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
        )
        best_params = tune_result["best_params"]
        tuned_clf = build_tuned_model(CHAMPION_MODEL, best_params, y_train=y_train)
        logger.info(
            f"Tuned {CHAMPION_MODEL}: CV ROC-AUC {cv_results[CHAMPION_MODEL]['val_roc_auc_mean']:.4f} → "
            f"{tune_result['best_value']:.4f} "
            f"(+{tune_result['best_value'] - cv_results[CHAMPION_MODEL]['val_roc_auc_mean']:+.4f})"
        )
    else:
        # Use baseline params
        tuned_clf = baseline_models[CHAMPION_MODEL]
        logger.info("Using baseline hyperparameters (pass --tune for Optuna optimization)")

    # ─────────────────────────────────────────────────────────────────────────
    # 6. TRAIN FINAL MODEL
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"\n[6/8] Training final {CHAMPION_MODEL} on full training set")
    final_pipeline = train_final_model(
        tuned_clf, preprocessor, X_train, y_train, model_name=CHAMPION_MODEL
    )

    # Bias-variance analysis
    bv = bias_variance_report(final_pipeline, X_train, y_train, X_val, y_val,
                              model_name=CHAMPION_MODEL)

    # ─────────────────────────────────────────────────────────────────────────
    # 7. EVALUATION
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[7/8] Evaluating on held-out test set")

    # Threshold tuning on validation set (never test set to avoid leakage)
    val_probs = final_pipeline.predict_proba(X_val)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(
        y_val.values, val_probs, strategy="cost", fn_cost=3.0, fp_cost=1.0
    )
    logger.info(f"Business-optimal threshold: {optimal_threshold:.2f}")

    test_metrics = evaluate_model(
        final_pipeline, X_test, y_test,
        threshold=optimal_threshold, split_name="Test"
    )
    print_classification_report(
        final_pipeline, X_test, y_test,
        threshold=optimal_threshold, model_name=CHAMPION_MODEL
    )

    # Save evaluation plots
    plot_evaluation_dashboard(
        final_pipeline, X_test, y_test,
        model_name=CHAMPION_MODEL, threshold=optimal_threshold,
        save_path=ARTIFACTS_DIR / "evaluation_dashboard.png"
    )

    # ─────────────────────────────────────────────────────────────────────────
    # 8. SHAP EXPLAINABILITY
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\n[8/8] Computing SHAP explanations")

    try:
        explainer, preproc = get_shap_explainer(final_pipeline, X_train.sample(200, random_state=42))
        shap_values, X_transformed = compute_shap_values(
            explainer, preproc, X_val, max_samples=300
        )

        fitted_preprocessor = final_pipeline.named_steps["preprocessor"]
        feature_names = get_feature_importance_names(fitted_preprocessor)

        plot_global_importance(
            shap_values, feature_names, top_n=15,
            save_path=ARTIFACTS_DIR / "shap_global_importance.png"
        )
        plot_shap_beeswarm(
            shap_values, X_transformed, feature_names, top_n=15,
            save_path=ARTIFACTS_DIR / "shap_beeswarm.png"
        )

        # Local explanation for a high-risk customer
        high_risk_idx = val_probs.argsort()[-1]
        X_sample = X_val.iloc[[high_risk_idx]]
        local_result = explain_single_prediction(
            final_pipeline, X_sample, explainer, feature_names,
            customer_id="HIGH_RISK_SAMPLE",
            save_path=ARTIFACTS_DIR / "shap_local_explanation.png"
        )
        logger.info(
            f"High-risk customer churn prob: {local_result['churn_probability']:.1%}"
        )

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}. Skipping explainability plots.")

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL REGISTRY — REGISTER + PROMOTE
    # ─────────────────────────────────────────────────────────────────────────
    logger.info("\nRegistering model in registry")
    registry = ModelRegistry()

    try:
        feature_names = get_feature_importance_names(
            final_pipeline.named_steps["preprocessor"]
        )
    except Exception:
        feature_names = NUMERIC_FEATURES

    version = registry.register_model(
        pipeline=final_pipeline,
        preprocessor_winsorizer=winsorizer,
        metrics=test_metrics,
        model_name=CHAMPION_MODEL,
        threshold=optimal_threshold,
        feature_names=feature_names,
        notes="Initial training run",
    )
    registry.promote_to_champion(version)

    # ─────────────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"""
{'='*60}
Training Complete — Champion Model: {CHAMPION_MODEL} ({version})
{'='*60}
Test Set Performance:
  ROC-AUC  : {test_metrics['roc_auc']:.4f}
  PR-AUC   : {test_metrics['pr_auc']:.4f}
  F1       : {test_metrics['f1']:.4f}
  Precision: {test_metrics['precision']:.4f}
  Recall   : {test_metrics['recall']:.4f}
  Threshold: {optimal_threshold:.2f}

Bias-Variance:
  Train ROC-AUC: {bv['train_roc_auc']:.4f}
  Val   ROC-AUC: {bv['val_roc_auc']:.4f}
  Diagnosis    : {bv['diagnosis']}

Artifacts saved to: {ARTIFACTS_DIR}
API ready: uvicorn api.app:app --host 0.0.0.0 --port 8000
{'='*60}
""")


if __name__ == "__main__":
    main()
