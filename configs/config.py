"""
Central configuration for the churn prediction system.
All paths, hyperparameters, and thresholds live here —
never hardcoded in business logic.
"""
from pathlib import Path
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DRIFT_DATA_DIR = DATA_DIR / "drift"
MODEL_DIR = BASE_DIR / "models"
REGISTRY_DIR = MODEL_DIR / "registry"
ARTIFACTS_DIR = MODEL_DIR / "artifacts"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist at import time
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, DRIFT_DATA_DIR, REGISTRY_DIR, ARTIFACTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Data ────────────────────────────────────────────────────────────────────
TARGET_COL = "churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1  # of training data

# ─── Feature groups ──────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "num_support_calls",
    "days_since_last_login",
    "avg_session_duration_mins",
    "usage_score",            # engineered
    "charge_per_month_ratio", # engineered
    "support_call_rate",      # engineered
    "tenure_x_products",      # engineered interaction
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "internet_service",
    "has_phone_service",
    "has_online_backup",
    "has_tech_support",
    "gender",
    "is_senior_citizen",
]

# ─── Model ───────────────────────────────────────────────────────────────────
CV_FOLDS = 5
SCORING_METRIC = "roc_auc"

# Business-driven threshold: we prefer catching churners (recall)
# over perfect precision. FN (missed churner) costs ~3x FP (false alarm).
# Threshold tuned in evaluation module.
DEFAULT_THRESHOLD = 0.40

# ─── Imbalance handling ──────────────────────────────────────────────────────
# "smote" | "class_weight"
IMBALANCE_STRATEGY = "class_weight"

# ─── Hyperparameter search ───────────────────────────────────────────────────
OPTUNA_TRIALS = 50
OPTUNA_TIMEOUT = 300  # seconds

# ─── Monitoring ──────────────────────────────────────────────────────────────
DRIFT_PSI_THRESHOLD = 0.2       # PSI > 0.2 → significant drift → retrain
DRIFT_KS_PVALUE_THRESHOLD = 0.05
PREDICTION_SHIFT_THRESHOLD = 0.1  # mean predicted prob shifts > 10pp → alert
MONITORING_WINDOW_DAYS = 7

# ─── Model Registry ──────────────────────────────────────────────────────────
CHAMPION_MODEL_NAME = "champion_model.joblib"
PIPELINE_NAME = "preprocessing_pipeline.joblib"
METADATA_NAME = "model_metadata.json"
