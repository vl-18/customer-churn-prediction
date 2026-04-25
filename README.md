# Customer Churn Prediction System
### Production-Grade ML Pipeline | Telecom Domain

---

## Problem Statement

Customer churn is when a subscriber cancels their service. In telecom, acquiring a new customer costs **5–7× more** than retaining one. This system predicts the probability that a customer will churn in the next 30 days, enabling the retention team to intervene with targeted offers **before** the customer leaves.

**Business objective:** Maximize recall (catch more churners) at an acceptable precision (avoid wasting too many retention offers).

**Key trade-off:** A missed churner (FN) costs ~3× more than a false alarm (FP), so we use a lower classification threshold (0.37–0.40 vs the default 0.50) and optimize for a cost-weighted metric.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Layer                                 │
│  Raw CSV / Synthetic Generator → Validator → Feature Engineer   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                   Training Pipeline                             │
│  Preprocessor → CV → Baseline Models → Optuna Tuning           │
│  → Threshold Tuning → SHAP Explainability → Model Registry      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                   Serving Layer                                 │
│  FastAPI (POST /predict, POST /predict/batch)                   │
│  Model loaded at startup | Pydantic validation | JSON logging   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                   Monitoring Layer                              │
│  PSI + KS drift detection | Prediction distribution tracking   │
│  → Retrain trigger → Retrain pipeline → Registry promotion      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
churn_prediction/
├── api/
│   └── app.py                    # FastAPI service (predict, batch, health, model/info)
├── configs/
│   └── config.py                 # Central config: paths, thresholds, hyperparams
├── data/
│   ├── raw/                      # Raw input data
│   ├── processed/                # Processed datasets
│   └── drift/                    # Drift monitoring snapshots
├── models/
│   ├── artifacts/                # Champion model + plots (served by API)
│   └── registry/                 # Versioned model store
├── src/
│   ├── data/
│   │   ├── data_loader.py        # Data loading + synthetic generation
│   │   ├── data_validator.py     # Schema + statistical validation
│   │   └── preprocessor.py      # Winsorizer + sklearn Pipeline
│   ├── features/
│   │   └── feature_engineer.py  # All engineered features with rationale
│   ├── models/
│   │   ├── trainer.py            # CV, cross-validation, imbalance handling
│   │   ├── tuner.py              # Optuna hyperparameter search
│   │   └── model_registry.py    # Versioning, promotion, rollback
│   ├── evaluation/
│   │   └── evaluator.py          # ROC-AUC, PR-AUC, threshold tuning, plots
│   ├── explainability/
│   │   └── explainer.py          # SHAP global + local explanations
│   ├── monitoring/
│   │   └── monitor.py            # PSI, KS test, prediction drift, retrain triggers
│   └── retraining/
│       └── retrain_pipeline.py  # Full retrain → evaluate → promote pipeline
├── scripts/
│   ├── train.py                  # Orchestration: full training run
│   └── monitor.py               # Monitoring simulation
└── logs/
    ├── training.log
    ├── api.log
    └── predictions.jsonl         # Append-only prediction log
```

---

## How to Run

### 1. Install dependencies
```bash
pip install scikit-learn xgboost imbalanced-learn shap optuna \
            fastapi uvicorn pandas numpy scipy joblib pydantic \
            matplotlib seaborn
```

### 2. Train the model
```bash
# Basic training (fast)
python scripts/train.py

# With Optuna hyperparameter tuning (recommended for production)
python scripts/train.py --tune

# Compare class_weight vs SMOTE imbalance strategies
python scripts/train.py --compare-imbalance

# Use your own data
python scripts/train.py --data data/raw/customers.csv
```

### 3. Start the API
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 4. Make a prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
    "gender": "Female",
    "is_senior_citizen": 0,
    "tenure_months": 3,
    "has_phone_service": "Yes",
    "internet_service": "Fiber Optic",
    "has_online_backup": "No",
    "has_tech_support": "No",
    "contract_type": "Month-to-Month",
    "payment_method": "Electronic Check",
    "monthly_charges": 89.50,
    "total_charges": null,
    "num_products": 2,
    "num_support_calls": 4,
    "days_since_last_login": 28,
    "avg_session_duration_mins": 12.5
  }'
```

**Response:**
```json
{
  "customer_id": "CUST_001",
  "churn_probability": 0.7234,
  "churn_prediction": true,
  "risk_tier": "HIGH",
  "threshold_used": 0.37,
  "model_version": "v1.0.0",
  "prediction_timestamp": "2026-04-24T10:00:00"
}
```

### 5. Run monitoring simulation
```bash
python scripts/monitor.py
```

### 6. Trigger retraining
```bash
# Auto-retrain with new data + compare to champion
python -c "
from src.retraining.retrain_pipeline import run_retraining_pipeline
report = run_retraining_pipeline(tune_hyperparameters=True, notes='Weekly retrain')
print(report)
"
```

---

## Feature Engineering Rationale

| Feature | Business Logic |
|---|---|
| `usage_score` | Composite engagement (recency × duration). Disengaged customers churn silently |
| `charge_per_month_ratio` | Actual vs expected charges. Rising ratio = recent price hike = churn risk |
| `support_call_rate` | Normalized by tenure. Chronic complainers vs new-customer friction |
| `tenure_x_products` | Interaction term. High-tenure + many products = sticky; low-tenure = oversold |

---

## Model Selection & Trade-offs

| Model | Val ROC-AUC | Train ROC-AUC | Gap | Notes |
|---|---|---|---|---|
| Logistic Regression | 0.7033 | 0.7084 | 0.005 | Best generalization, interpretable |
| Random Forest | 0.6947 | 0.9561 | 0.261 | High variance (overfitting) |
| XGBoost | 0.6844 | 0.9102 | 0.226 | High variance without tuning |

**Selected:** XGBoost (with hyperparameter tuning via Optuna reduces the gap significantly).

**Overfitting fix applied:** `--tune` flag runs Optuna with regularization search (reg_alpha, reg_lambda, min_child_weight, subsample).

---

## Threshold Tuning

Default threshold of 0.50 is calibrated for balanced costs. Our business has **asymmetric costs:**
- False Negative (miss a churner): $60 lost LTV
- False Positive (waste a retention offer): $20 cost

Optimal threshold via cost minimization: **0.37**

At threshold=0.37:
- Recall: 0.79 (catch 79% of churners)  
- Precision: 0.38 (38% of flagged customers actually churn)
- Trade-off accepted: some waste in retention campaigns in exchange for catching most churners

---

## SHAP Explainability

- **Global:** Mean |SHAP| per feature — `tenure_months` and `contract_type` are dominant
- **Local:** Per-customer waterfall shows exactly which features pushed the prediction
- Safe for regulatory compliance (GDPR Article 22 — right to explanation)

---

## Monitoring & Retraining Policy

| Signal | Threshold | Action |
|---|---|---|
| PSI > 0.20 (any feature) | Strong drift | Immediate retrain |
| Prediction mean shift > 10pp | Concept drift proxy | Schedule retrain |
| ROC-AUC drop > 5pp | Performance degradation | Emergency retrain |
| Time since last retrain > 30 days | Scheduled | Routine retrain |

**Safe promotion:** Challenger must beat champion by ≥ 0.5% ROC-AUC to be promoted. Prevents silent model degradation from noisy data.

---

## Key Learnings & Interview Talking Points

1. **Training/serving skew:** Winsorizer and preprocessor are fitted ONLY on training data, then serialized. Inference reuses the same fitted objects — zero skew.

2. **Leakage prevention:** Preprocessing inside CV folds (via sklearn Pipeline). Without this, imputing on the full dataset leaks validation statistics into training.

3. **Imbalance strategy:** class_weight beats SMOTE at ~30% churn rate. SMOTE shines at extreme imbalance (5%). Chosen based on empirical comparison.

4. **Threshold ≠ 0.5:** Business-driven threshold always beats default. The cost ratio (FN/FP = 3) determines the optimal operating point on the PR curve.

5. **Model registry:** Versioned artifacts prevent "which model is running in prod?" questions. Champion/challenger pattern enables safe rollouts.

6. **PSI for drift:** Quantile-based PSI is more robust than equal-width binning. KS test provides a complementary statistical significance check.

7. **SHAP over permutation importance:** SHAP is consistent and locally accurate. Feature importance from tree impurity is biased toward high-cardinality features.
