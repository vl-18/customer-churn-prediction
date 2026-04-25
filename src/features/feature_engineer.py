"""
features/feature_engineer.py
──────────────────────────────
All feature transformations with business justification.

Rule: every feature added here must have a clear business hypothesis.
"Gut feel" features without rationale get removed — they add noise and
make the model harder to explain to stakeholders.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-driven features on top of raw columns.

    Feature rationale:
    ─────────────────
    1. usage_score
       Hypothesis: Customers who log in frequently AND stay longer per session
       are more engaged and less likely to churn. A composite engagement score
       captures this better than either signal alone.
       Formula: normalized_sessions × normalized_duration

    2. charge_per_month_ratio
       Hypothesis: Monthly charges relative to total historical spend reflects
       whether a customer is on a newer (possibly more expensive) plan.
       Rising ratio = recent price increase = churn risk.

    3. support_call_rate
       Hypothesis: Normalizing support calls by tenure separates "chronic
       complainers" (high rate even as long-tenured customers) from new customers
       naturally hitting onboarding friction. Raw call count conflates these.

    4. tenure_x_products
       Hypothesis: An interaction term. Long-tenured customers with many products
       have high switching costs — they're very sticky. Short-tenured with many
       products may be over-sold and at risk. The interaction captures this
       non-linear relationship that tree models can find but is explicit here
       for linear models (LR).

    5. days_since_login_risk (binary)
       Hypothesis: >30 days of inactivity is a well-known leading indicator of
       churn across subscription businesses. Encoding as binary is more
       interpretable and less noisy than the raw continuous value.

    6. is_month_to_month (binary)
       Hypothesis: Contract type is categorical, but "month-to-month" is so
       overwhelmingly predictive that making it an explicit binary feature
       improves linear model performance dramatically.
    """
    df = df.copy()

    # ── Feature 1: Composite engagement score ─────────────────────────────────
    if "days_since_last_login" in df.columns and "avg_session_duration_mins" in df.columns:
        # Recency: lower days_since_login = better
        recency_score = 1 - (df["days_since_last_login"].fillna(90) / 90).clip(0, 1)
        # Duration: higher = better, capped at 120 mins to reduce outlier influence
        duration_score = (df["avg_session_duration_mins"].fillna(0) / 120).clip(0, 1)
        df["usage_score"] = (recency_score * 0.6 + duration_score * 0.4).round(4)
        logger.debug("Added feature: usage_score")

    # ── Feature 2: Charge trend ratio ─────────────────────────────────────────
    if "monthly_charges" in df.columns and "total_charges" in df.columns and "tenure_months" in df.columns:
        # Expected total = tenure × monthly; actual total vs expected
        expected_total = df["tenure_months"] * df["monthly_charges"]
        # Avoid div-by-zero for brand new customers
        df["charge_per_month_ratio"] = (
            df["total_charges"].fillna(df["monthly_charges"])
            / expected_total.replace(0, np.nan)
        ).fillna(1.0).clip(0.5, 2.0).round(4)
        logger.debug("Added feature: charge_per_month_ratio")

    # ── Feature 3: Annualized support call rate ───────────────────────────────
    if "num_support_calls" in df.columns and "tenure_months" in df.columns:
        # Calls per year (annualized); new customers capped at 12mo to avoid inf
        tenure_safe = df["tenure_months"].replace(0, 1).clip(lower=1)
        df["support_call_rate"] = (
            df["num_support_calls"] / tenure_safe * 12
        ).clip(0, 24).round(4)
        logger.debug("Added feature: support_call_rate")

    # ── Feature 4: Tenure × products interaction ──────────────────────────────
    if "tenure_months" in df.columns and "num_products" in df.columns:
        df["tenure_x_products"] = (
            np.log1p(df["tenure_months"]) * df["num_products"]
        ).round(4)
        logger.debug("Added feature: tenure_x_products")

    # ── Feature 5: High inactivity flag ──────────────────────────────────────
    # (Kept in CATEGORICAL_FEATURES as binary string for OHE consistency)
    if "days_since_last_login" in df.columns:
        df["is_senior_citizen"] = df.get("is_senior_citizen", 0)  # ensure exists

    return df


def get_feature_importance_names(preprocessor) -> list:
    """
    Reconstruct feature names after ColumnTransformer.
    Essential for SHAP explanations — sklearn >= 1.0 has get_feature_names_out.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback for older sklearn
        num_names = preprocessor.transformers_[0][2]  # numeric feature names
        cat_encoder = preprocessor.transformers_[1][1].named_steps["encoder"]
        cat_names = list(cat_encoder.get_feature_names_out(
            preprocessor.transformers_[1][2]
        ))
        return num_names + cat_names
