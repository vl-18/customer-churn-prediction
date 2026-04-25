"""
data/data_loader.py
───────────────────
Generates a realistic telecom-style churn dataset OR loads an existing CSV.

Design decisions:
  - Synthetic data mimics real-world class imbalance (~27% churn rate)
  - Correlations baked in (e.g., month-to-month → higher churn)
  - Missing values injected realistically (MCAR and MAR patterns)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def generate_churn_dataset(n_samples: int = 10_000, random_state: int = 42) -> pd.DataFrame:
    """
    Simulate a realistic telecom churn dataset.

    Key design choices:
    - Base churn probability ~27% (industry benchmark for telecom)
    - Churn correlates with: short tenure, high charges, month-to-month contracts,
      many support calls, low login frequency
    - Injects ~5% missing values in 'total_charges' and 'avg_session_duration_mins'
      (realistic: new customers missing billing history)
    """
    rng = np.random.RandomState(random_state)

    n = n_samples

    # ── Demographic features ──────────────────────────────────────────────────
    gender = rng.choice(["Male", "Female"], size=n)
    is_senior = rng.choice([0, 1], size=n, p=[0.84, 0.16])

    # ── Subscription features ─────────────────────────────────────────────────
    contract_type = rng.choice(
        ["Month-to-Month", "One Year", "Two Year"],
        size=n,
        p=[0.55, 0.25, 0.20],  # majority on flexible plans
    )
    payment_method = rng.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        size=n,
        p=[0.34, 0.23, 0.22, 0.21],
    )
    internet_service = rng.choice(
        ["DSL", "Fiber Optic", "No"],
        size=n,
        p=[0.34, 0.44, 0.22],
    )
    has_phone_service = rng.choice(["Yes", "No"], size=n, p=[0.90, 0.10])
    has_online_backup = rng.choice(["Yes", "No"], size=n, p=[0.44, 0.56])
    has_tech_support = rng.choice(["Yes", "No"], size=n, p=[0.41, 0.59])

    # ── Behavioral / usage features ───────────────────────────────────────────
    # Tenure skewed: many new + many long-term customers
    tenure_months = np.clip(
        rng.gamma(shape=1.5, scale=20, size=n).astype(int), 1, 72
    )

    # Monthly charges correlated with internet service
    base_charge = np.where(
        internet_service == "Fiber Optic", 80,
        np.where(internet_service == "DSL", 55, 25)
    )
    monthly_charges = np.clip(
        base_charge + rng.normal(0, 12, n), 18, 120
    ).round(2)

    # Total charges = tenure × monthly (with noise; missing for new customers)
    total_charges = (tenure_months * monthly_charges + rng.normal(0, 50, n)).round(2)
    # New customers (tenure < 3) often have no total_charges history
    new_customer_mask = tenure_months < 3
    total_charges[new_customer_mask & (rng.rand(n) < 0.5)] = np.nan

    num_products = np.clip(rng.poisson(2.5, n), 1, 6)
    num_support_calls = np.clip(rng.poisson(1.2, n), 0, 10)
    days_since_last_login = np.clip(rng.exponential(scale=15, size=n).astype(int), 0, 90)
    avg_session_duration_mins = np.clip(
        rng.gamma(shape=2, scale=15, size=n), 1, 180
    ).round(1)
    # ~5% missing for session data (app users only)
    avg_session_duration_mins[rng.rand(n) < 0.05] = np.nan

    # ── Churn label — built from realistic risk factors ────────────────────────
    # Each factor contributes logit units; final probability via sigmoid
    log_odds = (
        -1.5                                                             # base (intercept)
        + 1.5 * (contract_type == "Month-to-Month").astype(float)        # month-to-month = biggest risk
        + 0.5 * (contract_type == "One Year").astype(float)
        - 1.2 * np.log1p(tenure_months) / np.log1p(72)                  # longer tenure → loyal
        + 0.8 * (monthly_charges / 120)                                  # sticker shock
        + 0.6 * (num_support_calls / 10)                                 # frustration signal
        + 0.5 * (days_since_last_login / 90)                             # disengagement
        + 0.3 * (internet_service == "Fiber Optic").astype(float)        # higher expectations
        - 0.4 * (has_tech_support == "Yes").astype(float)                # support helps retention
        - 0.3 * (has_online_backup == "Yes").astype(float)
        + 0.2 * is_senior                                                # slight senior risk
        + rng.normal(0, 0.3, n)                                          # noise
    )
    churn_prob = 1 / (1 + np.exp(-log_odds))
    churn = (rng.rand(n) < churn_prob).astype(int)

    logger.info(f"Generated {n} samples | Churn rate: {churn.mean():.1%}")

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:06d}" for i in range(n)],
        "gender": gender,
        "is_senior_citizen": is_senior,
        "tenure_months": tenure_months,
        "has_phone_service": has_phone_service,
        "internet_service": internet_service,
        "has_online_backup": has_online_backup,
        "has_tech_support": has_tech_support,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_products": num_products,
        "num_support_calls": num_support_calls,
        "days_since_last_login": days_since_last_login,
        "avg_session_duration_mins": avg_session_duration_mins,
        "churn": churn,
    })

    return df


def load_raw_data(filepath: Path = None, n_samples: int = 10_000) -> pd.DataFrame:
    """
    Load raw data from CSV or generate synthetic data.
    Always returns a DataFrame with consistent schema.
    """
    if filepath and Path(filepath).exists():
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)
    else:
        logger.info("No CSV found — generating synthetic dataset")
        df = generate_churn_dataset(n_samples=n_samples)

    return df
