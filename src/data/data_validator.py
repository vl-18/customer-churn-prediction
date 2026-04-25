"""
data/data_validator.py
───────────────────────
Schema and statistical validation before any data enters the pipeline.

Fails fast on critical issues; warns on non-critical ones.
In production this would publish metrics to Prometheus/Datadog.
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def fail(self, msg: str):
        self.passed = False
        self.errors.append(msg)
        logger.error(f"[VALIDATION FAIL] {msg}")

    def warn(self, msg: str):
        self.warnings.append(msg)
        logger.warning(f"[VALIDATION WARN] {msg}")

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return (
            f"Validation {status} | "
            f"{len(self.errors)} errors | "
            f"{len(self.warnings)} warnings"
        )


EXPECTED_SCHEMA = {
    "customer_id": "object",
    "gender": "object",
    "is_senior_citizen": "int64",
    "tenure_months": "int64",
    "has_phone_service": "object",
    "internet_service": "object",
    "has_online_backup": "object",
    "has_tech_support": "object",
    "contract_type": "object",
    "payment_method": "object",
    "monthly_charges": "float64",
    "total_charges": "float64",  # nullable
    "num_products": "int64",
    "num_support_calls": "int64",
    "days_since_last_login": "int64",
    "avg_session_duration_mins": "float64",  # nullable
    "churn": "int64",
}

VALID_CATEGORIES = {
    "gender": {"Male", "Female"},
    "internet_service": {"DSL", "Fiber Optic", "No"},
    "contract_type": {"Month-to-Month", "One Year", "Two Year"},
    "payment_method": {"Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"},
    "has_phone_service": {"Yes", "No"},
    "has_online_backup": {"Yes", "No"},
    "has_tech_support": {"Yes", "No"},
}

NUMERIC_BOUNDS = {
    "tenure_months": (0, 120),
    "monthly_charges": (0, 500),
    "num_products": (0, 20),
    "num_support_calls": (0, 50),
    "days_since_last_login": (0, 365),
    "avg_session_duration_mins": (0, 1440),
}


def validate_dataframe(df: pd.DataFrame, require_target: bool = True) -> ValidationReport:
    """
    Run all validation checks. Returns a ValidationReport.

    Args:
        df: Raw or processed DataFrame
        require_target: Set False for inference-time validation
    """
    report = ValidationReport()

    # ── 1. Required columns ───────────────────────────────────────────────────
    expected_cols = set(EXPECTED_SCHEMA.keys())
    if not require_target:
        expected_cols.discard("churn")

    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        report.fail(f"Missing required columns: {missing_cols}")
        return report  # Cannot proceed

    # ── 2. Minimum row count ──────────────────────────────────────────────────
    if len(df) < 10:
        report.fail(f"Too few rows: {len(df)}. Minimum 10 required.")

    # ── 3. Duplicate customer IDs ─────────────────────────────────────────────
    dupe_count = df["customer_id"].duplicated().sum()
    if dupe_count > 0:
        report.warn(f"{dupe_count} duplicate customer_ids found")

    # ── 4. Missing value thresholds ───────────────────────────────────────────
    null_rates = df.isnull().mean()
    HIGH_NULL_THRESHOLD = 0.30  # >30% missing → critical
    WARN_NULL_THRESHOLD = 0.10  # >10% missing → warning

    for col, rate in null_rates.items():
        if rate > HIGH_NULL_THRESHOLD:
            report.fail(f"Column '{col}' has {rate:.1%} missing values (threshold: {HIGH_NULL_THRESHOLD:.0%})")
        elif rate > WARN_NULL_THRESHOLD:
            report.warn(f"Column '{col}' has {rate:.1%} missing values")

    # ── 5. Category set validation ────────────────────────────────────────────
    for col, valid_set in VALID_CATEGORIES.items():
        if col not in df.columns:
            continue
        actual_values = set(df[col].dropna().unique())
        unknown = actual_values - valid_set
        if unknown:
            report.warn(f"Column '{col}' has unexpected categories: {unknown}")

    # ── 6. Numeric range checks ───────────────────────────────────────────────
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        out_of_range = df[col].dropna().between(lo, hi, inclusive="both").sum()
        total_valid = df[col].notna().sum()
        violation_pct = 1 - (out_of_range / total_valid) if total_valid > 0 else 0
        if violation_pct > 0.01:  # >1% out of range
            report.warn(f"Column '{col}': {violation_pct:.1%} values outside [{lo}, {hi}]")

    # ── 7. Target distribution (training only) ────────────────────────────────
    if require_target and "churn" in df.columns:
        churn_rate = df["churn"].mean()
        if churn_rate < 0.02:
            report.warn(f"Very low churn rate: {churn_rate:.1%} — check label encoding")
        elif churn_rate > 0.60:
            report.warn(f"Very high churn rate: {churn_rate:.1%} — check data quality")

        if df["churn"].isin([0, 1]).sum() != len(df):
            report.fail("Target column 'churn' contains values other than 0/1")

    logger.info(report.summary())
    return report
