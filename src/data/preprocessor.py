"""
data/preprocessor.py
─────────────────────
Builds a sklearn Pipeline for end-to-end preprocessing.

Key design decisions:
  - StandardScaler chosen over MinMaxScaler for numeric features:
      → Tree models are scale-invariant, but Logistic Regression is not.
        StandardScaler preserves outlier information (unlike MinMax clipping)
        and is more robust to the heavy-tailed distributions we have here.
  - Missing values imputed BEFORE scaling (order matters in Pipeline)
  - Outlier capping (Winsorization) applied BEFORE imputation to avoid
    outliers biasing the mean imputer
  - Categorical features: OrdinalEncoder inside pipeline (XGBoost handles
    ordinals natively; LR/RF use OneHotEncoder — swappable via param)
  - The full Pipeline object is serialized alongside the model so inference
    always uses identical transformations. No training/serving skew.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from configs.config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL, RANDOM_STATE,
    TEST_SIZE, VAL_SIZE
)
from src.features.feature_engineer import add_engineered_features

logger = logging.getLogger(__name__)


# ── Outlier handling via Winsorization ────────────────────────────────────────

class WinsorizationTransformer:
    """
    Caps outliers at [lower_q, upper_q] percentiles (fit on train, apply to all).
    Safer than dropping rows; preserves dataset size.
    We use IQR method: cap at Q1 - 1.5*IQR and Q3 + 1.5*IQR.
    """

    def __init__(self, factor: float = 1.5):
        self.factor = factor
        self.lower_bounds_: dict = {}
        self.upper_bounds_: dict = {}

    def fit(self, df: pd.DataFrame, columns: list) -> "WinsorizationTransformer":
        for col in columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_bounds_[col] = q1 - self.factor * iqr
            self.upper_bounds_[col] = q3 + self.factor * iqr
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, lo in self.lower_bounds_.items():
            hi = self.upper_bounds_[col]
            original_outliers = ((df[col] < lo) | (df[col] > hi)).sum()
            if original_outliers > 0:
                logger.debug(f"Winsorizing {col}: {original_outliers} outliers capped to [{lo:.2f}, {hi:.2f}]")
            df[col] = df[col].clip(lower=lo, upper=hi)
        return df


def build_preprocessing_pipeline() -> Pipeline:
    """
    Construct the sklearn preprocessing Pipeline.

    Structure:
        Numeric:     Impute (median) → Scale (standard)
        Categorical: Impute (most_frequent) → OneHotEncode

    Median imputation chosen for numerics because our distributions
    are skewed (charges, tenure), making median more robust than mean.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",  # handles unseen categories at inference
                sparse_output=False,
                drop="first",            # avoid perfect multicollinearity for LR
            ),
        ),
    ])

    # NUMERIC_FEATURES already includes engineered features
    # Intersection with what's actually in the data (safety net)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # drop customer_id and other non-feature columns
    )

    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    winsorizer: Optional[WinsorizationTransformer] = None,
    fit_winsorizer: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], Optional[WinsorizationTransformer]]:
    """
    Full data preparation: feature engineering → outlier capping → split X/y.

    Returns:
        X: Feature DataFrame (not yet scaled — that happens in the Pipeline)
        y: Target Series (None if not present)
        winsorizer: Fitted winsorizer (for reuse at inference time)
    """
    df = df.copy()

    # Step 1: Feature engineering (adds new columns)
    df = add_engineered_features(df)

    # Step 2: Extract target before outlier handling
    y = None
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].astype(int)

    # Step 3: Winsorize ONLY numeric features (not target, not categoricals)
    numeric_cols_present = [c for c in NUMERIC_FEATURES if c in df.columns]

    if fit_winsorizer:
        winsorizer = WinsorizationTransformer(factor=1.5)
        winsorizer.fit(df, numeric_cols_present)

    if winsorizer is not None:
        df = winsorizer.transform(df)

    # Step 4: Keep only model features
    all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    available_features = [f for f in all_features if f in df.columns]
    X = df[available_features]

    return X, y, winsorizer


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple:
    """
    Stratified 3-way split: train / val / test.
    Stratification preserves class ratio in each split.
    """
    from sklearn.model_selection import train_test_split

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    # val is 10% of original → ~11% of train_val
    val_fraction = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_fraction,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    logger.info(
        f"Split sizes — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )
    logger.info(
        f"Churn rates — Train: {y_train.mean():.1%} | Val: {y_val.mean():.1%} | Test: {y_test.mean():.1%}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
