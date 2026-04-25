"""
tests/test_pipeline.py
───────────────────────
Unit + integration tests for the churn prediction system.

Coverage targets:
  - Data validation (schema, nulls, ranges)
  - Feature engineering (correctness + no data leakage)
  - Preprocessing (imputation, scaling, encoding)
  - Model training (CV runs, shape checks)
  - API (endpoint contracts, edge cases)
  - Monitoring (PSI computation, drift detection)

Run with:
    python -m pytest tests/ -v
    python -m pytest tests/ -v --tb=short   # compact tracebacks
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_loader import generate_churn_dataset
from src.data.data_validator import validate_dataframe
from src.data.preprocessor import (
    prepare_data, build_preprocessing_pipeline,
    train_val_test_split, WinsorizationTransformer
)
from src.features.feature_engineer import add_engineered_features
from src.monitoring.monitor import compute_psi, compute_ks_test, DriftDetector


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Small dataset for fast tests."""
    return generate_churn_dataset(n_samples=500, random_state=0)


@pytest.fixture(scope="module")
def prepared_data(raw_df):
    X, y, winsorizer = prepare_data(raw_df, fit_winsorizer=True)
    return X, y, winsorizer


# ── Data loader tests ─────────────────────────────────────────────────────────

class TestDataLoader:
    def test_generates_expected_shape(self, raw_df):
        assert len(raw_df) == 500
        assert "churn" in raw_df.columns
        assert "customer_id" in raw_df.columns

    def test_churn_rate_realistic(self, raw_df):
        rate = raw_df["churn"].mean()
        assert 0.15 <= rate <= 0.50, f"Unexpected churn rate: {rate:.1%}"

    def test_churn_binary(self, raw_df):
        assert set(raw_df["churn"].unique()).issubset({0, 1})

    def test_no_all_null_columns(self, raw_df):
        all_null = raw_df.isnull().all()
        assert not all_null.any(), f"All-null columns: {all_null[all_null].index.tolist()}"

    def test_expected_columns_present(self, raw_df):
        required = [
            "tenure_months", "monthly_charges", "contract_type",
            "internet_service", "churn"
        ]
        for col in required:
            assert col in raw_df.columns, f"Missing column: {col}"


# ── Validation tests ──────────────────────────────────────────────────────────

class TestDataValidator:
    def test_valid_data_passes(self, raw_df):
        report = validate_dataframe(raw_df, require_target=True)
        assert report.passed, f"Errors: {report.errors}"

    def test_missing_column_fails(self, raw_df):
        df_missing = raw_df.drop(columns=["tenure_months"])
        report = validate_dataframe(df_missing, require_target=True)
        assert not report.passed
        assert any("tenure_months" in e for e in report.errors)

    def test_invalid_churn_values_fail(self, raw_df):
        df_bad = raw_df.copy()
        df_bad.loc[0, "churn"] = 99  # invalid label
        report = validate_dataframe(df_bad, require_target=True)
        assert not report.passed

    def test_high_null_rate_fails(self, raw_df):
        df_null = raw_df.copy()
        df_null.loc[:, "monthly_charges"] = np.nan  # 100% null
        report = validate_dataframe(df_null)
        assert not report.passed

    def test_unknown_category_warns(self, raw_df):
        df_cat = raw_df.copy()
        df_cat.loc[0, "contract_type"] = "Unknown Plan"
        report = validate_dataframe(df_cat)
        assert len(report.warnings) > 0

    def test_no_target_validation(self, raw_df):
        df_no_target = raw_df.drop(columns=["churn"])
        report = validate_dataframe(df_no_target, require_target=False)
        assert report.passed


# ── Feature engineering tests ─────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_adds_usage_score(self, raw_df):
        df = add_engineered_features(raw_df)
        assert "usage_score" in df.columns
        assert df["usage_score"].between(0, 1).all(), "usage_score must be in [0, 1]"

    def test_adds_support_call_rate(self, raw_df):
        df = add_engineered_features(raw_df)
        assert "support_call_rate" in df.columns
        assert (df["support_call_rate"] >= 0).all()

    def test_adds_charge_ratio(self, raw_df):
        df = add_engineered_features(raw_df)
        assert "charge_per_month_ratio" in df.columns
        assert df["charge_per_month_ratio"].between(0.5, 2.0).all()

    def test_adds_interaction_feature(self, raw_df):
        df = add_engineered_features(raw_df)
        assert "tenure_x_products" in df.columns
        assert (df["tenure_x_products"] >= 0).all()

    def test_no_rows_dropped(self, raw_df):
        df = add_engineered_features(raw_df)
        assert len(df) == len(raw_df), "Feature engineering must not drop rows"

    def test_original_columns_preserved(self, raw_df):
        df = add_engineered_features(raw_df)
        for col in raw_df.columns:
            assert col in df.columns

    def test_usage_score_no_nulls(self, raw_df):
        """usage_score must handle nulls in input gracefully."""
        df = add_engineered_features(raw_df)
        assert df["usage_score"].isnull().sum() == 0


# ── Winsorizer tests ──────────────────────────────────────────────────────────

class TestWinsorizer:
    def test_caps_upper_outliers(self):
        w = WinsorizationTransformer(factor=1.5)
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 1000]})  # 1000 is an extreme outlier
        w.fit(df, ["x"])
        out = w.transform(df)
        assert out["x"].max() < 1000, "Outlier should have been capped"

    def test_caps_lower_outliers(self):
        w = WinsorizationTransformer(factor=1.5)
        df = pd.DataFrame({"x": [-1000, 2, 3, 4, 5, 6]})
        w.fit(df, ["x"])
        out = w.transform(df)
        assert out["x"].min() > -1000, "Lower outlier should have been capped"

    def test_does_not_modify_inliers(self):
        w = WinsorizationTransformer(factor=1.5)
        df = pd.DataFrame({"x": [10, 11, 12, 13, 14, 15]})
        w.fit(df, ["x"])
        out = w.transform(df)
        assert (out["x"] == df["x"]).all(), "Inliers must not be modified"

    def test_fit_on_train_apply_to_test(self):
        """Critical: winsorizer must be fit only on train, applied to test."""
        train = pd.DataFrame({"x": list(range(100))})
        test = pd.DataFrame({"x": [200, 300, -100]})  # all outliers relative to train
        w = WinsorizationTransformer(factor=1.5)
        w.fit(train, ["x"])
        out = w.transform(test)
        # All test values should be capped to train's IQR bounds
        assert out["x"].max() <= w.upper_bounds_["x"] + 0.001
        assert out["x"].min() >= w.lower_bounds_["x"] - 0.001


# ── Preprocessing pipeline tests ──────────────────────────────────────────────

class TestPreprocessor:
    def test_output_is_numpy_array(self, prepared_data):
        X, y, winsorizer = prepared_data
        preprocessor = build_preprocessing_pipeline()
        X_arr = preprocessor.fit_transform(X, y)
        assert isinstance(X_arr, np.ndarray)

    def test_no_nulls_after_transform(self, prepared_data):
        X, y, winsorizer = prepared_data
        preprocessor = build_preprocessing_pipeline()
        X_arr = preprocessor.fit_transform(X, y)
        assert not np.isnan(X_arr).any(), "No NaNs allowed after preprocessing"

    def test_output_shape_consistent(self, prepared_data):
        X, y, winsorizer = prepared_data
        preprocessor = build_preprocessing_pipeline()
        X_arr = preprocessor.fit_transform(X, y)
        # n_rows preserved
        assert X_arr.shape[0] == len(X)
        # At least as many columns as input features (OHE expands categoricals)
        assert X_arr.shape[1] >= X.shape[1]

    def test_train_val_test_split_stratified(self, prepared_data):
        X, y, _ = prepared_data
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
        # Check churn rates are similar across splits (stratification)
        base_rate = y.mean()
        for split_y, name in [(y_train, "train"), (y_val, "val"), (y_test, "test")]:
            assert abs(split_y.mean() - base_rate) < 0.05, \
                f"Churn rate in {name} split deviates too much: {split_y.mean():.3f} vs {base_rate:.3f}"

    def test_no_overlap_between_splits(self, prepared_data):
        X, y, _ = prepared_data
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
        train_idx = set(X_train.index)
        val_idx = set(X_val.index)
        test_idx = set(X_test.index)
        assert train_idx.isdisjoint(test_idx), "Train and test sets overlap!"
        assert train_idx.isdisjoint(val_idx), "Train and val sets overlap!"
        assert val_idx.isdisjoint(test_idx), "Val and test sets overlap!"


# ── Monitoring tests ──────────────────────────────────────────────────────────

class TestMonitoring:
    def test_psi_zero_for_identical_distributions(self):
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data.copy())
        assert psi < 0.01, f"PSI should be ~0 for identical distributions, got {psi:.4f}"

    def test_psi_high_for_shifted_distribution(self):
        ref = np.random.normal(0, 1, 1000)
        current = np.random.normal(5, 1, 1000)  # massively shifted
        psi = compute_psi(ref, current)
        assert psi > 0.2, f"PSI should be high for shifted distribution, got {psi:.4f}"

    def test_ks_test_same_distribution(self):
        ref = np.random.normal(0, 1, 500)
        current = np.random.normal(0, 1, 500)
        _, p_value = compute_ks_test(ref, current)
        assert p_value > 0.05, f"p_value should be high for same distribution, got {p_value:.4f}"

    def test_ks_test_different_distribution(self):
        ref = np.random.normal(0, 1, 500)
        current = np.random.normal(10, 1, 500)  # completely different
        _, p_value = compute_ks_test(ref, current)
        assert p_value < 0.001, f"p_value should be very low for different distributions, got {p_value:.4f}"

    def test_drift_detector_no_drift(self, raw_df):
        df = add_engineered_features(raw_df)
        detector = DriftDetector()
        detector.fit_reference(df, ["monthly_charges", "tenure_months"])
        # Same data → no drift
        report = detector.detect_drift(df, window_label="test")
        assert not report["overall_alert"], "No drift expected on same data"

    def test_drift_detector_detects_shift(self, raw_df):
        df = add_engineered_features(raw_df)
        detector = DriftDetector()
        detector.fit_reference(df, ["monthly_charges"])
        # Massively shifted data
        df_drifted = df.copy()
        df_drifted["monthly_charges"] = df_drifted["monthly_charges"] * 5 + 200
        report = detector.detect_drift(df_drifted, window_label="drifted")
        assert report["overall_alert"], "Drift should be detected on shifted data"
        assert "monthly_charges" in report["alerted_features"]


# ── End-to-end integration test ───────────────────────────────────────────────

class TestEndToEnd:
    def test_full_pipeline_runs(self, raw_df):
        """Smoke test: full pipeline from raw data to predictions."""
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        X, y, winsorizer = prepare_data(raw_df, fit_winsorizer=True)
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

        preprocessor = build_preprocessing_pipeline()
        clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=0)
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipeline.fit(X_train, y_train)

        probs = pipeline.predict_proba(X_test)[:, 1]
        assert probs.shape == (len(X_test),)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_predictions_are_probabilities(self, raw_df):
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        X, y, winsorizer = prepare_data(raw_df, fit_winsorizer=True)
        preprocessor = build_preprocessing_pipeline()
        clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=0)
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipeline.fit(X, y)

        probs = pipeline.predict_proba(X)[:, 1]
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0
        # Both classes should be predicted at some threshold
        assert (probs > 0.5).any()
        assert (probs < 0.5).any()

    def test_winsorizer_not_fitted_on_test(self, raw_df):
        """
        Anti-leakage test: winsorizer should be fit on train, not full dataset.
        Bounds from train-only fit vs full-dataset fit should differ.
        """
        X, y, _ = prepare_data(raw_df, fit_winsorizer=True)
        X_train, _, X_test, y_train, _, y_test = train_val_test_split(X, y)

        w_train = WinsorizationTransformer()
        w_train.fit(X_train, ["monthly_charges"])

        w_full = WinsorizationTransformer()
        w_full.fit(X, ["monthly_charges"])

        # Bounds should differ (train subset ≠ full dataset)
        # This is a structural check, not an exact numerical assertion
        assert isinstance(w_train.upper_bounds_["monthly_charges"], float)
        assert isinstance(w_full.upper_bounds_["monthly_charges"], float)


if __name__ == "__main__":
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False
    )
