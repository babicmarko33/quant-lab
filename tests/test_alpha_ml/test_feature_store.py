"""Tests for FeatureStore ML feature matrix builder."""

import numpy as np
import pandas as pd
import pytest

from alpha_ml.features.feature_store import FeatureStore


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """3 years of synthetic daily OHLCV data."""
    rng = np.random.default_rng(42)
    n = 3 * 252
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    log_ret = rng.normal(0.0003, 0.012, n)
    close = 100 * np.exp(np.cumsum(log_ret))
    opens = np.roll(close, 1)
    opens[0] = 100.0
    return pd.DataFrame(
        {
            "open": opens,
            "high": close * (1 + rng.uniform(0, 0.01, n)),
            "low": close * (1 - rng.uniform(0, 0.01, n)),
            "close": close,
            "volume": rng.integers(500_000, 2_000_000, n).astype(float),
        },
        index=dates,
    )


class TestFeatureStore:
    def test_returns_x_and_y_dataframes(self, sample_ohlcv: pd.DataFrame) -> None:
        fs = FeatureStore()
        X, y = fs.build(sample_ohlcv, horizon=5)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_x_and_y_same_index(self, sample_ohlcv: pd.DataFrame) -> None:
        fs = FeatureStore()
        X, y = fs.build(sample_ohlcv, horizon=5)
        assert X.index.equals(y.index)

    def test_no_nan_in_x(self, sample_ohlcv: pd.DataFrame) -> None:
        fs = FeatureStore()
        X, y = fs.build(sample_ohlcv, horizon=5)
        assert not X.isna().any().any(), "NaN found in feature matrix X"

    def test_target_direction_binary(self, sample_ohlcv: pd.DataFrame) -> None:
        fs = FeatureStore()
        X, y = fs.build(sample_ohlcv, horizon=5, target_type="direction")
        unique = set(y.unique())
        assert unique.issubset({0, 1}), f"Non-binary labels: {unique}"

    def test_target_return_is_float(self, sample_ohlcv: pd.DataFrame) -> None:
        fs = FeatureStore()
        X, y = fs.build(sample_ohlcv, horizon=5, target_type="return")
        assert y.dtype == float

    def test_x_has_multiple_features(self, sample_ohlcv: pd.DataFrame) -> None:
        fs = FeatureStore()
        X, y = fs.build(sample_ohlcv, horizon=5)
        assert X.shape[1] >= 5, f"Expected >= 5 features, got {X.shape[1]}"

    def test_winsorization_caps_outliers(self, sample_ohlcv: pd.DataFrame) -> None:
        """After 1% winsorization, z-scores should be bounded."""
        fs = FeatureStore(winsorize_pct=0.01)
        X, _ = fs.build(sample_ohlcv, horizon=5)
        # After winsorization, values should be within reasonable bounds
        # (exact bound depends on distribution, but extreme outliers gone)
        for col in X.columns:
            col_std = X[col].std()
            if col_std > 0:
                z = (X[col] - X[col].mean()) / col_std
                assert z.abs().max() <= 10.0, f"Column {col} has extreme outlier after winsorize"

    def test_feature_names_are_strings(self, sample_ohlcv: pd.DataFrame) -> None:
        fs = FeatureStore()
        X, _ = fs.build(sample_ohlcv, horizon=5)
        assert all(isinstance(c, str) for c in X.columns)

    def test_min_samples_with_horizon(self, sample_ohlcv: pd.DataFrame) -> None:
        """Longer horizon loses more rows at the end."""
        fs = FeatureStore()
        X5, y5 = fs.build(sample_ohlcv, horizon=5)
        X21, y21 = fs.build(sample_ohlcv, horizon=21)
        # Longer horizon → fewer valid samples (both lose warmup + tail)
        assert len(X5) >= len(X21)
