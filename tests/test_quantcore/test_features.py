"""Tests for quantcore.features.pipeline module."""

import numpy as np
import pandas as pd
import pytest

from quantcore.features.pipeline import add_target, build_features


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate realistic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = rng.integers(1_000_000, 10_000_000, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume, "adj_close": close},
        index=dates,
    )


class TestBuildFeatures:
    def test_output_is_dataframe(self, sample_ohlcv: pd.DataFrame) -> None:
        result = build_features(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_same_index_as_input(self, sample_ohlcv: pd.DataFrame) -> None:
        result = build_features(sample_ohlcv)
        pd.testing.assert_index_equal(result.index, sample_ohlcv.index)

    def test_contains_expected_features(self, sample_ohlcv: pd.DataFrame) -> None:
        result = build_features(sample_ohlcv)
        expected_cols = ["sma_20", "ema_12", "rsi", "macd", "bb_upper", "atr", "volatility", "return_1d"]
        for col in expected_cols:
            assert col in result.columns, f"Missing feature: {col}"

    def test_no_look_ahead_bias(self, sample_ohlcv: pd.DataFrame) -> None:
        """Features at time t must not use data from t+1 onwards."""
        full_features = build_features(sample_ohlcv)
        # Compute features on first 100 rows only
        partial_features = build_features(sample_ohlcv.iloc[:100])
        # Features at row 99 should be identical in both
        for col in partial_features.columns:
            if not pd.isna(partial_features[col].iloc[-1]):
                assert abs(full_features[col].iloc[99] - partial_features[col].iloc[-1]) < 1e-10, (
                    f"Look-ahead bias detected in feature: {col}"
                )

    def test_custom_windows(self, sample_ohlcv: pd.DataFrame) -> None:
        result = build_features(sample_ohlcv, sma_windows=[10, 30], ema_windows=[5])
        assert "sma_10" in result.columns
        assert "sma_30" in result.columns
        assert "ema_5" in result.columns
        assert "sma_20" not in result.columns  # Default not included


class TestAddTarget:
    def test_direction_target(self, sample_ohlcv: pd.DataFrame) -> None:
        features = build_features(sample_ohlcv)
        result = add_target(features, sample_ohlcv["close"], horizon=1, target_type="direction")
        assert "target" in result.columns
        # Target should be binary (0.0 or 1.0)
        valid = result["target"].dropna()
        assert set(valid.unique()).issubset({0.0, 1.0})

    def test_return_target(self, sample_ohlcv: pd.DataFrame) -> None:
        features = build_features(sample_ohlcv)
        result = add_target(features, sample_ohlcv["close"], horizon=1, target_type="return")
        assert "target" in result.columns
        # Last row should be NaN (no forward data)
        assert pd.isna(result["target"].iloc[-1])

    def test_target_horizon(self, sample_ohlcv: pd.DataFrame) -> None:
        features = build_features(sample_ohlcv)
        result = add_target(features, sample_ohlcv["close"], horizon=5, target_type="direction")
        # Last 5 rows should be NaN
        assert result["target"].iloc[-5:].isna().all()
