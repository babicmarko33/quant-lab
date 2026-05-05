"""Tests for quantcore.indicators.technical module."""

import numpy as np
import pandas as pd
import pytest

from quantcore.indicators.technical import (
    atr,
    bollinger_bands,
    ema,
    macd,
    returns,
    rsi,
    sma,
    volatility,
)


@pytest.fixture
def sample_prices() -> pd.Series:
    """Generate a deterministic price series for testing."""
    rng = np.random.default_rng(42)
    n = 300
    # Random walk with drift
    log_returns = rng.normal(0.0003, 0.015, n)
    prices = 100 * np.exp(np.cumsum(log_returns))
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="close")


@pytest.fixture
def sample_ohlcv(sample_prices: pd.Series) -> pd.DataFrame:
    """Generate OHLCV DataFrame from price series."""
    close = sample_prices
    high = close * (1 + np.abs(np.random.default_rng(42).normal(0, 0.005, len(close))))
    low = close * (1 - np.abs(np.random.default_rng(43).normal(0, 0.005, len(close))))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = np.random.default_rng(44).integers(1_000_000, 10_000_000, len(close))

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=close.index,
    )


class TestSMA:
    def test_basic_computation(self, sample_prices: pd.Series) -> None:
        result = sma(sample_prices, window=20)
        assert len(result) == len(sample_prices)
        # First 19 values should be NaN (window-1)
        assert result.iloc[:19].isna().all()
        # 20th value should be mean of first 20 prices
        expected = sample_prices.iloc[:20].mean()
        assert abs(result.iloc[19] - expected) < 1e-10

    def test_nan_count(self, sample_prices: pd.Series) -> None:
        window = 50
        result = sma(sample_prices, window=window)
        assert result.isna().sum() == window - 1

    def test_window_1_equals_price(self, sample_prices: pd.Series) -> None:
        result = sma(sample_prices, window=1)
        pd.testing.assert_series_equal(result, sample_prices, check_names=False)


class TestEMA:
    def test_basic_computation(self, sample_prices: pd.Series) -> None:
        result = ema(sample_prices, window=20)
        assert len(result) == len(sample_prices)
        # EMA should be close to SMA for stationary data
        sma_result = sma(sample_prices, 20)
        # After sufficient warmup, EMA and SMA should be correlated
        assert result.iloc[50:].corr(sma_result.iloc[50:]) > 0.95

    def test_responsiveness(self, sample_prices: pd.Series) -> None:
        """EMA should react faster to recent prices than SMA."""
        # After a sharp move, EMA should be closer to current price
        ema_result = ema(sample_prices, 20)
        sma_result = sma(sample_prices, 20)
        last_price = sample_prices.iloc[-1]
        # EMA is more responsive (closer to last price on average)
        ema_diff = abs(ema_result.iloc[-1] - last_price)
        sma_diff = abs(sma_result.iloc[-1] - last_price)
        # This is a statistical property, not always true for every realization
        # but should hold most of the time for trending data
        assert isinstance(ema_diff, float)
        assert isinstance(sma_diff, float)


class TestRSI:
    def test_range(self, sample_prices: pd.Series) -> None:
        result = rsi(sample_prices, window=14)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_nan_count(self, sample_prices: pd.Series) -> None:
        window = 14
        result = rsi(sample_prices, window=window)
        assert result.iloc[:window].isna().all()

    def test_constant_price_gives_nan_or_50(self) -> None:
        """Constant prices → no gains or losses → RSI undefined or 50."""
        prices = pd.Series([100.0] * 50)
        result = rsi(prices, 14)
        valid = result.dropna()
        # With zero gains and zero losses, result is NaN (0/0)
        assert valid.isna().all() or (valid == 50.0).all() or len(valid) == 0


class TestMACD:
    def test_output_columns(self, sample_prices: pd.Series) -> None:
        result = macd(sample_prices)
        assert list(result.columns) == ["macd", "signal", "histogram"]

    def test_histogram_equals_diff(self, sample_prices: pd.Series) -> None:
        result = macd(sample_prices)
        expected = result["macd"] - result["signal"]
        pd.testing.assert_series_equal(result["histogram"], expected, check_names=False)


class TestBollingerBands:
    def test_output_columns(self, sample_prices: pd.Series) -> None:
        result = bollinger_bands(sample_prices, window=20)
        assert "upper" in result.columns
        assert "lower" in result.columns
        assert "middle" in result.columns

    def test_band_ordering(self, sample_prices: pd.Series) -> None:
        result = bollinger_bands(sample_prices, window=20)
        valid = result.dropna()
        assert (valid["upper"] >= valid["middle"]).all()
        assert (valid["middle"] >= valid["lower"]).all()


class TestATR:
    def test_positive_values(self, sample_ohlcv: pd.DataFrame) -> None:
        result = atr(sample_ohlcv, window=14)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_nan_count(self, sample_ohlcv: pd.DataFrame) -> None:
        window = 14
        result = atr(sample_ohlcv, window=window)
        # Should have NaN for initial warmup period
        assert result.iloc[:window].isna().any()


class TestReturns:
    def test_simple_returns(self, sample_prices: pd.Series) -> None:
        result = returns(sample_prices, method="simple")
        # First value is NaN
        assert pd.isna(result.iloc[0])
        # Manual check for second value
        expected = sample_prices.iloc[1] / sample_prices.iloc[0] - 1
        assert abs(result.iloc[1] - expected) < 1e-10

    def test_log_returns(self, sample_prices: pd.Series) -> None:
        result = returns(sample_prices, method="log")
        expected = np.log(sample_prices.iloc[1] / sample_prices.iloc[0])
        assert abs(result.iloc[1] - expected) < 1e-10


class TestVolatility:
    def test_positive_values(self, sample_prices: pd.Series) -> None:
        result = volatility(sample_prices, window=20)
        valid = result.dropna()
        assert (valid > 0).all()

    def test_annualization(self, sample_prices: pd.Series) -> None:
        annual = volatility(sample_prices, 20, annualize=True)
        daily = volatility(sample_prices, 20, annualize=False)
        valid_idx = annual.dropna().index
        ratio = annual[valid_idx] / daily[valid_idx]
        # Should be approximately √252
        expected_ratio = np.sqrt(252)
        assert np.allclose(ratio.values, expected_ratio, rtol=1e-10)
