"""Tests for Strategy ABC, Momentum, and Mean-Reversion strategies."""

import numpy as np
import pandas as pd
import pytest

from alpha_engine.strategies.base import Strategy
from alpha_engine.strategies.mean_reversion import BollingerMeanReversionStrategy
from alpha_engine.strategies.momentum import MomentumStrategy


@pytest.fixture
def trending_ohlcv() -> pd.DataFrame:
    """5 years of daily OHLCV with consistent uptrend."""
    rng = np.random.default_rng(42)
    n = 5 * 252
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    # Clear uptrend: +0.05% daily + noise
    log_ret = rng.normal(0.0005, 0.012, n)
    close = 100 * np.exp(np.cumsum(log_ret))
    opens = np.roll(close, 1)
    opens[0] = 100.0
    return pd.DataFrame(
        {"open": opens, "high": close * 1.005, "low": close * 0.995, "close": close, "volume": 1_000_000},
        index=dates,
    )


@pytest.fixture
def mean_reverting_ohlcv() -> pd.DataFrame:
    """Synthetic mean-reverting price series (OU process)."""
    rng = np.random.default_rng(7)
    n = 3 * 252
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    theta, mu, sigma = 0.1, 100.0, 2.0
    prices = np.zeros(n)
    prices[0] = mu
    for i in range(1, n):
        prices[i] = prices[i - 1] + theta * (mu - prices[i - 1]) + sigma * rng.normal()
    return pd.DataFrame(
        {"open": np.roll(prices, 1), "high": prices * 1.005, "low": prices * 0.995,
         "close": prices, "volume": 1_000_000},
        index=dates,
    )


class TestStrategyABC:
    def test_strategy_is_abstract(self) -> None:
        """Cannot instantiate Strategy directly."""
        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]

    def test_concrete_strategy_has_name(self, trending_ohlcv: pd.DataFrame) -> None:
        strat = MomentumStrategy()
        assert isinstance(strat.name, str)
        assert len(strat.name) > 0

    def test_run_returns_backtest_result(self, trending_ohlcv: pd.DataFrame) -> None:
        from alpha_engine.backtest.types import BacktestResult

        strat = MomentumStrategy()
        result = strat.run(trending_ohlcv)
        assert isinstance(result, BacktestResult)


class TestMomentumStrategy:
    def test_signals_in_valid_range(self, trending_ohlcv: pd.DataFrame) -> None:
        strat = MomentumStrategy()
        signals = strat.generate_signals(trending_ohlcv)
        valid = signals.dropna()
        assert valid.isin([-1.0, 0.0, 1.0]).all()

    def test_warmup_period_is_nan(self, trending_ohlcv: pd.DataFrame) -> None:
        strat = MomentumStrategy(lookback=252, skip=21)
        signals = strat.generate_signals(trending_ohlcv)
        # First lookback+skip days should be NaN or 0
        early = signals.iloc[: strat.lookback + strat.skip]
        assert (early.isna() | (early == 0)).all()

    def test_uptrend_mostly_long(self, trending_ohlcv: pd.DataFrame) -> None:
        strat = MomentumStrategy()
        signals = strat.generate_signals(trending_ohlcv)
        # Skip warmup period (lookback + skip days)
        active = signals.iloc[strat.lookback + strat.skip :]
        active_nonzero = active[active != 0]
        long_pct = (active_nonzero == 1.0).mean()
        assert long_pct > 0.5  # Uptrend → majority long signals

    def test_backtest_returns_valid_result(self, trending_ohlcv: pd.DataFrame) -> None:
        strat = MomentumStrategy()
        result = strat.run(trending_ohlcv)
        assert not np.isnan(result.sharpe)
        assert not np.isnan(result.max_drawdown)


class TestBollingerMeanReversionStrategy:
    def test_signals_in_valid_range(self, mean_reverting_ohlcv: pd.DataFrame) -> None:
        strat = BollingerMeanReversionStrategy()
        signals = strat.generate_signals(mean_reverting_ohlcv)
        valid = signals.dropna()
        assert valid.isin([-1.0, 0.0, 1.0]).all()

    def test_below_lower_band_goes_long(self) -> None:
        """When price is below lower BB, signal should be +1."""
        # Build a series: 20 stable days then one big drop
        n = 60
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        # Stable at 100 for window days, then crash to 80 (well below lower band)
        prices = np.full(n, 100.0)
        # Drop by 5 std — will be far below lower band
        prices[30:] = 70.0
        df = pd.DataFrame(
            {"open": prices, "high": prices, "low": prices, "close": prices, "volume": 1_000_000},
            index=dates,
        )
        strat = BollingerMeanReversionStrategy(window=20, num_std=2.0)
        signals = strat.generate_signals(df)
        # Days 30-49: price is far below lower band → expect +1
        mid_signals = signals.iloc[30:40]
        assert (mid_signals == 1.0).any()

    def test_backtest_returns_valid_result(self, mean_reverting_ohlcv: pd.DataFrame) -> None:
        strat = BollingerMeanReversionStrategy()
        result = strat.run(mean_reverting_ohlcv)
        assert not np.isnan(result.sharpe)
