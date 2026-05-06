"""Tests for RLStrategy — RLPortfolioAgent wired into BacktestEngine via Strategy ABC."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest.types import BacktestResult
from alpha_engine.rl.rl_strategy import RLStrategy

pytestmark = pytest.mark.slow


@pytest.fixture
def ohlcv_df():
    """60-bar synthetic OHLCV DataFrame (single asset)."""
    rng = np.random.default_rng(0)
    n = 60
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    return pd.DataFrame({
        "open": close * (1 + rng.uniform(-0.003, 0.003, n)),
        "high": close * (1 + rng.uniform(0.001, 0.01, n)),
        "low": close * (1 - rng.uniform(0.001, 0.01, n)),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


class TestRLStrategy:
    def test_instantiation(self):
        """RLStrategy can be created with default parameters."""
        strategy = RLStrategy()
        assert strategy is not None

    def test_name_property(self):
        """name property returns expected string."""
        strategy = RLStrategy()
        assert strategy.name == "rl_portfolio"

    def test_is_strategy_subclass(self):
        """RLStrategy is a subclass of Strategy."""
        from alpha_engine.strategies.base import Strategy
        assert issubclass(RLStrategy, Strategy)

    def test_generate_signals_returns_series(self, ohlcv_df):
        """generate_signals returns a pd.Series of same length as input."""
        strategy = RLStrategy(total_timesteps=100)
        signals = strategy.generate_signals(ohlcv_df)
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(ohlcv_df)

    def test_signals_in_valid_range(self, ohlcv_df):
        """All signals are in [-1, 0, +1]."""
        strategy = RLStrategy(total_timesteps=100)
        signals = strategy.generate_signals(ohlcv_df)
        assert signals.isin([-1.0, 0.0, 1.0]).all()

    def test_signals_index_matches_df(self, ohlcv_df):
        """Signal index matches the input DataFrame index."""
        strategy = RLStrategy(total_timesteps=100)
        signals = strategy.generate_signals(ohlcv_df)
        assert signals.index.equals(ohlcv_df.index)

    def test_run_returns_backtest_result(self, ohlcv_df):
        """run() returns a BacktestResult (via Strategy base run())."""
        strategy = RLStrategy(total_timesteps=100)
        result = strategy.run(ohlcv_df)
        assert isinstance(result, BacktestResult)
