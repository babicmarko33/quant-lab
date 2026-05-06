"""Tests for RegimeFilteredStrategy. Written before implementation (TDD RED)."""
import numpy as np
import pandas as pd
import pytest

from alpha_engine.regime.regime_filtered_strategy import RegimeFilteredStrategy
from alpha_engine.strategies.momentum import MomentumStrategy

N = 400


@pytest.fixture
def ohlcv_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = pd.Series(100.0 + np.cumsum(rng.standard_normal(N) * 0.01))
    idx = pd.bdate_range("2020-01-01", periods=N)
    df = pd.DataFrame({
        "open": close.values * 0.999,
        "high": close.values * 1.005,
        "low": close.values * 0.995,
        "close": close.values,
        "volume": 1_000_000.0,
    }, index=idx)
    return df


class TestRegimeFilteredStrategy:
    def test_instantiation(self):
        inner = MomentumStrategy()
        strat = RegimeFilteredStrategy(inner, active_regime=0)
        assert isinstance(strat, RegimeFilteredStrategy)

    def test_name(self):
        inner = MomentumStrategy()
        strat = RegimeFilteredStrategy(inner, active_regime=0)
        assert "regime" in strat.name

    def test_generate_signals_returns_series(self, ohlcv_df):
        inner = MomentumStrategy()
        strat = RegimeFilteredStrategy(inner, active_regime=0, n_regimes=2)
        signals = strat.generate_signals(ohlcv_df)
        assert isinstance(signals, pd.Series)

    def test_signals_aligned_to_index(self, ohlcv_df):
        inner = MomentumStrategy()
        strat = RegimeFilteredStrategy(inner, active_regime=0, n_regimes=2)
        signals = strat.generate_signals(ohlcv_df)
        assert signals.index.equals(ohlcv_df.index)

    def test_signals_in_valid_range(self, ohlcv_df):
        inner = MomentumStrategy()
        strat = RegimeFilteredStrategy(inner, active_regime=0, n_regimes=2)
        signals = strat.generate_signals(ohlcv_df)
        assert signals.isin([-1, 0, 1]).all()

    def test_inactive_regime_zeroes_signals(self, ohlcv_df):
        """When active_regime is set to a regime that never appears, all signals=0."""
        inner = MomentumStrategy()
        # Use regime 99 — will never be classified
        strat = RegimeFilteredStrategy(inner, active_regime=99, n_regimes=2)
        signals = strat.generate_signals(ohlcv_df)
        assert (signals == 0).all()

    def test_subclass_of_strategy(self):
        from alpha_engine.strategies.base import Strategy
        inner = MomentumStrategy()
        strat = RegimeFilteredStrategy(inner, active_regime=0)
        assert isinstance(strat, Strategy)
