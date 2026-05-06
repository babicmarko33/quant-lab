import numpy as np
import pandas as pd
import pytest

from alpha_engine.strategies.ema_crossover import EMACrossoverStrategy


@pytest.fixture
def price_series() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    trend = np.concatenate([np.linspace(100, 200, 200), np.linspace(200, 100, 100)])
    noise = rng.normal(0, 0.5, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(trend + noise, index=idx)
    return pd.DataFrame({"open": close * 0.999, "close": close}, index=idx)


class TestEMACrossoverStrategy:
    def test_name(self):
        assert EMACrossoverStrategy().name == "ema_crossover"

    def test_default_windows(self):
        s = EMACrossoverStrategy()
        assert s.fast == 12
        assert s.slow == 26

    def test_custom_windows(self):
        s = EMACrossoverStrategy(fast=5, slow=20)
        assert s.fast == 5
        assert s.slow == 20

    def test_fast_must_be_less_than_slow(self):
        with pytest.raises(ValueError, match="fast.*slow"):
            EMACrossoverStrategy(fast=50, slow=10)

    def test_signals_in_valid_set(self, price_series):
        s = EMACrossoverStrategy(fast=5, slow=20)
        signals = s.generate_signals(price_series)
        assert set(signals.dropna().unique()).issubset({-1, 0, 1})

    def test_signals_length_matches_input(self, price_series):
        s = EMACrossoverStrategy(fast=5, slow=20)
        signals = s.generate_signals(price_series)
        assert len(signals) == len(price_series)

    def test_crossover_produces_long_signal(self, price_series):
        s = EMACrossoverStrategy(fast=5, slow=20)
        signals = s.generate_signals(price_series)
        assert (signals == 1).any()

    def test_crossover_produces_short_signal(self, price_series):
        s = EMACrossoverStrategy(fast=5, slow=20)
        signals = s.generate_signals(price_series)
        assert (signals == -1).any()

    def test_ema_reacts_faster_than_sma(self, price_series):
        """EMA crossover should generate signals earlier than SMA crossover
        with equivalent windows due to EMA's higher weight on recent prices."""
        from alpha_engine.strategies.sma_crossover import SMACrossoverStrategy
        ema_s = EMACrossoverStrategy(fast=5, slow=20)
        sma_s = SMACrossoverStrategy(fast=5, slow=20)
        ema_sig = ema_s.generate_signals(price_series)
        sma_sig = sma_s.generate_signals(price_series)
        # Both should eventually signal; EMA strategy should have first signal <= SMA
        ema_first = ema_sig[ema_sig != 0].index.min()
        sma_first = sma_sig[sma_sig != 0].index.min()
        assert ema_first <= sma_first

    def test_run_returns_backtest_result(self, price_series):
        from alpha_engine.backtest.types import BacktestResult
        result = EMACrossoverStrategy(fast=5, slow=20).run(price_series)
        assert isinstance(result, BacktestResult)
