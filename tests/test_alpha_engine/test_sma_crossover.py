import numpy as np
import pandas as pd
import pytest

from alpha_engine.strategies.sma_crossover import SMACrossoverStrategy


@pytest.fixture
def price_series() -> pd.DataFrame:
    """Trending up then down — crosses should fire."""
    rng = np.random.default_rng(42)
    n = 300
    trend = np.concatenate([np.linspace(100, 200, 200), np.linspace(200, 100, 100)])
    noise = rng.normal(0, 0.5, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(trend + noise, index=idx)
    return pd.DataFrame({"open": close * 0.999, "close": close}, index=idx)


class TestSMACrossoverStrategy:
    def test_name(self):
        assert SMACrossoverStrategy().name == "sma_crossover"

    def test_default_windows(self):
        s = SMACrossoverStrategy()
        assert s.fast == 50
        assert s.slow == 200

    def test_custom_windows(self):
        s = SMACrossoverStrategy(fast=10, slow=30)
        assert s.fast == 10
        assert s.slow == 30

    def test_fast_must_be_less_than_slow(self):
        with pytest.raises(ValueError, match="fast.*slow"):
            SMACrossoverStrategy(fast=200, slow=50)

    def test_signals_in_valid_set(self, price_series):
        s = SMACrossoverStrategy(fast=10, slow=30)
        signals = s.generate_signals(price_series)
        assert set(signals.dropna().unique()).issubset({-1, 0, 1})

    def test_signals_length_matches_input(self, price_series):
        s = SMACrossoverStrategy(fast=10, slow=30)
        signals = s.generate_signals(price_series)
        assert len(signals) == len(price_series)

    def test_golden_cross_produces_buy(self, price_series):
        """Trending up series should produce at least one long signal."""
        s = SMACrossoverStrategy(fast=10, slow=30)
        signals = s.generate_signals(price_series)
        assert (signals == 1).any()

    def test_death_cross_produces_sell(self, price_series):
        """Trending down after peak should produce at least one short signal."""
        s = SMACrossoverStrategy(fast=10, slow=30)
        signals = s.generate_signals(price_series)
        assert (signals == -1).any()

    def test_no_lookahead(self, price_series):
        """Signal at position i depends only on data ≤ i."""
        s = SMACrossoverStrategy(fast=10, slow=30)
        full_signals = s.generate_signals(price_series)
        partial = s.generate_signals(price_series.iloc[:100])
        # Signals on shared prefix must agree
        pd.testing.assert_series_equal(
            full_signals.iloc[:100], partial.iloc[:100], check_names=False
        )

    def test_run_returns_backtest_result(self, price_series):
        from alpha_engine.backtest.types import BacktestResult
        result = SMACrossoverStrategy(fast=10, slow=30).run(price_series)
        assert isinstance(result, BacktestResult)
