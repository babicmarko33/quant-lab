import numpy as np
import pandas as pd
import pytest

from alpha_engine.strategies.rsi_strategy import RSIMeanReversionStrategy


@pytest.fixture
def oscillating_series() -> pd.DataFrame:
    """Price that overshoots both directions — RSI extremes should trigger."""
    rng = np.random.default_rng(99)
    n = 500
    # Sine wave creates repeated overbought/oversold conditions
    base = 100 + 20 * np.sin(np.linspace(0, 8 * np.pi, n))
    noise = rng.normal(0, 0.3, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(base + noise, index=idx)
    return pd.DataFrame({"open": close * 0.999, "close": close}, index=idx)


class TestRSIMeanReversionStrategy:
    def test_name(self):
        assert RSIMeanReversionStrategy().name == "rsi_mean_reversion"

    def test_default_params(self):
        s = RSIMeanReversionStrategy()
        assert s.window == 14
        assert s.oversold == 30
        assert s.overbought == 70

    def test_custom_params(self):
        s = RSIMeanReversionStrategy(window=7, oversold=25, overbought=75)
        assert s.window == 7
        assert s.oversold == 25
        assert s.overbought == 75

    def test_invalid_thresholds(self):
        with pytest.raises(ValueError, match="oversold.*overbought"):
            RSIMeanReversionStrategy(oversold=70, overbought=30)

    def test_signals_in_valid_set(self, oscillating_series):
        s = RSIMeanReversionStrategy()
        signals = s.generate_signals(oscillating_series)
        assert set(signals.dropna().unique()).issubset({-1, 0, 1})

    def test_signals_length_matches_input(self, oscillating_series):
        s = RSIMeanReversionStrategy()
        signals = s.generate_signals(oscillating_series)
        assert len(signals) == len(oscillating_series)

    def test_oversold_triggers_long(self, oscillating_series):
        """RSI < 30 should produce long signals."""
        s = RSIMeanReversionStrategy()
        signals = s.generate_signals(oscillating_series)
        assert (signals == 1).any(), "No long signals generated — check RSI oversold region"

    def test_overbought_triggers_short(self, oscillating_series):
        """RSI > 70 should produce short signals."""
        s = RSIMeanReversionStrategy()
        signals = s.generate_signals(oscillating_series)
        assert (signals == -1).any(), "No short signals generated — check RSI overbought region"

    def test_no_lookahead(self, oscillating_series):
        s = RSIMeanReversionStrategy()
        full_signals = s.generate_signals(oscillating_series)
        partial = s.generate_signals(oscillating_series.iloc[:200])
        pd.testing.assert_series_equal(
            full_signals.iloc[:200], partial.iloc[:200], check_names=False
        )

    def test_run_returns_backtest_result(self, oscillating_series):
        from alpha_engine.backtest.types import BacktestResult
        result = RSIMeanReversionStrategy().run(oscillating_series)
        assert isinstance(result, BacktestResult)

    def test_threshold_25_75_produces_fewer_signals(self, oscillating_series):
        """Tighter thresholds = fewer extreme signals."""
        s_wide = RSIMeanReversionStrategy(oversold=35, overbought=65)
        s_tight = RSIMeanReversionStrategy(oversold=20, overbought=80)
        signals_wide = s_wide.generate_signals(oscillating_series)
        signals_tight = s_tight.generate_signals(oscillating_series)
        n_wide = (signals_wide != 0).sum()
        n_tight = (signals_tight != 0).sum()
        assert n_wide >= n_tight
