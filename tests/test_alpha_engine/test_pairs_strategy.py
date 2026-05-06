"""Tests for PairsStrategy. Written before implementation (TDD RED)."""
import numpy as np
import pandas as pd
import pytest

from alpha_engine.strategies.pairs import PairsStrategy

N = 400


@pytest.fixture
def cointegrated_ohlcv() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    x_close = pd.Series(np.cumsum(rng.standard_normal(N)) + 100.0)
    y_close = pd.Series(1.5 * x_close + rng.standard_normal(N) * 1.5)
    idx = pd.bdate_range("2020-01-01", periods=N)

    def _make_ohlcv(close: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(index=idx)
        df["close"] = close.values
        df["open"] = df["close"] * 0.999
        df["high"] = df["close"] * 1.005
        df["low"] = df["close"] * 0.995
        df["volume"] = 1_000_000.0
        return df

    return _make_ohlcv(x_close), _make_ohlcv(y_close)


class TestPairsStrategy:
    def test_instantiation(self):
        strat = PairsStrategy(entry_z=1.5, exit_z=0.5)
        assert isinstance(strat, PairsStrategy)

    def test_name(self):
        strat = PairsStrategy()
        assert strat.name == "pairs_trading"

    def test_generate_signals_returns_series(self, cointegrated_ohlcv):
        df_x, df_y = cointegrated_ohlcv
        strat = PairsStrategy(entry_z=1.0, exit_z=0.25)
        signals = strat.generate_signals(df_x, df_y)
        assert isinstance(signals, pd.Series)

    def test_signals_aligned_to_index(self, cointegrated_ohlcv):
        df_x, df_y = cointegrated_ohlcv
        strat = PairsStrategy(entry_z=1.0, exit_z=0.25)
        signals = strat.generate_signals(df_x, df_y)
        assert signals.index.equals(df_x.index)

    def test_signals_in_valid_range(self, cointegrated_ohlcv):
        df_x, df_y = cointegrated_ohlcv
        strat = PairsStrategy(entry_z=1.0, exit_z=0.25)
        signals = strat.generate_signals(df_x, df_y)
        assert signals.isin([-1, 0, 1]).all()

    def test_generates_some_nonzero_signals(self, cointegrated_ohlcv):
        """A cointegrated pair should produce at least some trade signals."""
        df_x, df_y = cointegrated_ohlcv
        strat = PairsStrategy(entry_z=0.5, exit_z=0.1)
        signals = strat.generate_signals(df_x, df_y)
        assert (signals != 0).sum() > 5

    def test_raises_on_mismatched_lengths(self, cointegrated_ohlcv):
        df_x, df_y = cointegrated_ohlcv
        strat = PairsStrategy()
        with pytest.raises(ValueError, match="same length"):
            strat.generate_signals(df_x, df_y.iloc[:-10])

    def test_default_params(self):
        strat = PairsStrategy()
        assert strat.entry_z > 0
        assert strat.exit_z >= 0
        assert strat.entry_z > strat.exit_z
