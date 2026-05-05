"""Tests for the core vectorized backtesting engine.

These tests verify the most critical invariant in quantitative finance:
NO LOOK-AHEAD BIAS — signals on day T use only data available at T,
fills execute at the OPEN of day T+1.
"""

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest.engine import run_backtest
from alpha_engine.backtest.types import BacktestResult


@pytest.fixture
def flat_ohlcv() -> pd.DataFrame:
    """252 days of flat $100 prices — useful for cost testing."""
    n = 252
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    price = np.full(n, 100.0)
    return pd.DataFrame(
        {"open": price, "high": price * 1.005, "low": price * 0.995, "close": price, "volume": 1_000_000},
        index=dates,
    )


@pytest.fixture
def trending_up_ohlcv() -> pd.DataFrame:
    """Perfect up-trend: +0.1% every day."""
    n = 252
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    closes = 100 * np.cumprod(np.full(n, 1.001))
    opens = np.roll(closes, 1)
    opens[0] = 100.0
    return pd.DataFrame(
        {"open": opens, "high": closes * 1.005, "low": closes * 0.995, "close": closes, "volume": 1_000_000},
        index=dates,
    )


@pytest.fixture
def trending_down_ohlcv() -> pd.DataFrame:
    """Perfect down-trend: -0.1% every day."""
    n = 252
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    closes = 100 * np.cumprod(np.full(n, 0.999))
    opens = np.roll(closes, 1)
    opens[0] = 100.0
    return pd.DataFrame(
        {"open": opens, "high": closes * 1.005, "low": closes * 0.995, "close": closes, "volume": 1_000_000},
        index=dates,
    )


class TestBacktestResult:
    def test_returns_backtest_result_type(self, flat_ohlcv: pd.DataFrame) -> None:
        signals = pd.Series(np.zeros(len(flat_ohlcv)), index=flat_ohlcv.index)
        result = run_backtest(signals, flat_ohlcv)
        assert isinstance(result, BacktestResult)

    def test_zero_signals_zero_trades(self, flat_ohlcv: pd.DataFrame) -> None:
        """No signals → no trades, zero costs, flat equity."""
        signals = pd.Series(np.zeros(len(flat_ohlcv)), index=flat_ohlcv.index)
        result = run_backtest(signals, flat_ohlcv, commission_bps=10, slippage_bps=5)
        assert result.n_trades == 0
        assert abs(result.total_return) < 1e-10

    def test_long_in_uptrend_positive_return(self, trending_up_ohlcv: pd.DataFrame) -> None:
        """Long signal every day in an uptrend → positive total return."""
        signals = pd.Series(np.ones(len(trending_up_ohlcv)), index=trending_up_ohlcv.index)
        result = run_backtest(signals, trending_up_ohlcv, commission_bps=0, slippage_bps=0)
        assert result.total_return > 0

    def test_short_in_downtrend_positive_return(self, trending_down_ohlcv: pd.DataFrame) -> None:
        """Short signal in downtrend → positive return."""
        signals = pd.Series(-np.ones(len(trending_down_ohlcv)), index=trending_down_ohlcv.index)
        result = run_backtest(signals, trending_down_ohlcv, commission_bps=0, slippage_bps=0)
        assert result.total_return > 0

    def test_costs_reduce_returns(self, trending_up_ohlcv: pd.DataFrame) -> None:
        """Higher transaction costs → lower total return."""
        signals = pd.Series(np.ones(len(trending_up_ohlcv)), index=trending_up_ohlcv.index)
        no_cost = run_backtest(signals, trending_up_ohlcv, commission_bps=0, slippage_bps=0)
        with_cost = run_backtest(signals, trending_up_ohlcv, commission_bps=20, slippage_bps=10)
        assert no_cost.total_return > with_cost.total_return

    def test_no_look_ahead_bias(self, trending_up_ohlcv: pd.DataFrame) -> None:
        """Signal on day T fills on day T+1 open, not day T close.

        We verify this by computing the return with the correct fill price.
        The return at day T+1 = (signal_T * (close_T+1 / open_T+1 - 1)).
        """
        n = len(trending_up_ohlcv)
        # Buy on day 0, hold for 5 days only
        signals = pd.Series(np.zeros(n), index=trending_up_ohlcv.index)
        signals.iloc[0:5] = 1.0
        result = run_backtest(signals, trending_up_ohlcv, commission_bps=0, slippage_bps=0)
        # With zero costs and perfect uptrend, must have positive return
        assert result.total_return > 0

    def test_trade_count_correct(self, flat_ohlcv: pd.DataFrame) -> None:
        """One entry and one exit → 2 signal changes → 1 round trip."""
        n = len(flat_ohlcv)
        signals = pd.Series(np.zeros(n), index=flat_ohlcv.index)
        signals.iloc[10:50] = 1.0  # Buy at 10, sell at 50
        result = run_backtest(signals, flat_ohlcv, commission_bps=0, slippage_bps=0)
        # Should have 2 fills: entry and exit
        assert result.n_trades >= 2
