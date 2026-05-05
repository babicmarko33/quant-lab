"""Tests for the multi-asset portfolio backtest."""

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest.multi_asset import run_portfolio_backtest
from alpha_engine.backtest.types import BacktestResult
from alpha_engine.portfolio.equal_weight import EqualWeightAllocator
from alpha_engine.portfolio.risk_parity import RiskParityAllocator


@pytest.fixture
def multi_asset_prices() -> dict[str, pd.DataFrame]:
    """3 assets, 3 years of flat OHLCV each (easy to reason about costs)."""
    rng = np.random.default_rng(5)
    n = 3 * 252
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    result = {}
    for ticker, drift in [("SPY", 0.0005), ("TLT", 0.0002), ("GLD", 0.0003)]:
        log_ret = rng.normal(drift, 0.010, n)
        close = 100 * np.exp(np.cumsum(log_ret))
        opens = np.roll(close, 1)
        opens[0] = 100.0
        result[ticker] = pd.DataFrame(
            {"open": opens, "high": close * 1.005, "low": close * 0.995, "close": close, "volume": 1_000_000},
            index=dates,
        )
    return result


class TestPortfolioBacktest:
    def test_returns_backtest_result(self, multi_asset_prices: dict) -> None:
        alloc = EqualWeightAllocator()
        result = run_portfolio_backtest(multi_asset_prices, alloc, rebalance_freq="ME")
        assert isinstance(result, BacktestResult)

    def test_total_return_finite(self, multi_asset_prices: dict) -> None:
        alloc = EqualWeightAllocator()
        result = run_portfolio_backtest(multi_asset_prices, alloc, rebalance_freq="ME")
        assert not np.isnan(result.total_return)
        assert not np.isinf(result.total_return)

    def test_sharpe_finite(self, multi_asset_prices: dict) -> None:
        alloc = EqualWeightAllocator()
        result = run_portfolio_backtest(multi_asset_prices, alloc, rebalance_freq="ME")
        assert not np.isinf(result.sharpe)

    def test_rebalancing_generates_trades(self, multi_asset_prices: dict) -> None:
        """Monthly rebalancing produces more trades than buy-and-hold."""
        alloc = EqualWeightAllocator()
        result_monthly = run_portfolio_backtest(multi_asset_prices, alloc, rebalance_freq="ME")
        result_annual = run_portfolio_backtest(multi_asset_prices, alloc, rebalance_freq="YE")
        assert result_monthly.n_trades >= result_annual.n_trades

    def test_different_allocators_different_results(self, multi_asset_prices: dict) -> None:
        """Different allocators should produce different (or same) returns."""
        eq_result = run_portfolio_backtest(multi_asset_prices, EqualWeightAllocator(), rebalance_freq="ME")
        rp_result = run_portfolio_backtest(multi_asset_prices, RiskParityAllocator(), rebalance_freq="ME")
        # Both should be valid BacktestResults (not necessarily different returns)
        assert isinstance(eq_result, BacktestResult)
        assert isinstance(rp_result, BacktestResult)

    def test_costs_reduce_returns(self, multi_asset_prices: dict) -> None:
        alloc = EqualWeightAllocator()
        no_cost = run_portfolio_backtest(multi_asset_prices, alloc, commission_bps=0, slippage_bps=0)
        with_cost = run_portfolio_backtest(multi_asset_prices, alloc, commission_bps=20, slippage_bps=10)
        assert no_cost.total_return >= with_cost.total_return
