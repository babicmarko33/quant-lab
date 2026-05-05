"""Tests for the walk-forward validation harness."""

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest.types import BacktestResult
from alpha_engine.backtest.walk_forward import WalkForwardResult, walk_forward_validation
from alpha_engine.strategies.momentum import MomentumStrategy


@pytest.fixture
def long_ohlcv() -> pd.DataFrame:
    """7 years of daily OHLCV data — enough for meaningful WF splits."""
    rng = np.random.default_rng(99)
    n = 7 * 252
    dates = pd.date_range("2015-01-01", periods=n, freq="B")
    log_ret = rng.normal(0.0003, 0.012, n)
    close = 100 * np.exp(np.cumsum(log_ret))
    opens = np.roll(close, 1)
    opens[0] = 100.0
    return pd.DataFrame(
        {"open": opens, "high": close * 1.005, "low": close * 0.995, "close": close, "volume": 1_000_000},
        index=dates,
    )


class TestWalkForwardResult:
    def test_contains_list_of_backtest_results(self, long_ohlcv: pd.DataFrame) -> None:
        strat = MomentumStrategy(lookback=126, skip=21)
        wf = walk_forward_validation(strat, long_ohlcv, train_size=252, test_size=63)
        assert isinstance(wf, WalkForwardResult)
        assert isinstance(wf.folds, list)
        assert all(isinstance(f, BacktestResult) for f in wf.folds)

    def test_minimum_folds_created(self, long_ohlcv: pd.DataFrame) -> None:
        """With 7 years data, train=252, test=63 → at least 5 OOS folds."""
        strat = MomentumStrategy(lookback=126, skip=21)
        wf = walk_forward_validation(strat, long_ohlcv, train_size=252, test_size=63)
        assert len(wf.folds) >= 5

    def test_no_look_ahead_across_folds(self, long_ohlcv: pd.DataFrame) -> None:
        """Each fold's test period must not overlap with train period."""
        strat = MomentumStrategy(lookback=126, skip=21)
        wf = walk_forward_validation(strat, long_ohlcv, train_size=252, test_size=63)
        # Check fold slices have no overlap
        for i, (train_slice, test_slice) in enumerate(wf.fold_slices):
            assert test_slice.start >= train_slice.stop, f"Fold {i}: test overlaps train"

    def test_embargo_prevents_leakage(self, long_ohlcv: pd.DataFrame) -> None:
        """With embargo > 0, test start must be at least embargo days after train end."""
        embargo = 5
        strat = MomentumStrategy(lookback=126, skip=21)
        wf = walk_forward_validation(
            strat, long_ohlcv, train_size=252, test_size=63, embargo=embargo
        )
        for i, (train_slice, test_slice) in enumerate(wf.fold_slices):
            gap = test_slice.start - train_slice.stop
            assert gap >= embargo, f"Fold {i}: embargo gap {gap} < {embargo}"

    def test_aggregate_stats_available(self, long_ohlcv: pd.DataFrame) -> None:
        strat = MomentumStrategy(lookback=126, skip=21)
        wf = walk_forward_validation(strat, long_ohlcv, train_size=252, test_size=63)
        assert hasattr(wf, "mean_sharpe")
        assert hasattr(wf, "mean_return")
        assert not np.isnan(wf.mean_sharpe)
        assert not np.isnan(wf.mean_return)
