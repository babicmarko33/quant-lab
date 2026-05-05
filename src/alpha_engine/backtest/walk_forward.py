"""Walk-forward validation harness.

Design principles:
  - Anchored (expanding) train window with fixed test size
  - Embargo period between train end and test start (prevents leakage from
    autocorrelated features or stale order books)
  - No data snooping: test window is completely out-of-sample
  - Returns list of BacktestResult (one per fold) plus aggregate statistics

Reference:
  López de Prado, M. (2018). Advances in Financial Machine Learning.
  Chapter 7: Cross-Validation in Finance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha_engine.backtest.types import BacktestResult
from alpha_engine.strategies.base import Strategy


@dataclass
class WalkForwardResult:
    """Aggregate results from a walk-forward validation run.

    Attributes
    ----------
    folds : list[BacktestResult]
        One BacktestResult per OOS test fold.
    fold_slices : list[tuple[slice, slice]]
        (train_slice, test_slice) for each fold. Useful for leak auditing.
    mean_sharpe : float
        Mean annualized Sharpe across all OOS folds.
    mean_return : float
        Mean total return across all OOS folds.
    """

    folds: list[BacktestResult] = field(default_factory=list)
    fold_slices: list[tuple[slice, slice]] = field(default_factory=list)
    mean_sharpe: float = field(init=False)
    mean_return: float = field(init=False)

    def __post_init__(self) -> None:
        if self.folds:
            sharpes = [f.sharpe for f in self.folds if not np.isnan(f.sharpe)]
            returns = [f.total_return for f in self.folds]
            self.mean_sharpe = float(np.mean(sharpes)) if sharpes else float("nan")
            self.mean_return = float(np.mean(returns))
        else:
            self.mean_sharpe = float("nan")
            self.mean_return = float("nan")


def walk_forward_validation(
    strategy: Strategy,
    df: pd.DataFrame,
    train_size: int = 252,
    test_size: int = 63,
    embargo: int = 0,
    initial_capital: float = 100_000.0,
    commission_bps: int = 10,
    slippage_bps: int = 5,
) -> WalkForwardResult:
    """Run anchored walk-forward validation.

    The train window expands with each fold (anchored/expanding).
    Test window is strictly out-of-sample with an optional embargo gap.

    Fold construction (step size = test_size):
      Train: [0, train_end)
      Gap:   [train_end, train_end + embargo)   ← purged, not used
      Test:  [train_end + embargo, test_end)

    Parameters
    ----------
    strategy : Strategy
        Any concrete Strategy subclass.
    df : pd.DataFrame
        Full OHLCV history (train + OOS combined).
    train_size : int
        Minimum training window in bars (anchored → only minimum matters).
    test_size : int
        OOS test window size in bars.
    embargo : int
        Number of bars between train end and test start.
    initial_capital : float
        Starting capital per fold.
    commission_bps : int
        One-way commission in basis points.
    slippage_bps : int
        One-way slippage in basis points.

    Returns
    -------
    WalkForwardResult
    """
    n = len(df)
    folds: list[BacktestResult] = []
    fold_slices: list[tuple[slice, slice]] = []

    # Anchored expanding window: train starts at 0, expands each step
    train_end = train_size
    while train_end + embargo + test_size <= n:
        test_start = train_end + embargo
        test_end = test_start + test_size

        train_slice = slice(0, train_end)
        test_slice = slice(test_start, test_end)

        # Generate signals on FULL data up to test_end (no future data)
        # but only backtest on test slice (OOS)
        test_df = df.iloc[test_slice].copy()

        # Strategy generates signals for test window only, using
        # data available up to that point (signals from FULL history up to test_end)
        full_signals = strategy.generate_signals(df.iloc[:test_end])
        oos_signals = full_signals.iloc[test_slice]

        from alpha_engine.backtest.engine import run_backtest

        fold_result = run_backtest(
            oos_signals,
            test_df,
            initial_capital=initial_capital,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )

        folds.append(fold_result)
        fold_slices.append((train_slice, test_slice))

        # Expand train end by one test period
        train_end += test_size

    return WalkForwardResult(folds=folds, fold_slices=fold_slices)
