"""Strategy Comparison Framework.

Compare multiple strategies head-to-head on the same OHLCV data.
Returns a structured ComparisonResult with per-strategy BacktestResult objects
and a summary DataFrame ranked by Sharpe ratio.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha_engine.backtest.types import BacktestResult
from alpha_engine.strategies.base import Strategy


@dataclass(frozen=True)
class ComparisonResult:
    """Aggregated comparison output for multiple strategies.

    Attributes
    ----------
    results : dict[str, BacktestResult]
        Per-strategy backtest result, keyed by strategy name.
    summary : pd.DataFrame
        DataFrame with one row per strategy, columns: sharpe, total_return,
        max_drawdown, calmar, annualised_vol. Sorted descending by Sharpe.
    best_sharpe : str
        Name of the strategy with the highest Sharpe ratio.
    """

    results: dict[str, BacktestResult]
    summary: pd.DataFrame
    best_sharpe: str


def compare_strategies(
    strategies: list[Strategy],
    df: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission_bps: int = 10,
    slippage_bps: int = 5,
) -> ComparisonResult:
    """Run all strategies on the same data and summarise results.

    Parameters
    ----------
    strategies : list[Strategy]
        Strategies to compare. Must be instantiated (not classes).
    df : pd.DataFrame
        OHLCV data passed to each strategy's ``run()`` method.
    initial_capital, commission_bps, slippage_bps
        Forwarded to each strategy's backtest engine.

    Returns
    -------
    ComparisonResult
        Aggregated results with summary table and best-Sharpe name.
    """
    results: dict[str, BacktestResult] = {}
    for strategy in strategies:
        results[strategy.name] = strategy.run(
            df,
            initial_capital=initial_capital,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )

    rows = []
    for name, res in results.items():
        rows.append(
            {
                "strategy": name,
                "sharpe": round(res.sharpe, 4),
                "total_return": round(res.total_return, 4),
                "max_drawdown": round(res.max_drawdown, 4),
                "calmar": round(res.calmar, 4),
            }
        )

    summary = (
        pd.DataFrame(rows)
        .set_index("strategy")
        .sort_values("sharpe", ascending=False)
    )

    best_sharpe = summary["sharpe"].idxmax()

    return ComparisonResult(results=results, summary=summary, best_sharpe=best_sharpe)
