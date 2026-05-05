#!/usr/bin/env python
"""Smoke test CLI — fetch SPY data, run all strategies, print performance summary.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --ticker AAPL --years 5

This script validates the full pipeline end-to-end:
  data fetch → strategy signals → backtest → performance metrics → display
"""

import argparse
import sys
from pathlib import Path

# Ensure src/ is on the path when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from alpha_engine.strategies import BollingerMeanReversionStrategy, MomentumStrategy
from quantcore.data.fetcher import fetch_ohlcv


def print_result_table(results: list[tuple[str, object]]) -> None:
    """Print a nicely formatted performance summary table."""
    header = f"{'Strategy':<35} {'Total Return':>13} {'Sharpe':>8} {'Max DD':>9} {'Trades':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for name, res in results:
        total_ret = f"{res.total_return * 100:+.2f}%"
        sharpe = f"{res.sharpe:.3f}" if not np.isnan(res.sharpe) else "  N/A"
        max_dd = f"{res.max_drawdown * 100:.2f}%"
        trades = str(res.n_trades)
        print(f"{name:<35} {total_ret:>13} {sharpe:>8} {max_dd:>9} {trades:>8}")
    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strategies on historical data")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol (default: SPY)")
    parser.add_argument("--years", type=int, default=5, help="Years of history (default: 5)")
    args = parser.parse_args()

    end = pd.Timestamp.today()
    start = end - pd.DateOffset(years=args.years)

    print(f"\nFetching {args.ticker} data: {start.date()} → {end.date()}")
    df = fetch_ohlcv(
        ticker=args.ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        use_cache=True,
    )
    print(f"Loaded {len(df):,} bars\n")

    strategies = [
        MomentumStrategy(lookback=252, skip=21),
        MomentumStrategy(lookback=126, skip=21),
        BollingerMeanReversionStrategy(window=20, num_std=2.0),
        BollingerMeanReversionStrategy(window=20, num_std=1.5),
    ]

    results = []
    for strat in strategies:
        result = strat.run(df, commission_bps=10, slippage_bps=5)
        results.append((strat.name, result))
        print(f"  ✓ {strat.name}")

    print(f"\nPerformance Summary — {args.ticker} ({args.years}y)\n")
    print_result_table(results)
    print()


if __name__ == "__main__":
    main()
