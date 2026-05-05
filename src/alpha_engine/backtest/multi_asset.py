"""Multi-asset portfolio backtester.

Takes per-asset OHLCV DataFrames + an Allocator, rebalances at specified
frequency, and returns a single ``BacktestResult`` for the combined portfolio.

Workflow:
    1. For each rebalance date, call ``allocator.fit(returns_window)``
       to compute target weights.
    2. Compute portfolio return for each day as the weighted sum of
       individual asset returns.
    3. Apply commission + slippage costs on the weight *change* at each
       rebalance (turnover-based).
    4. Compound daily returns into an equity curve.
    5. Pass equity curve to ``BacktestResult`` constructor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.backtest.types import BacktestResult
from alpha_engine.portfolio.allocator import Allocator

# Minimum lookback in trading days to compute covariance reliably
_MIN_LOOKBACK = 30


def run_portfolio_backtest(
    prices: dict[str, pd.DataFrame],
    allocator: Allocator,
    *,
    rebalance_freq: str = "ME",
    lookback: int = 252,
    initial_capital: float = 100_000.0,
    commission_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> BacktestResult:
    """Run a multi-asset portfolio backtest.

    Parameters
    ----------
    prices : dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame. All DataFrames must share
        a common datetime index (or be aligned). Required columns: ``close``.
    allocator : Allocator
        Portfolio weight allocator (EqualWeight, MV, RiskParity, CVaR …).
    rebalance_freq : str
        pandas offset alias for rebalancing frequency (e.g. ``"ME"`` = month end,
        ``"YE"`` = year end, ``"W"`` = weekly). Default ``"ME"``.
    lookback : int
        Number of trading days of history fed to the allocator each rebalance.
    initial_capital : float
        Starting portfolio value in base currency.
    commission_bps : float
        One-way commission in basis points applied to each rebalance turnover.
    slippage_bps : float
        One-way market-impact slippage in bps on each rebalance turnover.

    Returns
    -------
    BacktestResult
        Aggregate portfolio performance with equity curve and standard metrics.
    """
    tickers = list(prices.keys())
    if not tickers:
        raise ValueError("prices must contain at least one asset")

    # Align close prices into a single DataFrame
    close_df = pd.concat(
        {ticker: prices[ticker]["close"] for ticker in tickers}, axis=1
    )
    close_df.columns = pd.Index(tickers)
    close_df = close_df.sort_index().dropna()

    if len(close_df) < _MIN_LOOKBACK + 2:
        raise ValueError(
            f"Insufficient data: need at least {_MIN_LOOKBACK + 2} rows, got {len(close_df)}"
        )

    # Daily returns matrix (T × n)
    daily_returns = close_df.pct_change().dropna()

    # Build rebalance dates (end-of-period days that exist in the index)
    rebal_dates_series = close_df.index.to_series().resample(rebalance_freq).last()
    rebal_dates = sorted(set(rebal_dates_series.values) & set(close_df.index))

    # State
    equity = initial_capital
    equity_curve: list[float] = [equity]
    current_weights = np.full(len(tickers), 1.0 / len(tickers))  # Start equal weight
    n_trades = 0
    cost_factor = (commission_bps + slippage_bps) / 10_000.0  # per unit turnover, one-way

    rebal_set = set(rebal_dates)

    for date, row in daily_returns.iterrows():
        # Rebalance on designated dates (if enough history)
        if date in rebal_set:
            hist_end_loc = close_df.index.get_loc(date)
            hist_start_loc = max(0, hist_end_loc - lookback)
            history_slice = daily_returns.iloc[hist_start_loc:hist_end_loc]
            if len(history_slice) >= _MIN_LOOKBACK:
                new_weights_series = allocator.fit(history_slice)
                new_weights = new_weights_series.reindex(tickers).fillna(0.0).values
                # Normalize (safety)
                if new_weights.sum() > 1e-10:
                    new_weights = new_weights / new_weights.sum()
                # Cost on turnover (sum of absolute weight changes, halved for one-way)
                turnover = np.abs(new_weights - current_weights).sum() / 2.0
                equity *= 1.0 - cost_factor * turnover
                n_trades += int(np.round(turnover * 10))  # approximate trade count
                current_weights = new_weights

        # Apply weighted daily return
        asset_returns = row.reindex(tickers).fillna(0.0).values
        portfolio_return = float(current_weights @ asset_returns)
        equity *= 1.0 + portfolio_return
        equity_curve.append(equity)

    equity_series = pd.Series(equity_curve, dtype=float)
    # Derive daily returns from the equity curve (skip the first element = initial value)
    portfolio_returns = equity_series.pct_change().dropna()
    # Align index to trading days (reuse daily_returns index)
    portfolio_returns.index = daily_returns.index[: len(portfolio_returns)]
    return BacktestResult(returns=portfolio_returns, n_trades=n_trades)
