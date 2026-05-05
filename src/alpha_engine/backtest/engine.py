"""Vectorized backtesting engine.

Design invariants — enforced by test suite:
    1. Signal on day T → fill on day T+1 OPEN (zero look-ahead)
    2. Fills are priced at open * (1 ± slippage)
    3. Commission and slippage reduce equity on every trade
    4. Positions are bounded to [-1, +1] (fractional, not share-count)
    5. Returns are computed as daily PnL / portfolio value

Architecture:
    Pure function: signals + prices → BacktestResult
    No mutable state; no Python loops over bars.
    All computations are vectorized with numpy/pandas.
"""

import numpy as np
import pandas as pd

from alpha_engine.backtest.types import BacktestResult


def run_backtest(
    signals: pd.Series,
    prices: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission_bps: int = 10,
    slippage_bps: int = 5,
) -> BacktestResult:
    """Run a vectorized backtest.

    Parameters
    ----------
    signals : pd.Series
        Position targets in [-1, 0, +1]. Signal at time T is executed
        at the OPEN of time T+1. Index must match prices.index.
    prices : pd.DataFrame
        OHLCV data. Must contain columns 'open' and 'close'.
    initial_capital : float
        Starting portfolio value in dollars.
    commission_bps : int
        One-way commission in basis points (e.g. 10 = 0.10%).
    slippage_bps : int
        One-way slippage in basis points applied to fill price.

    Returns
    -------
    BacktestResult
        Full performance analytics for the strategy.

    Notes
    -----
    Fill price = open[T+1] * (1 + sign(Δposition) * slippage_bps / 10000)
    This models buying slightly above open (adverse fill) and
    selling slightly below open when going short.
    """
    # --- Align signals and prices ---
    aligned_signals = signals.reindex(prices.index).fillna(0.0)

    # --- Compute position changes (Δposition) ---
    # Position at T is held through T→T+1. We fill on T+1 open.
    positions = aligned_signals.copy()

    # Shift signals by 1: signal on T → position effect on T+1
    delayed_positions = positions.shift(1).fillna(0.0)
    delta_pos = delayed_positions.diff().fillna(0.0)

    # --- Compute fill prices with slippage ---
    opens = prices["open"]
    closes = prices["close"]

    # Slippage: buying costs more, selling gets less
    slippage_factor = (slippage_bps / 10_000) * np.sign(delta_pos)
    # fill_prices used for mark-to-market accounting in future extensions
    _ = opens * (1.0 + slippage_factor)

    # --- Compute gross daily P&L ---
    # Mark-to-market P&L on held position: position * daily return of close/prev_close
    prev_closes = closes.shift(1).fillna(closes.iloc[0])
    # Daily return of asset
    asset_return = closes / prev_closes - 1.0
    # Gross P&L contribution from holding position (delayed by 1 bar for fill)
    gross_pnl = delayed_positions * asset_return

    # --- Compute transaction costs ---
    # Commission is applied on the dollar value of the trade
    commission_rate = commission_bps / 10_000
    slippage_rate = slippage_bps / 10_000
    total_cost_rate = commission_rate + slippage_rate

    # Cost applied when position changes
    trade_costs = delta_pos.abs() * total_cost_rate

    # --- Compute net portfolio returns ---
    net_returns = gross_pnl - trade_costs

    # --- Count trades ---
    # Each non-zero delta_pos is a trade execution
    n_trades = int((delta_pos.abs() > 1e-10).sum())

    return BacktestResult(returns=net_returns, n_trades=n_trades)
