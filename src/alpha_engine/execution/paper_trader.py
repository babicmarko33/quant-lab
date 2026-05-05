"""PaperTrader — wires a Strategy's signals to a Broker for paper order execution."""

from __future__ import annotations

import pandas as pd

from alpha_engine.execution.broker import Broker
from alpha_engine.execution.order import Order
from alpha_engine.strategies.base import Strategy


class PaperTrader:
    """Executes paper trades based on strategy signals.

    Maintains state of the last submitted signal to avoid duplicate orders
    when the signal has not changed.

    Parameters
    ----------
    symbol : str
        Ticker symbol to trade.
    qty_per_trade : float
        Number of units per order.
    broker : Broker
        Broker adapter used to submit orders.
    """

    def __init__(self, symbol: str, qty_per_trade: float, broker: Broker) -> None:
        self.symbol = symbol
        self.qty_per_trade = qty_per_trade
        self._broker = broker
        self._last_signal: int = 0

    def run(self, df: pd.DataFrame, strategy: Strategy) -> list[Order]:
        """Generate signals from ``strategy`` and submit order if signal changed.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with at least ``close`` column.
        strategy : Strategy
            Strategy instance used for signal generation.

        Returns
        -------
        list[Order]
            Orders submitted this call (empty if signal unchanged).
        """
        signals = strategy.generate_signals(df)
        current_signal = int(signals.iloc[-1])
        submitted: list[Order] = []

        if current_signal != self._last_signal:
            if current_signal == 1:
                order = Order(symbol=self.symbol, side="buy", qty=self.qty_per_trade)
                submitted.append(self._broker.submit(order))
            elif current_signal == -1:
                order = Order(symbol=self.symbol, side="sell", qty=self.qty_per_trade)
                submitted.append(self._broker.submit(order))
            self._last_signal = current_signal

        return submitted
