"""Alpaca paper/live trading broker adapter."""

from __future__ import annotations

import dataclasses

import alpaca_trade_api as tradeapi

from alpha_engine.execution.broker import Broker
from alpha_engine.execution.order import Order


class AlpacaBroker(Broker):
    """Broker adapter for Alpaca REST API (paper or live).

    Parameters
    ----------
    api_key : str
        Alpaca API key ID.
    secret_key : str
        Alpaca secret key.
    base_url : str
        Alpaca base URL. Use ``https://paper-api.alpaca.markets`` for paper trading.
    """

    def __init__(self, api_key: str, secret_key: str, base_url: str) -> None:
        self._api = tradeapi.REST(api_key, secret_key, base_url)

    def submit(self, order: Order) -> Order:
        """Submit a market order and return an updated copy."""
        resp = self._api.submit_order(
            symbol=order.symbol,
            qty=order.qty,
            side=order.side,
            type="market",
            time_in_force="day",
        )
        return dataclasses.replace(order, order_id=str(resp.id), status=str(resp.status))
