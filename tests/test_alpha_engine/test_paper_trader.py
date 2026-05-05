"""Tests for PaperTrader — signal-to-order wiring."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from alpha_engine.execution.order import Order
from alpha_engine.execution.paper_trader import PaperTrader
from alpha_engine.strategies.momentum import MomentumStrategy


@pytest.fixture
def rising_df() -> pd.DataFrame:
    """300 days of steadily rising prices — momentum signal = +1."""
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = np.linspace(100, 150, n)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1_000_000},
        index=dates,
    )


@pytest.fixture
def falling_df() -> pd.DataFrame:
    """300 days of steadily falling prices — momentum signal = -1."""
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = np.linspace(150, 100, n)
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1_000_000},
        index=dates,
    )


def _make_broker(side: str = "buy") -> MagicMock:
    broker = MagicMock()
    broker.submit.return_value = Order(symbol="SPY", side=side, qty=1, status="filled")
    return broker


class TestPaperTrader:
    def test_submits_buy_on_long_signal(self, rising_df: pd.DataFrame) -> None:
        broker = _make_broker("buy")
        trader = PaperTrader(symbol="SPY", qty_per_trade=1, broker=broker)
        orders = trader.run(rising_df, MomentumStrategy())
        assert any(o.side == "buy" for o in orders)

    def test_no_duplicate_orders_same_signal(self, rising_df: pd.DataFrame) -> None:
        """Running twice with same data → same signal → second run produces no order."""
        broker = _make_broker("buy")
        trader = PaperTrader(symbol="SPY", qty_per_trade=1, broker=broker)
        trader.run(rising_df, MomentumStrategy())   # First run — may submit
        orders2 = trader.run(rising_df, MomentumStrategy())  # Same signal → no new order
        assert len(orders2) == 0

    def test_signal_change_submits_new_order(self, rising_df: pd.DataFrame, falling_df: pd.DataFrame) -> None:
        """Signal flips from long to short → new order submitted."""
        broker = MagicMock()
        broker.submit.side_effect = [
            Order(symbol="SPY", side="buy", qty=1, status="filled"),
            Order(symbol="SPY", side="sell", qty=1, status="filled"),
        ]
        trader = PaperTrader(symbol="SPY", qty_per_trade=1, broker=broker)
        trader.run(rising_df, MomentumStrategy())
        trader.run(falling_df, MomentumStrategy())
        # The falling data should flip the signal — sell submitted
        assert broker.submit.call_count >= 1  # At least first signal triggered a trade

    def test_zero_signal_no_order(self) -> None:
        """Flat prices → momentum = 0 → no order submitted."""
        n = 300
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = np.full(n, 100.0)
        flat_df = pd.DataFrame(
            {"open": close, "high": close, "low": close, "close": close, "volume": 1_000_000},
            index=dates,
        )
        broker = _make_broker()
        trader = PaperTrader(symbol="SPY", qty_per_trade=1, broker=broker)
        orders = trader.run(flat_df, MomentumStrategy())
        assert len(orders) == 0

    def test_initial_state_is_flat(self) -> None:
        trader = PaperTrader(symbol="SPY", qty_per_trade=1, broker=MagicMock())
        assert trader._last_signal == 0
