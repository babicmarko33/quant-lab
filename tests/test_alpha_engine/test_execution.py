"""Tests for the execution layer: Order, Broker ABC, AlpacaBroker, PaperTrader."""

from unittest.mock import MagicMock, patch

import pytest

from alpha_engine.execution.order import Order


class TestOrder:
    def test_order_defaults(self) -> None:
        o = Order(symbol="SPY", side="buy", qty=10)
        assert o.status == "pending"
        assert o.order_id is not None  # auto-generated UUID

    def test_order_auto_generates_order_id(self) -> None:
        o1 = Order(symbol="SPY", side="buy", qty=1)
        o2 = Order(symbol="SPY", side="buy", qty=1)
        assert o1.order_id != o2.order_id

    def test_order_side_validation(self) -> None:
        with pytest.raises(ValueError, match="side must be one of"):
            Order(symbol="SPY", side="short", qty=10)

    def test_order_qty_positive(self) -> None:
        with pytest.raises(ValueError, match="qty must be positive"):
            Order(symbol="SPY", side="buy", qty=-5)

    def test_order_qty_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="qty must be positive"):
            Order(symbol="SPY", side="buy", qty=0)

    def test_order_repr_contains_symbol(self) -> None:
        o = Order(symbol="AAPL", side="sell", qty=5)
        assert "AAPL" in repr(o)

    def test_order_explicit_order_id(self) -> None:
        o = Order(symbol="SPY", side="buy", qty=1, order_id="custom-id-123")
        assert o.order_id == "custom-id-123"


class TestBroker:
    def test_broker_is_abstract(self) -> None:
        from alpha_engine.execution.broker import Broker

        with pytest.raises(TypeError):
            Broker()  # type: ignore[abstract]


class TestAlpacaBroker:
    def test_submit_buy_returns_filled_order(self) -> None:
        from alpha_engine.execution.alpaca_broker import AlpacaBroker

        mock_resp = MagicMock()
        mock_resp.id = "abc123"
        mock_resp.status = "filled"

        mock_api = MagicMock()
        mock_api.submit_order.return_value = mock_resp

        with patch("alpha_engine.execution.alpaca_broker.tradeapi.REST", return_value=mock_api):
            broker = AlpacaBroker(
                api_key="k", secret_key="s", base_url="https://paper-api.alpaca.markets"
            )
            order = Order(symbol="SPY", side="buy", qty=1)
            filled = broker.submit(order)

        assert filled.status == "filled"
        assert filled.order_id == "abc123"

    def test_submit_calls_alpaca_with_correct_params(self) -> None:
        from alpha_engine.execution.alpaca_broker import AlpacaBroker

        mock_resp = MagicMock(id="xyz", status="filled")
        mock_api = MagicMock()
        mock_api.submit_order.return_value = mock_resp

        with patch("alpha_engine.execution.alpaca_broker.tradeapi.REST", return_value=mock_api):
            broker = AlpacaBroker(api_key="k", secret_key="s", base_url="https://paper-api.alpaca.markets")
            broker.submit(Order(symbol="AAPL", side="sell", qty=5))

        mock_api.submit_order.assert_called_once_with(
            symbol="AAPL",
            qty=5,
            side="sell",
            type="market",
            time_in_force="day",
        )

    def test_original_order_not_mutated(self) -> None:
        from alpha_engine.execution.alpaca_broker import AlpacaBroker

        mock_api = MagicMock()
        mock_api.submit_order.return_value = MagicMock(id="new-id", status="filled")

        with patch("alpha_engine.execution.alpaca_broker.tradeapi.REST", return_value=mock_api):
            broker = AlpacaBroker(api_key="k", secret_key="s", base_url="https://paper-api.alpaca.markets")
            original = Order(symbol="SPY", side="buy", qty=1)
            original_id = original.order_id
            broker.submit(original)

        assert original.order_id == original_id  # Unchanged
        assert original.status == "pending"
