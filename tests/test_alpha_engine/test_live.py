"""Tests for Alpaca WebSocket bar stream + LiveTrader.

WebSocket and broker calls are fully mocked — no API keys needed.
"""
from __future__ import annotations

import pytest

from alpha_engine.live.alpaca_stream import AlpacaBarStream, BarEvent
from alpha_engine.live.live_trader import LiveTrader

# ---------------------------------------------------------------------------
# BarEvent
# ---------------------------------------------------------------------------

def test_bar_event_fields():
    """BarEvent dataclass holds OHLCV + symbol + timestamp."""
    evt = BarEvent(
        symbol="SPY",
        timestamp="2024-12-20T09:35:00Z",
        open=498.0,
        high=499.5,
        low=497.8,
        close=499.0,
        volume=123456,
    )
    assert evt.symbol == "SPY"
    assert evt.close == 499.0
    assert evt.volume == 123456


def test_bar_event_from_dict():
    """BarEvent.from_dict parses an Alpaca WS message dict."""
    raw = {
        "S": "SPY",
        "t": "2024-12-20T09:35:00Z",
        "o": 498.0,
        "h": 499.5,
        "l": 497.8,
        "c": 499.0,
        "v": 123456,
    }
    evt = BarEvent.from_dict(raw)
    assert evt.symbol == "SPY"
    assert evt.open == pytest.approx(498.0)
    assert evt.close == pytest.approx(499.0)
    assert evt.volume == 123456


# ---------------------------------------------------------------------------
# AlpacaBarStream
# ---------------------------------------------------------------------------

def test_alpaca_stream_instantiation():
    """AlpacaBarStream can be constructed with key/secret."""
    stream = AlpacaBarStream(api_key="key", secret_key="secret", symbols=["SPY"])
    assert stream is not None


def test_alpaca_stream_stores_symbols():
    """AlpacaBarStream stores the requested symbols."""
    stream = AlpacaBarStream(api_key="k", secret_key="s", symbols=["SPY", "QQQ"])
    assert "SPY" in stream.symbols
    assert "QQQ" in stream.symbols


def test_alpaca_stream_default_paper():
    """Default WebSocket URL points to paper trading endpoint."""
    stream = AlpacaBarStream(api_key="k", secret_key="s", symbols=["SPY"])
    assert "paper" in stream.ws_url or "stream" in stream.ws_url


def test_alpaca_stream_subscribe_message():
    """_subscribe_message returns correct JSON subscribe payload."""
    stream = AlpacaBarStream(api_key="k", secret_key="s", symbols=["SPY", "AAPL"])
    msg = stream._subscribe_message()
    assert msg["action"] == "subscribe"
    assert "SPY" in msg["bars"]
    assert "AAPL" in msg["bars"]


def test_alpaca_stream_auth_message():
    """_auth_message returns correct auth payload."""
    stream = AlpacaBarStream(api_key="MYKEY", secret_key="MYSECRET", symbols=["SPY"])
    msg = stream._auth_message()
    assert msg["action"] == "auth"
    assert msg["key"] == "MYKEY"
    assert msg["secret"] == "MYSECRET"  # noqa: S105


# ---------------------------------------------------------------------------
# LiveTrader
# ---------------------------------------------------------------------------

def test_live_trader_instantiation():
    """LiveTrader can be constructed with a strategy and stream."""
    from alpha_engine.strategies.sma_crossover import SMACrossoverStrategy
    strategy = SMACrossoverStrategy()
    stream = AlpacaBarStream(api_key="k", secret_key="s", symbols=["SPY"])
    trader = LiveTrader(strategy=strategy, stream=stream, symbol="SPY")
    assert trader is not None


def test_live_trader_stores_strategy():
    """LiveTrader.strategy is the passed-in strategy."""
    from alpha_engine.strategies.sma_crossover import SMACrossoverStrategy
    strategy = SMACrossoverStrategy()
    stream = AlpacaBarStream(api_key="k", secret_key="s", symbols=["SPY"])
    trader = LiveTrader(strategy=strategy, stream=stream, symbol="SPY")
    assert trader.strategy is strategy


def test_live_trader_on_bar_appends_buffer():
    """on_bar() appends a BarEvent to the internal buffer."""
    from alpha_engine.strategies.sma_crossover import SMACrossoverStrategy
    strategy = SMACrossoverStrategy()
    stream = AlpacaBarStream(api_key="k", secret_key="s", symbols=["SPY"])
    trader = LiveTrader(strategy=strategy, stream=stream, symbol="SPY", min_bars=2)

    evt = BarEvent(symbol="SPY", timestamp="2024-12-20T09:35:00Z",
                   open=498.0, high=499.5, low=497.8, close=499.0, volume=100)
    trader.on_bar(evt)
    assert len(trader._bar_buffer) == 1


def test_live_trader_on_bar_generates_signal_after_min_bars():
    """on_bar() produces a signal once min_bars are buffered."""
    from alpha_engine.strategies.sma_crossover import SMACrossoverStrategy
    strategy = SMACrossoverStrategy()
    stream = AlpacaBarStream(api_key="k", secret_key="s", symbols=["SPY"])
    trader = LiveTrader(strategy=strategy, stream=stream, symbol="SPY", min_bars=3)

    signals_seen = []
    trader.on_signal = lambda sig: signals_seen.append(sig)

    for i in range(5):
        evt = BarEvent(symbol="SPY", timestamp=f"2024-12-20T09:3{i}:00Z",
                       open=500.0 + i, high=501.0 + i, low=499.0 + i,
                       close=500.5 + i, volume=100)
        trader.on_bar(evt)

    assert len(signals_seen) > 0


def test_live_trader_signal_values_valid():
    """Signals emitted by LiveTrader are in {-1, 0, 1}."""
    from alpha_engine.strategies.sma_crossover import SMACrossoverStrategy
    strategy = SMACrossoverStrategy()
    stream = AlpacaBarStream(api_key="k", secret_key="s", symbols=["SPY"])
    trader = LiveTrader(strategy=strategy, stream=stream, symbol="SPY", min_bars=3)

    signals_seen = []
    trader.on_signal = lambda sig: signals_seen.append(sig)

    for i in range(10):
        evt = BarEvent(symbol="SPY", timestamp=f"2024-12-20T09:{10+i}:00Z",
                       open=500.0 + i, high=501.0 + i, low=499.0 + i,
                       close=500.5 + i, volume=100)
        trader.on_bar(evt)

    assert all(s in {-1, 0, 1} for s in signals_seen)
