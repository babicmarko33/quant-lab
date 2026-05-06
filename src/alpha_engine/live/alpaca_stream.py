"""Alpaca WebSocket real-time bar stream.

Connects to Alpaca's market data stream and emits :class:`BarEvent`
objects for each 1-minute bar received.

Usage::

    stream = AlpacaBarStream(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
        symbols=["SPY", "QQQ"],
    )
    stream.on_bar = my_handler   # callable(BarEvent) -> None
    asyncio.run(stream.run())
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_PAPER_WS_URL = "wss://stream.data.alpaca.markets/v2/iex"
_LIVE_WS_URL = "wss://stream.data.alpaca.markets/v2/sip"


@dataclass
class BarEvent:
    """A single completed 1-minute OHLCV bar from Alpaca.

    Attributes
    ----------
    symbol:
        Ticker symbol.
    timestamp:
        ISO-8601 bar close time string.
    open, high, low, close:
        OHLC prices.
    volume:
        Bar volume.
    """

    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    @classmethod
    def from_dict(cls, data: dict) -> BarEvent:
        """Parse an Alpaca streaming bar message.

        Alpaca sends abbreviated keys: S, t, o, h, l, c, v.
        """
        return cls(
            symbol=data["S"],
            timestamp=data["t"],
            open=float(data["o"]),
            high=float(data["h"]),
            low=float(data["l"]),
            close=float(data["c"]),
            volume=int(data["v"]),
        )


class AlpacaBarStream:
    """Alpaca WebSocket bar stream client.

    Parameters
    ----------
    api_key:
        Alpaca API key ID.
    secret_key:
        Alpaca secret key.
    symbols:
        List of ticker symbols to subscribe to.
    paper:
        Use IEX paper data feed (default). Set ``False`` for SIP live feed.
    on_bar:
        Optional callback invoked with each :class:`BarEvent`. Can also be
        set as an attribute after construction.
    """

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        symbols: list[str],
        paper: bool = True,
        on_bar: Callable[[BarEvent], None] | None = None,
    ) -> None:
        self._api_key = api_key
        self._secret_key = secret_key
        self.symbols = list(symbols)
        self.ws_url = _PAPER_WS_URL if paper else _LIVE_WS_URL
        self.on_bar: Callable[[BarEvent], None] | None = on_bar
        self._running = False

    def _auth_message(self) -> dict:
        """Build the authentication payload."""
        return {"action": "auth", "key": self._api_key, "secret": self._secret_key}

    def _subscribe_message(self) -> dict:
        """Build the subscription payload for configured symbols."""
        return {"action": "subscribe", "bars": self.symbols}

    async def run(self) -> None:
        """Connect, authenticate, subscribe and stream bars.

        Requires ``websockets`` package (``pip install websockets``).
        Blocks until the connection is closed or an error occurs.
        """
        try:
            import websockets  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "Install websockets: pip install 'websockets>=12'"
            ) from exc

        self._running = True
        async with websockets.connect(self.ws_url) as ws:
            # Authenticate
            await ws.send(json.dumps(self._auth_message()))
            await ws.recv()  # auth response

            # Subscribe
            await ws.send(json.dumps(self._subscribe_message()))
            await ws.recv()  # subscription confirmation

            while self._running:
                raw = await ws.recv()
                messages = json.loads(raw)
                if not isinstance(messages, list):
                    messages = [messages]
                for msg in messages:
                    if msg.get("T") == "b":  # bar message type
                        evt = BarEvent.from_dict(msg)
                        if self.on_bar is not None:
                            self.on_bar(evt)

    def stop(self) -> None:
        """Signal the stream loop to stop after the next message."""
        self._running = False
