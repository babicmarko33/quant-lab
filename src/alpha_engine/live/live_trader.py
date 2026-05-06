"""Live trading engine that wires an AlpacaBarStream to a Strategy.

Buffers incoming :class:`BarEvent` objects into a rolling OHLCV
DataFrame and generates position signals each time a new bar arrives
(once ``min_bars`` are available).

Usage::

    trader = LiveTrader(
        strategy=MovingAverageCrossStrategy(),
        stream=stream,
        symbol="SPY",
        min_bars=50,
    )
    trader.on_signal = lambda sig: place_order(sig)
    stream.on_bar = trader.on_bar
    asyncio.run(stream.run())
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable

import pandas as pd

from alpha_engine.live.alpaca_stream import AlpacaBarStream, BarEvent
from alpha_engine.strategies.base import Strategy


class LiveTrader:
    """Routes live bars from :class:`AlpacaBarStream` through a Strategy.

    Parameters
    ----------
    strategy:
        Any :class:`Strategy` subclass.
    stream:
        Configured :class:`AlpacaBarStream`.
    symbol:
        Symbol this trader tracks.
    min_bars:
        Minimum bars to buffer before generating signals.
    on_signal:
        Optional callback invoked with each integer signal ``{-1, 0, 1}``.
        Can also be set as an attribute after construction.
    max_buffer:
        Maximum bar buffer size (rolling window). Default 500.
    """

    def __init__(
        self,
        strategy: Strategy,
        stream: AlpacaBarStream,
        symbol: str,
        min_bars: int = 50,
        on_signal: Callable[[int], None] | None = None,
        max_buffer: int = 500,
    ) -> None:
        self.strategy = strategy
        self.stream = stream
        self.symbol = symbol
        self.min_bars = min_bars
        self.on_signal: Callable[[int], None] | None = on_signal
        self._bar_buffer: deque[dict] = deque(maxlen=max_buffer)

    def on_bar(self, evt: BarEvent) -> None:
        """Process a new :class:`BarEvent`.

        Appends the bar to the internal buffer. If the buffer has at least
        ``min_bars`` entries, runs the strategy and emits a signal.

        Parameters
        ----------
        evt:
            The incoming bar event.
        """
        if evt.symbol != self.symbol:
            return

        self._bar_buffer.append({
            "open": evt.open,
            "high": evt.high,
            "low": evt.low,
            "close": evt.close,
            "volume": float(evt.volume),
        })

        if len(self._bar_buffer) < self.min_bars:
            return

        df = pd.DataFrame(list(self._bar_buffer))
        signals = self.strategy.generate_signals(df)
        latest_signal = int(signals.iloc[-1])

        if self.on_signal is not None:
            self.on_signal(latest_signal)
