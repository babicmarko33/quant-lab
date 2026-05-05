#!/usr/bin/env python
"""Paper trading CLI -- fetch latest OHLCV and submit orders via Alpaca paper API.

Usage:
    python scripts/run_paper_trader.py --ticker SPY --strategy momentum

Required .env variables (optional -- signal-only mode if absent):
    ALPACA_API_KEY=...
    ALPACA_SECRET_KEY=...
    ALPACA_BASE_URL=https://paper-api.alpaca.markets
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

sys.path.insert(0, "src")

try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]

    load_dotenv()
except ImportError:
    pass  # python-dotenv optional

from alpha_engine.execution.alpaca_broker import AlpacaBroker
from alpha_engine.execution.paper_trader import PaperTrader
from alpha_engine.strategies import REGISTRY
from quantcore.data.fetcher import fetch_ohlcv

SIGNAL_LABELS = {1: "LONG", -1: "SHORT", 0: "FLAT"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper trading loop for one signal cycle")
    parser.add_argument("--ticker", default="SPY", help="Ticker symbol to trade")
    parser.add_argument("--strategy", default="momentum", choices=list(REGISTRY.keys()))
    parser.add_argument("--qty", type=float, default=1.0, help="Units per order")
    parser.add_argument("--period", default="1y", help="Historical data period (e.g. 1y, 2y)")
    args = parser.parse_args()

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    live_mode = bool(api_key and secret_key)
    if not live_mode:
        print("[warn] ALPACA_API_KEY or ALPACA_SECRET_KEY not set -- running in signal-only mode")

    print(f"[info] Fetching {args.ticker} ({args.period})...")
    years = int(args.period.rstrip("y"))
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    df = fetch_ohlcv(args.ticker, start=start)
    if df.empty:
        print(f"[error] No data returned for {args.ticker}")
        sys.exit(1)

    strategy = REGISTRY[args.strategy]()
    signals = strategy.generate_signals(df)
    last_signal = int(signals.iloc[-1])
    print(f"[signal] {args.ticker} current signal: {SIGNAL_LABELS.get(last_signal, 'UNKNOWN')} ({last_signal})")

    if live_mode:
        broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, base_url=base_url)
        trader = PaperTrader(symbol=args.ticker, qty_per_trade=args.qty, broker=broker)
        orders = trader.run(df, strategy)
        for o in orders:
            print(f"[order] {o.side.upper()} {o.qty} {o.symbol} -> status={o.status} id={o.order_id}")
        if not orders:
            print("[info] No new orders (signal unchanged or FLAT)")
    else:
        print("[info] Skipping order submission (no Alpaca credentials)")


if __name__ == "__main__":
    main()
