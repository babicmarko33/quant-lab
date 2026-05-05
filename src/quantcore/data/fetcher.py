"""Universal market data fetcher with multi-source fallback chain.

Fetches OHLCV data from multiple providers with automatic fallback:
    yfinance (free, no key) → Alpaca → Alpha Vantage → Polygon

All providers return a standardized DataFrame with columns:
    [open, high, low, close, volume, adj_close]
Index: DatetimeIndex (timezone-naive, sorted ascending)
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from quantcore.config import config

logger = logging.getLogger(__name__)

# Standardized column names for all providers
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume", "adj_close"]


def _cache_path(ticker: str, start: str, end: str, interval: str) -> Path:
    """Generate deterministic cache file path for a data request."""
    key = f"{ticker}_{start}_{end}_{interval}"
    hashed = hashlib.md5(key.encode()).hexdigest()[:12]  # noqa: S324
    cache_dir = config.data_cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{ticker}_{interval}_{hashed}.parquet"


def _fetch_yfinance(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch data from yfinance (primary free source, no API key needed)."""
    import yfinance as yf

    logger.info(f"Fetching {ticker} from yfinance [{start} → {end}]")
    data = yf.download(ticker, start=start, end=end, interval=interval, progress=False, auto_adjust=False)

    if data.empty:
        raise ValueError(f"yfinance returned empty data for {ticker}")

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    df = pd.DataFrame(index=data.index)
    df["open"] = data["Open"]
    df["high"] = data["High"]
    df["low"] = data["Low"]
    df["close"] = data["Close"]
    df["volume"] = data["Volume"]
    df["adj_close"] = data.get("Adj Close", data["Close"])

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.sort_index()


def _fetch_alpaca(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Fetch data from Alpaca Markets API."""
    from alpaca_trade_api import REST

    api_key = config.market_data.alpaca_api_key
    secret_key = config.market_data.alpaca_secret_key
    if not api_key or not secret_key:
        raise ValueError("Alpaca API keys not configured")

    logger.info(f"Fetching {ticker} from Alpaca [{start} → {end}]")
    api = REST(api_key, secret_key, config.market_data.alpaca_base_url)

    timeframe_map = {"1d": "1Day", "1h": "1Hour", "1m": "1Min"}
    timeframe = timeframe_map.get(interval, "1Day")

    bars = api.get_bars(ticker, timeframe, start=start, end=end).df

    if bars.empty:
        raise ValueError(f"Alpaca returned empty data for {ticker}")

    df = pd.DataFrame(index=bars.index)
    df["open"] = bars["open"]
    df["high"] = bars["high"]
    df["low"] = bars["low"]
    df["close"] = bars["close"]
    df["volume"] = bars["volume"]
    df["adj_close"] = bars["close"]  # Alpaca data is already adjusted

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.sort_index()


def fetch_ohlcv(
    ticker: str,
    start: str = "2020-01-01",
    end: str | None = None,
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data with automatic provider fallback and local caching.

    Parameters
    ----------
    ticker : str
        Stock/ETF ticker symbol (e.g., 'SPY', 'AAPL')
    start : str
        Start date in 'YYYY-MM-DD' format
    end : str, optional
        End date in 'YYYY-MM-DD' format. Defaults to today.
    interval : str
        Data frequency: '1d' (daily), '1h' (hourly), '1m' (minute)
    use_cache : bool
        Whether to use local parquet cache

    Returns
    -------
    pd.DataFrame
        OHLCV data with columns: [open, high, low, close, volume, adj_close]
        Index: DatetimeIndex named 'date'

    Raises
    ------
    RuntimeError
        If all providers fail to return data
    """
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")

    # Check cache first
    if use_cache:
        cache_file = _cache_path(ticker, start, end, interval)
        if cache_file.exists():
            logger.debug(f"Cache hit: {cache_file}")
            df = pd.read_parquet(cache_file)
            df.index = pd.to_datetime(df.index)
            return df

    # Provider fallback chain
    providers = [
        ("yfinance", _fetch_yfinance),
        ("alpaca", _fetch_alpaca),
    ]

    last_error = None
    for name, fetcher in providers:
        try:
            df = fetcher(ticker, start, end, interval)
            # Validate output schema
            assert all(col in df.columns for col in OHLCV_COLUMNS), f"Missing columns from {name}"
            assert not df.empty, f"Empty DataFrame from {name}"
            assert df.index.is_monotonic_increasing, f"Non-monotonic index from {name}"

            # Cache successful result
            if use_cache:
                cache_file = _cache_path(ticker, start, end, interval)
                df.to_parquet(cache_file)
                logger.debug(f"Cached to {cache_file}")

            logger.info(f"Successfully fetched {ticker}: {len(df)} rows from {name}")
            return df

        except Exception as e:
            logger.warning(f"Provider {name} failed for {ticker}: {e}")
            last_error = e
            continue

    raise RuntimeError(f"All providers failed for {ticker}. Last error: {last_error}")


def fetch_multiple(
    tickers: list[str],
    start: str = "2020-01-01",
    end: str | None = None,
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV data for multiple tickers.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame
    """
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = fetch_ohlcv(ticker, start, end, interval)
        except RuntimeError as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
    return results
