---
name: quant-data-pipeline
description: "Use when adding a new data source, modifying the OHLCV fetcher, working with the parquet cache, or building a new feature engineering pipeline. Covers the fetcher fallback chain, caching conventions, schema validation, and look-ahead bias prevention in features."
---

# quant-data-pipeline: Market Data and Feature Engineering

## Fetcher Architecture

```
fetch_ohlcv(ticker, start, end, interval)
    │
    ├─ Check parquet cache (_cache_path)
    │     └─ If fresh → return cached DataFrame
    │
    ├─ Try yfinance
    │     └─ Success → validate schema → cache → return
    │
    ├─ Try Alpaca REST (requires ALPACA_API_KEY + ALPACA_SECRET_KEY)
    │     └─ Success → validate schema → cache → return
    │
    ├─ Try Alpha Vantage (stub — not yet implemented)
    └─ Try Polygon (stub — not yet implemented)
```

Source: `src/quantcore/data/fetcher.py`

---

## OHLCV Schema

All DataFrames returned by `fetch_ohlcv` must conform to:

```python
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]

# Index: DatetimeIndex, UTC-normalized, business days only
# All prices: float64
# Volume: float64 (not int — avoids overflow)
```

**Never** add extra columns to the returned DataFrame — callers depend on this schema.

---

## Cache Convention

```python
# Cache path pattern
data/cache/{ticker}_{start}_{end}_{interval}.parquet

# Cache TTL
- Daily OHLCV: 24 hours (re-fetches if stale)
- Fama-French factors: 7 days
- Options chains: no cache (always live)
```

Tests must **never** hit the network. Patch the fetcher:
```python
@pytest.fixture
def mock_fetch(monkeypatch):
    def _fake_fetch(ticker, start=None, end=None, interval="1d"):
        n = 252
        idx = pd.bdate_range("2020-01-01", periods=n)
        return pd.DataFrame({
            "open": 100.0, "high": 102.0, "low": 98.0,
            "close": 101.0, "volume": 1e6,
        }, index=idx)
    monkeypatch.setattr("quantcore.data.fetcher.fetch_ohlcv", _fake_fetch)
    return _fake_fetch
```

---

## Feature Engineering Rules

### Look-Ahead Bias Prevention

```python
# WRONG
df["sma20"] = df["close"].rolling(20).mean()     # Today's close in today's signal

# CORRECT
df["sma20"] = df["close"].rolling(20).mean().shift(1)  # Yesterday's SMA

# CORRECT for forward returns (target variable only)
df["fwd_ret"] = df["close"].pct_change().shift(-1)  # Tomorrow's return as target
```

**Rule:** Any feature used as a predictor must be `.shift(1)` or computed entirely from data at T-1 or earlier.

### FeatureStore

```python
from alpha_ml.features.feature_store import FeatureStore

store = FeatureStore(horizon=5, winsorize_clip=3.0)
X, y = store.fit_transform(df)
# X: winsorized + z-scored features, no lookahead
# y: forward return sign (binary)
```

---

## Adding a New Data Source

1. Add API key to `src/quantcore/config.py` (`MarketDataConfig`)
2. Add `.env.example` entry
3. Implement `_fetch_from_<source>(ticker, start, end) -> pd.DataFrame`
4. Validate with `_validate_ohlcv_schema(df)` before caching
5. Add to fallback chain in `fetch_ohlcv` after existing sources
6. Write tests with `monkeypatch` to mock `requests.get` or library calls — **never hit network in tests**

---

## Parquet Cache Management

```python
# Read
df = pd.read_parquet(cache_path)

# Write (preserve index)
df.to_parquet(cache_path, index=True)

# Check freshness
import os
from datetime import datetime, timedelta
mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
is_fresh = (datetime.now() - mtime) < timedelta(hours=24)
```

The `data/` directory is gitignored. Never commit parquet files.

---

## Alpaca-Specific Notes

- Base URL: `https://data.alpaca.markets/v2` (data) vs `https://paper-api.alpaca.markets` (orders)
- Auth header: `APCA-API-KEY-ID` + `APCA-API-SECRET-KEY`
- Response bars use `t`, `o`, `h`, `l`, `c`, `v` — map to standard OHLCV columns before returning
- Rate limit: 200 req/min on free tier

---

## Fama-French Data (Phase G)

```python
# Kenneth French CSV URLs (with 7-day cache TTL)
FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"

# IMPORTANT: CSV values are in percent — divide by 100 before returning
factors_df = factors_df / 100
```
