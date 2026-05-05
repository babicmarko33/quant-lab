# Data Handling — Standard Operating Procedure

## Purpose
Ensure all market data used in research and production is clean, consistent, properly cached, and free from common biases.

---

## Data Flow Architecture

```
External APIs → Raw Fetch → Validation → Standardization → Cache (Parquet) → Consumer
                                                                    ↑
                                                              Cache Hit → Skip fetch
```

---

## Provider Fallback Chain

| Priority | Provider | Auth Required | Rate Limit | Coverage |
|----------|----------|---------------|------------|----------|
| 1 | yfinance | No | Unofficial | US equities, ETFs |
| 2 | Alpaca | API Key | 200/min | US equities |
| 3 | Alpha Vantage | API Key | 5/min (free) | Global |
| 4 | Polygon | API Key | Unlimited (paid) | US equities, options |

---

## Standardized Schema

All data MUST conform to this schema before caching:

```python
# Columns (lowercase):
["open", "high", "low", "close", "volume", "adj_close"]

# Index:
DatetimeIndex, tz-naive (UTC converted to naive), name="date"

# Sorting:
Ascending by date (oldest first)

# Types:
open/high/low/close/adj_close: float64
volume: int64 or float64
```

---

## Validation Rules (Applied Before Caching)

1. **No NaN prices** — Drop rows where OHLC has NaN (log warning)
2. **High ≥ Low** — Always true; if violated, data is corrupted → reject
3. **Volume ≥ 0** — No negative volume
4. **Monotonic index** — Dates must be strictly increasing
5. **No future data** — Last date must be ≤ today
6. **Reasonable price range** — No prices < $0.01 or > $1,000,000

---

## Caching Strategy

- **Format**: Apache Parquet (columnar, compressed, fast random access)
- **Location**: `data/cache/{TICKER}_{interval}_{hash}.parquet`
- **Hash**: MD5 of `{ticker}_{start}_{end}_{interval}` (deterministic)
- **Invalidation**: Manual delete or `--no-cache` flag
- **Max age**: Not enforced (user explicitly passes dates)

---

## Survivorship Bias Prevention

When backtesting over historical periods:
1. Use historical constituent lists (e.g., S&P 500 as of 2010, not today's list)
2. Include delisted stocks in universe
3. Document any survivorship bias assumptions in research notebooks

---

## Corporate Actions

- **Splits**: Use adjusted prices (`adj_close`) for returns computation
- **Dividends**: Included in `adj_close` (total return)
- **For raw OHLC signals**: Use unadjusted prices but adjusted volume

---

## Anti-Patterns

- ❌ Using `close` instead of `adj_close` for multi-year returns
- ❌ Forward-filling missing data without logging
- ❌ Fetching same ticker repeatedly without caching
- ❌ Mixing timezone-aware and timezone-naive timestamps
- ❌ Storing CSVs instead of Parquet for large datasets
