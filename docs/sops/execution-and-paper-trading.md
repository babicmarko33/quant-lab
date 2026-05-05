# Execution & Paper Trading — Standard Operating Procedure

## Purpose
Define safe, auditable procedures for generating trade signals and submitting paper/live orders via the Alpaca broker adapter.

---

## Architecture Overview

```
Strategy.generate_signals(df)
         │
         ▼
    PaperTrader.run(df, strategy)
         │  (signal changed?)
         ▼
    Broker.submit(order)
         │
         ▼
    AlpacaBroker → Alpaca REST API (paper-api.alpaca.markets)
```

**Key invariant:** Orders are only submitted when the signal *changes*. `PaperTrader` maintains `_last_signal` state — no duplicate orders on repeated runs.

---

## Environment Setup

Required `.env` variables:

```env
ALPACA_API_KEY=your-paper-key-id
ALPACA_SECRET_KEY=your-paper-secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**NEVER** use live URL (`https://api.alpaca.markets`) without:
1. Full risk review of the strategy
2. Position sizing limits configured
3. Explicit stakeholder sign-off

---

## Running the Paper Trader

```bash
# Signal-only mode (no Alpaca keys needed)
python scripts/run_paper_trader.py --ticker SPY --strategy momentum

# Live paper submission (keys in .env)
python scripts/run_paper_trader.py --ticker SPY --strategy bollinger_mr --qty 10
```

Output signals: `LONG (+1)`, `SHORT (-1)`, `FLAT (0)`.

---

## Order Lifecycle

| Status | Meaning |
|--------|---------|
| `pending` | Created locally, not yet submitted |
| `filled` | Alpaca confirmed execution |
| `cancelled` | Alpaca rejected or cancelled |

Orders are **immutable** after creation. Broker returns a new `Order` copy via `dataclasses.replace()`.

---

## Safety Rules

1. **Never submit live orders** without running paper trading for minimum 2 weeks
2. **Always validate signals offline first** using `scripts/run_paper_trader.py` (no keys)
3. **Maximum position size**: Configure `--qty` conservatively; default = 1 unit
4. **Check Alpaca account cash** before changing to fractional sizes
5. **Reconnect handling**: If broker throws, log and skip — do not retry silently
6. **Signal frequency**: Run at market open only (9:35 AM EST). Never intra-session for daily strategies.

---

## Testing Execution Code

All execution tests MUST use `unittest.mock.patch` to mock Alpaca REST:

```python
from unittest.mock import MagicMock, patch

with patch("alpha_engine.execution.alpaca_broker.tradeapi.REST", return_value=mock_api):
    broker = AlpacaBroker(api_key="k", secret_key="s", base_url="...")
```

**Never** use real Alpaca credentials in tests. No live API calls in CI.

---

## Adding a New Broker Adapter

1. Subclass `alpha_engine.execution.broker.Broker`
2. Implement `submit(order: Order) -> Order` — MUST return updated copy, MUST NOT mutate input
3. Write tests with mock for all external calls
4. Register in `src/alpha_engine/execution/__init__.py`
5. Document required env vars in `.env.example`
