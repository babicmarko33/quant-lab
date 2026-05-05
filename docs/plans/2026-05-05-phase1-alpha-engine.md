# quant-lab Phase 1 Implementation Plan — AlphaEngine Backtester + Strategies

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans skill to implement this plan task-by-task.

**Goal:** Build a production-grade, vectorized event-driven backtesting engine with walk-forward validation, then implement three battle-tested strategies — 12-1 Momentum, Mean-Reversion (Bollinger), and Pairs Stat-Arb — each with full performance analytics (Sharpe, Calmar, PSR, max-DD, turnover) and TDD coverage.

**Architecture:**
- `alpha_engine/backtest/engine.py` — vectorized backtest engine (no Python loops over bars; signal-on-close, fill-on-open)
- `alpha_engine/strategies/` — pluggable Strategy ABC; each strategy produces a signal Series in [-1, 0, +1]
- `alpha_engine/analytics/performance.py` — all performance metrics in one module
- `alpha_engine/risk/position_sizing.py` — Kelly criterion + max-DD stop

**Tech Stack:** numpy (vectorized), pandas (Series/DataFrame), scipy (statistical tests), quantstats (optional overlay), pytest (TDD cycle)

**Constraints:**
- NO look-ahead bias: signals computed on day T, fills executed on day T+1 open
- ALL strategies must pass walk-forward validation (purged k-fold)
- ALL commits must have green tests and clean ruff
- Conventional commits: `feat(alpha-engine):`, `test(backtest):`, `perf(backtest):`

---

## Task 1: BacktestResult dataclass

**Files:**
- Create: `src/alpha_engine/backtest/__init__.py`
- Create: `src/alpha_engine/backtest/types.py`
- Create: `tests/test_alpha_engine/__init__.py`
- Create: `tests/test_alpha_engine/test_types.py`

**Step 1: Write failing test**
```python
from alpha_engine.backtest.types import BacktestResult
def test_backtest_result_fields():
    r = BacktestResult(returns=pd.Series([0.01, -0.005, 0.02]))
    assert r.returns is not None
    assert r.total_return > 0
```

**Step 2: Run → verify RED**
Run: `pytest tests/test_alpha_engine/test_types.py -v`

**Step 3: Implement** — BacktestResult dataclass with fields:
- `returns: pd.Series`
- `equity_curve: pd.Series` (computed property)
- `total_return: float`
- `sharpe: float`
- `max_drawdown: float`
- `calmar: float`
- `n_trades: int`

**Step 4: Run → verify GREEN**
**Step 5: Commit** `feat(alpha-engine): add BacktestResult dataclass`

---

## Task 2: Performance analytics module

**Files:**
- Create: `src/alpha_engine/analytics/__init__.py`
- Create: `src/alpha_engine/analytics/performance.py`
- Create: `tests/test_alpha_engine/test_performance.py`

**Metrics to implement (all vectorized, no loops):**
| Function | Formula | Notes |
|----------|---------|-------|
| `sharpe_ratio(returns, rf, freq)` | (mean-rf)/std * √freq | annualized |
| `sortino_ratio(returns, rf, freq)` | (mean-rf)/downside_std * √freq | downside-only |
| `calmar_ratio(returns, freq)` | ann_return / abs(max_dd) | |
| `max_drawdown(returns)` | max(1 - equity/peak) | peak-to-trough |
| `probabilistic_sharpe(returns, sr_star, freq)` | PSR as per López de Prado 2018 | |
| `information_coefficient(signals, forward_returns)` | Spearman rank IC | |
| `annual_turnover(positions)` | % portfolio turned per year | |

**Step 1: Write tests for each function with known analytical results**
**Step 2: Run → verify RED for each**
**Step 3: Implement each function**
**Step 4: Run → verify ALL GREEN**
**Step 5: Commit** `feat(alpha-engine): vectorized performance analytics`

---

## Task 3: Core backtesting engine

**Files:**
- Create: `src/alpha_engine/backtest/engine.py`
- Create: `tests/test_alpha_engine/test_engine.py`

**Design:**
```python
def run_backtest(
    signals: pd.Series,       # -1/0/+1 on close of day T
    prices: pd.DataFrame,     # OHLCV — fills use Open of T+1
    initial_capital: float,
    commission_bps: int,
    slippage_bps: int,
) -> BacktestResult: ...
```

**Key invariants to test:**
1. Signal on T → fill on T+1 open (no look-ahead)
2. Transaction costs reduce equity correctly
3. Long-only constraint works
4. Zero signals → no trades, equity stays flat (no costs)
5. Constant up-trend with long signal → positive returns

**Step 1: Write all 5 tests**
**Step 2: Run → verify RED**
**Step 3: Implement vectorized engine**
**Step 4: Run → verify GREEN**
**Step 5: Commit** `feat(alpha-engine): vectorized backtest engine`

---

## Task 4: Strategy ABC + registry

**Files:**
- Create: `src/alpha_engine/strategies/__init__.py`
- Create: `src/alpha_engine/strategies/base.py`

**Design:**
```python
class Strategy(ABC):
    name: str
    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return Series of float in [-1, 0, +1] indexed by date."""
        ...
    def run(self, df, config) -> BacktestResult:
        signals = self.generate_signals(df)
        return run_backtest(signals, df, **config)
```

**Step 1: Write test verifying ABC contract**
**Step 2: Implement**
**Step 3: Commit** `feat(alpha-engine): Strategy ABC`

---

## Task 5: Momentum strategy (12-1 cross-sectional)

**Files:**
- Create: `src/alpha_engine/strategies/momentum.py`
- Create: `tests/test_alpha_engine/test_momentum.py`

**Algorithm:**
- Signal = sign of 12-month return, skipping most recent 1 month (avoid reversal)
- Long when trailing 12-1 month return > 0
- Short (or flat) when < 0

**Tests:**
1. Signals are in {-1, 0, +1}
2. No NaN in signal series after warmup (>252 bars)
3. Bullish regime (250 up days) → positive signal ratio
4. Smoke test: full backtest on SPY data returns valid BacktestResult

**Step 1-5: TDD cycle + commit** `feat(alpha-engine): momentum 12-1 strategy`

---

## Task 6: Mean-Reversion strategy (Bollinger Band squeeze)

**Files:**
- Create: `src/alpha_engine/strategies/mean_reversion.py`
- Create: `tests/test_alpha_engine/test_mean_reversion.py`

**Algorithm:**
- Long when close < lower BB (oversold, expect reversion)
- Short when close > upper BB (overbought, expect reversion)
- Exit when close crosses middle band

**Tests:**
1. Signal range [-1, 0, +1]
2. Persistent above-upper-BB data → all -1 signals
3. Persistent below-lower-BB data → all +1 signals

**Step 1-5: TDD cycle + commit** `feat(alpha-engine): mean-reversion BB strategy`

---

## Task 7: Walk-forward validation harness

**Files:**
- Create: `src/alpha_engine/backtest/walk_forward.py`
- Create: `tests/test_alpha_engine/test_walk_forward.py`

**Design:**
```python
def walk_forward(
    strategy: Strategy,
    df: pd.DataFrame,
    n_splits: int = 5,
    train_pct: float = 0.6,
    embargo_days: int = 10,
) -> list[BacktestResult]: ...
```

**Tests:**
1. Returns `n_splits` results
2. No overlap between train/test windows
3. Embargo gap is respected
4. OOS equity curves are not look-ahead contaminated

**Step 1-5: TDD cycle + commit** `feat(alpha-engine): walk-forward validation harness`

---

## Task 8: Risk module (position sizing)

**Files:**
- Create: `src/alpha_engine/risk/__init__.py`
- Create: `src/alpha_engine/risk/position_sizing.py`
- Create: `tests/test_alpha_engine/test_risk.py`

**Functions:**
- `kelly_fraction(win_rate, win_loss_ratio)` — full Kelly
- `fractional_kelly(returns, fraction=0.25)` — practical Kelly
- `vol_target_size(target_vol, asset_vol, capital)` — volatility targeting

**Step 1-5: TDD cycle + commit** `feat(alpha-engine): Kelly + vol-target position sizing`

---

## Task 9: CLI smoke-test script

**Files:**
- Create: `scripts/run_backtest.py`

**Purpose:** End-to-end test — fetch SPY, run momentum + mean-reversion, print a summary table. Human-verifiable smoke test.

**Step 1: Implement**
**Step 2: Run: `python scripts/run_backtest.py`**
**Step 3: Verify output shows sensible numbers**
**Step 4: Commit** `feat(scripts): backtest smoke test script`

---

## Task 10: Create develop branch + PR

**Steps:**
1. `git checkout -b develop`
2. `git push -u origin develop`
3. PR: main ← develop (for future feature branches to target develop)
4. Update README with Phase 1 badge + "What's implemented" section

**Commit:** `docs: update README for Phase 1 milestone`

---

## Verification Checklist (required before claiming Phase 1 done)

```bash
# All tests pass (no integration markers)
pytest -m "not integration" -q

# Lint clean
ruff check src/ tests/

# Smoke test runs
python scripts/run_backtest.py

# Coverage report
pytest --cov=src --cov-report=term-missing -m "not integration" -q
```

All four must produce green output before any "Phase 1 complete" claim.

---

## What Comes After Phase 1

- **Phase 2:** AlphaML — XGBoost + LSTM price prediction with purged k-fold, IC analysis
- **Phase 3:** Portfolio-level risk + multi-strategy allocation (Kelly + correlation)
- **Phase 4:** Streamlit dashboard + live paper trading (Alpaca)
- **Phase 5:** Derivatives Lab — Black-Scholes PDE, Monte Carlo pricer, Vol surface
