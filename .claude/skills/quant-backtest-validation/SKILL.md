---
name: quant-backtest-validation
description: "Use when validating a backtest result, checking for bias in a strategy, or deciding if a backtest is trustworthy enough to move to paper trading. Covers look-ahead bias detection, overfitting checks, statistical significance tests, and the walk-forward protocol."
---

# quant-backtest-validation: Backtest Integrity Checklist

## The 7 Failure Modes of Backtests

1. **Look-ahead bias** — signal uses data not available at decision time
2. **Survivorship bias** — universe only includes currently-alive stocks
3. **Overfitting** — strategy tuned on the test set (IS = OOS)
4. **Transaction cost underestimation** — ignoring bid-ask, market impact, borrow cost
5. **Data snooping** — multiple strategies tried; best selected without correction
6. **Regime mismatch** — trained in bull market, tested only in bull market
7. **Look-back contamination** — rolling window statistics bleed future data

---

## Signal Construction Rules

```python
# WRONG — look-ahead bias
df["signal"] = df["close"].rolling(20).mean()  # uses today's close

# CORRECT — shift(1) before any signal use
df["sma20"] = df["close"].rolling(20).mean().shift(1)
df["signal"] = np.where(df["close"].shift(1) > df["sma20"], 1, 0)
```

**Rule:** Every feature/indicator used in `generate_signals()` must be computed on `df.shift(1)` or the rolling window must end at T-1 before the signal for T.

---

## Engine Fill Convention

quant-lab's `BacktestEngine` enforces:
- Signal at close T → fill at **open T+1**
- This is the only correct fill convention for EOD strategies

---

## Walk-Forward Protocol

```python
from alpha_engine.backtest.walk_forward import WalkForwardValidator

validator = WalkForwardValidator(
    train_size=252 * 3,   # 3 years IS
    test_size=63,          # 1 quarter OOS
    step_size=63,          # re-fit every quarter
    embargo=5,             # 5-day gap between IS and OOS
)
results = validator.run(df, strategy)
oos_sharpe = results.mean_oos_sharpe
```

**Minimum bar:** `oos_sharpe > 0.5` across all folds, `fraction_positive_folds >= 0.6`.

---

## Statistical Significance

Use Probabilistic Sharpe Ratio (PSR) from `alpha_engine.analytics.performance`:

```python
from alpha_engine.analytics.performance import probabilistic_sharpe_ratio

psr = probabilistic_sharpe_ratio(
    observed_sharpe=result.sharpe,
    benchmark_sharpe=0.0,
    n_returns=len(returns),
    skewness=returns.skew(),
    kurtosis=returns.kurtosis(),
)
assert psr > 0.95, f"PSR {psr:.3f} below 95% threshold — not significant"
```

---

## Overfitting Detection

### Deflated Sharpe Ratio
When N strategies are tested, adjust threshold:
```
DSR_threshold = SR_benchmark + sqrt(variance_SR) * Φ^{-1}(1 - p / N)
```
Rule: if testing >5 parameter combinations, apply Bonferroni correction.

### IS/OOS Decay
```
decay = (IS_sharpe - OOS_sharpe) / IS_sharpe
assert decay < 0.5, f"Sharpe decay {decay:.1%} > 50% — likely overfit"
```

---

## Transaction Cost Model

Default in `BacktestEngine`:
- **Commission:** 0.1% per trade (both sides)
- **Slippage:** 0.05% per trade (half-spread approximation)
- **Total round-trip:** 0.3%

For high-frequency signals (turnover > 1×/week), use `slippage=0.1` explicitly.

---

## Validation Checklist (run before claiming a strategy is "good")

```
[ ] generate_signals uses only shift(1) data
[ ] BacktestEngine fill is T+1 OPEN
[ ] Walk-forward: mean OOS Sharpe > 0.5
[ ] Walk-forward: ≥ 60% positive Sharpe folds
[ ] PSR > 0.95 (single strategy) or DSR threshold (multi-param)
[ ] IS/OOS Sharpe decay < 50%
[ ] Commission + slippage modeled
[ ] Tested across at least 2 distinct market regimes (bull + bear/sideways)
[ ] No parameters fit on test data
```

---

## Reporting Template

```
Strategy:     <name>
Period:       <start> – <end>
Universe:     <tickers>
Annual Ret:   <x.x%>
Sharpe:       <x.xx>
Max DD:       <x.x%>
Calmar:       <x.xx>
Turnover:     <x.x×/yr>
OOS Sharpe:   <x.xx> (walk-forward mean)
PSR:          <x.xxx>
Verdict:      PASS / FAIL — <reason>
```
