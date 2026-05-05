# Portfolio Allocation — Standard Operating Procedure

## Purpose
Define the standard workflow for constructing, validating, and deploying multi-asset portfolio allocations using the `alpha_engine.portfolio` module.

---

## Available Allocators

| Allocator | Class | Objective | Solver |
|-----------|-------|-----------|--------|
| Equal Weight | `EqualWeightAllocator` | 1/N benchmark | Closed form |
| Max Sharpe | `MeanVarianceAllocator(objective="max_sharpe")` | Maximize Sharpe | scipy SLSQP |
| Min Variance | `MeanVarianceAllocator(objective="min_variance")` | Minimize volatility | scipy SLSQP |
| Risk Parity | `RiskParityAllocator` | Equal risk contribution | scipy L-BFGS-B |
| CVaR | `CVaRAllocator(alpha=0.95)` | Minimize Expected Shortfall | cvxpy CLARABEL |

---

## Weight Contract (enforced by tests)

All allocators MUST return a `pd.Series` where:
- `weights.sum() == 1.0` (tolerance 1e-6) — fully invested
- `weights >= 0` — long-only (no shorts)
- `weights.index == returns.columns` — correct asset labelling

---

## Workflow

### 1. Prepare Return Data

```python
from quantcore.data.fetcher import fetch_ohlcv
import pandas as pd

tickers = ["SPY", "TLT", "GLD", "QQQ"]
closes = {t: fetch_ohlcv(t, start="2022-01-01")["close"] for t in tickers}
returns = pd.DataFrame(closes).pct_change().dropna()
```

### 2. Fit Weights

```python
from alpha_engine.portfolio import RiskParityAllocator

alloc = RiskParityAllocator()
weights = alloc.fit(returns)
print(weights)
# SPY    0.312
# TLT    0.284
# GLD    0.231
# QQQ    0.173
```

### 3. Backtest with Rebalancing

```python
from alpha_engine.backtest.multi_asset import run_portfolio_backtest
from quantcore.data.fetcher import fetch_ohlcv

prices = {t: fetch_ohlcv(t, start="2020-01-01") for t in tickers}
result = run_portfolio_backtest(prices, alloc, rebalance_freq="ME")
print(f"Sharpe: {result.sharpe:.2f}  MaxDD: {result.max_drawdown:.1%}")
```

---

## Choosing an Allocator

| Situation | Recommended Allocator |
|-----------|----------------------|
| Baseline comparison | `EqualWeightAllocator` (hardest to beat) |
| Views on expected returns | `MeanVarianceAllocator("max_sharpe")` |
| Volatility-sensitive mandate | `MeanVarianceAllocator("min_variance")` |
| No return views, risk parity | `RiskParityAllocator` |
| Tail-risk mandate (CVaR limit) | `CVaRAllocator(alpha=0.95)` |

**DeMiguel et al. (2009)**: 1/N outperforms MV in OOS tests on many datasets. Always include EqualWeight as a benchmark.

---

## Rebalancing Frequency Guidelines

| Asset Class | Typical Frequency | pandas Alias |
|------------|------------------|--------------|
| Equity ETFs | Monthly | `"ME"` |
| Factor portfolios | Quarterly | `"QE"` |
| Long-only bonds | Annual | `"YE"` |

Higher rebalancing = more turnover costs. Always model costs with `commission_bps` and `slippage_bps`.

---

## Anti-Patterns

1. **Training on full history then testing on part of it** — Always use expanding or rolling window
2. **Using daily returns with CVaR** without sufficient history (< 252 obs) — results unreliable
3. **Ignoring covariance conditioning** — If `corr > 0.95` for two assets, consider deduplication
4. **No transaction costs in backtest** — Use at minimum `commission_bps=10, slippage_bps=5`
5. **Concentrating > 50% in one asset** — Apply max weight constraint if needed (extend `MeanVarianceAllocator`)

---

## Extending the Framework

To add a new allocator:

1. Subclass `alpha_engine.portfolio.allocator.Allocator`
2. Implement `fit(returns: pd.DataFrame) -> pd.Series`
3. Register in `src/alpha_engine/portfolio/__init__.py`
4. Add to parametrized `TestAllocatorInvariants` fixture in `tests/test_alpha_engine/test_portfolio.py`
5. Write class-specific tests: numerical properties, edge cases (single asset, correlated assets)

---

## References

- Markowitz (1952). Portfolio Selection. *Journal of Finance*
- DeMiguel, Garlappi, Uppal (2009). Optimal Versus Naive Diversification. *Review of Financial Studies*
- Maillard, Roncalli, Teiletche (2010). Properties of Equally Weighted Risk Contributions Portfolios. *JPM*
- Rockafellar, Uryasev (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*
