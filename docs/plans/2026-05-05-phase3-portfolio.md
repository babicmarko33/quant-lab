# Portfolio Allocation Phase 3 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a portfolio allocation layer that takes multiple strategy signals or asset returns and computes optimal weights — mean-variance (Markowitz), risk parity, and CVaR minimization.

**Architecture:** All allocators implement `Allocator` ABC with `fit(returns) -> weights`. Weights are pd.Series indexed by asset name summing to 1.0. Allocation feeds into a multi-asset backtest that replaces single-asset `run_backtest()`.

**Tech Stack:** numpy, scipy.optimize, pandas (existing), cvxpy for CVaR (optional fallback to scipy)

---

## Prerequisites

```powershell
.\.venv\Scripts\activate; pip install cvxpy
```

---

### Task 1: Allocator ABC + Equal-Weight Baseline

**Files:**
- Create: `src/alpha_engine/portfolio/allocator.py`
- Create: `src/alpha_engine/portfolio/__init__.py`
- Test: `tests/test_alpha_engine/test_portfolio.py`

**Allocator ABC:**
```python
class Allocator(ABC):
    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> pd.Series:
        """Returns: weights indexed by asset name, sums to 1.0"""
```

**EqualWeightAllocator** — trivial baseline, always `1/n` per asset.

---

### Task 2: Mean-Variance (Markowitz) Allocator

**Theory:** Maximize Sharpe = (w·mu - rf) / sqrt(w·Sigma·w)
Equivalent to minimizing portfolio variance for a given return target.

Uses `scipy.optimize.minimize` with constraints:
- sum(w) = 1.0
- w_i >= 0 (long-only)

---

### Task 3: Risk Parity Allocator

**Theory:** Each asset contributes equal risk. Marginal Risk Contribution = w_i * (Sigma·w)_i / sqrt(w·Sigma·w).

Risk parity: MRC_i = target_risk / n for all i.

Uses iterative optimization via `scipy.optimize.minimize`.

---

### Task 4: CVaR Minimization Allocator

**Theory:** Minimize Conditional Value-at-Risk (Expected Shortfall) at confidence level alpha.
CVaR = E[loss | loss > VaR_alpha]

Uses cvxpy for linear programming formulation (Rockafellar & Uryasev 2000).

---

### Task 5: Multi-Asset Backtest

**Files:**
- Create: `src/alpha_engine/backtest/multi_asset.py`
- Test: `tests/test_alpha_engine/test_multi_asset.py`

Takes DataFrame of per-asset signals + OHLCV dict, applies allocator to compute weights, runs combined backtest.

---

### Task 6: Full suite + push
