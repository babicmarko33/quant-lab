---
name: quant-dev
description: "Use when implementing any new quantitative finance module in quant-lab — strategies, allocators, derivatives pricing, ML models, or dashboard pages. Covers TDD patterns, naming conventions, module structure, and integration requirements specific to this codebase."
---

# quant-dev: Quantitative Finance Module Development

## Module Location Map

| What you're building | Location |
|---------------------|----------|
| New strategy | `src/alpha_engine/strategies/<name>.py` |
| New allocator | `src/alpha_engine/portfolio/<name>.py` |
| New derivatives pricer | `src/alpha_engine/derivatives/options/<name>.py` |
| New vol model | `src/alpha_engine/derivatives/volatility/<name>.py` |
| New ML model | `src/alpha_ml/models/<name>.py` |
| New dashboard page | `src/quant_dashboard/pages/N_Name.py` |
| New chart helper | `src/quant_dashboard/components/charts.py` |
| Tests | `tests/<matching_mirror_path>/test_<name>.py` |

---

## Mandatory Base Classes

### Strategy
```python
from alpha_engine.strategies.base import Strategy, BacktestResult
class MyStrategy(Strategy):
    @property
    def name(self) -> str: return "my_strategy"
    def generate_signals(self, df: pd.DataFrame) -> pd.Series: ...
```

### Allocator
```python
from alpha_engine.portfolio.allocator import Allocator
class MyAllocator(Allocator):
    def fit(self, returns: pd.DataFrame) -> pd.Series: ...
```

### ModelTrainer
```python
from alpha_ml.models.base import ModelTrainer
class MyModel(ModelTrainer):
    def fit(self, X, y): ...
    def predict(self, X) -> np.ndarray: ...
    def predict_proba(self, X) -> np.ndarray: ...
    def feature_importance(self) -> dict: ...
```

---

## TDD Patterns for Each Module Type

### Derivatives pricer test template
```python
import pytest
def test_<name>_put_call_parity():
    call = <name>_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    put  = <name>_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")
    assert abs(call - put - (100 - 100 * math.exp(-0.05))) < 0.05

def test_<name>_matches_bsm_at_zero_variance():
    # Pricing methods must converge to BSM when parameters reduce to BSM case
    ...
```

### Portfolio allocator test template
```python
def test_<name>_weights_sum_to_one(sample_returns):
    w = MyAllocator().fit(sample_returns)
    assert abs(w.sum() - 1.0) < 1e-6

def test_<name>_all_weights_non_negative(sample_returns):
    w = MyAllocator().fit(sample_returns)
    assert (w >= -1e-9).all()
```

### Strategy test template
```python
def test_<name>_returns_series_aligned_to_df_index(ohlcv_df):
    signals = MyStrategy().generate_signals(ohlcv_df)
    assert signals.index.equals(ohlcv_df.index)
    assert signals.isin([-1, 0, 1]).all()
```

---

## ruff Configuration

`pyproject.toml` per-file-ignores (already set — do not change):
```toml
[tool.ruff.lint.per-file-ignores]
"src/alpha_ml/**" = ["N803", "N806"]
"src/alpha_engine/derivatives/**" = ["N803", "N806"]
"src/alpha_engine/portfolio/**" = ["N803", "N806"]
```

Use uppercase variable names freely for math: `Sigma`, `P`, `Q`, `Omega`, `K`, `T`.

---

## Known Library Constraints

- **scikit-learn 1.8**: Use `l1_ratio` / `C` params. `penalty` FutureWarning is harmless.
- **cvxpy 1.8.2**: Cardinality constraints require `cp.Variable(boolean=True)` + SCIP solver.
- **PyTorch**: Import guard `try: import torch` — skip tests if not available.
- **arch (GARCH)**: Input percent returns (×100). Variance output is ×10000 — divide back.
- **pytest_asyncio**: Never use `@pytest_asyncio.fixture async def` — crashes Python 3.12.

---

## Dashboard Page Checklist

- [ ] All charts use helpers from `components/charts.py`
- [ ] `st.sidebar` for all parameter inputs
- [ ] `try/except` around heavy computation → `st.error(str(e))`
- [ ] No `st.cache_data` on functions that take mutable inputs

---

## Commit Convention

```
feat(<scope>): implement <module>          # new feature
test(<scope>): add tests for <module>      # tests only
fix(<scope>): fix <bug>                    # bug fix
docs(<scope>): update <doc>               # docs only
refactor(<scope>): <description>          # no behavior change
```

Scopes: `quantcore`, `alpha-engine`, `alpha-ml`, `portfolio`, `derivatives`, `execution`, `dashboard`

---

## Pre-Commit Gate (every commit)

```powershell
ruff check src/ tests/          # must be zero errors
pytest -q --tb=short            # must be green
npx gitnexus analyze            # after every commit
```
