# Project Workflow — Standard Operating Procedure

## Purpose
Define the development workflow for all quant-lab modules — from idea to deployed code.

---

## Development Cycle

```
Idea → Hypothesis → Research → Implement → Test → Validate → PR → Merge → Monitor
```

---

## Task Execution Protocol

### Before Starting ANY Task
1. Create feature branch: `git checkout -b feat/<scope>-<description>`
2. Write tests first (TDD) — see below
3. Implement minimal code to pass tests
4. Run full validation before PR

### TDD Cycle (Mandatory)
```
1. Write failing test
2. Run test → verify RED
3. Write minimal implementation
4. Run test → verify GREEN
5. Refactor if needed → stay GREEN
6. Commit: test(<scope>): add tests for <feature>
7. Commit: feat(<scope>): implement <feature>
```

---

## Module Development Order

```
Phase 0: Infrastructure (COMPLETE)
  └── pyproject.toml, CI/CD, configs, data fetcher, indicators, features, 5 SOPs, README

Phase 1: AlphaEngine (COMPLETE — 87 tests)
  └── BacktestResult, performance analytics, vectorized engine,
      Strategy ABC + Momentum + Bollinger, walk-forward, risk module, CLI

Phase 2: AlphaML (COMPLETE — 129 tests)
  └── PurgedKFold, FeatureStore, ModelTrainer ABC + XGBoost, CVResult, MLSignalPipeline

Phase 3: Portfolio Allocation (COMPLETE — 162 tests)
  └── Allocator ABC, EqualWeight, MeanVariance, RiskParity, CVaR,
      multi-asset backtester with rebalancing

Phase 4: Dashboard + Paper Trading (COMPLETE — 190 tests)
  └── Execution layer (Order, Broker, AlpacaBroker, PaperTrader),
      Streamlit multi-page dashboard, paper trading CLI

Phase 5: Derivatives Lab (IN PROGRESS)
  └── Black-Scholes pricing, Greeks, implied volatility, volatility surface

Phase 6: Production
  └── Live execution, monitoring, alerting
```

---

## Code Quality Gates

Every PR must pass:
- [ ] `ruff check src/ tests/` — Zero lint errors
- [ ] `ruff format --check src/ tests/` — Consistent formatting
- [ ] `pytest -x -m "not slow and not integration"` — All fast tests green
- [ ] `pytest --cov=src` — Coverage ≥ 80% for touched files
- [ ] Manual review of any strategy logic for bias

---

## Documentation Requirements

| Change Type | Required Docs |
|-------------|--------------|
| New module | Docstrings + README section |
| New strategy | Research notebook + backtest results |
| API change | Update affected docstrings |
| Bug fix | Add regression test |

---

## Environment Setup

```bash
# First time
git clone https://github.com/babicmarko33/quant-lab.git
cd quant-lab
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e ".[dev]"
cp .env.example .env
pre-commit install

# Daily
git pull origin develop
pip install -e ".[dev]"  # if deps changed
```

---

## Release Checklist

Before merging to `main`:
1. All tests pass on CI
2. Version bumped in `pyproject.toml`
3. CHANGELOG updated
4. README updated if public API changed
5. No `TODO` or `FIXME` in changed files
