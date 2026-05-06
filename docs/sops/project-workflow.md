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
  └── pyproject.toml, CI/CD, configs, data fetcher, indicators, features, 8 SOPs, README

Phase 1: AlphaEngine (COMPLETE — 87 tests)
  └── BacktestResult, performance analytics, vectorized engine,
      Strategy ABC + Momentum + Bollinger + SMA/EMA cross + RSI mean-reversion,
      walk-forward, risk module, CLI

Phase 2: AlphaML (COMPLETE — 129 tests)
  └── PurgedKFold, FeatureStore, ModelTrainer ABC + XGBoost, CVResult, MLSignalPipeline

Phase 3: Portfolio Allocation (COMPLETE — 162 tests)
  └── Allocator ABC, EqualWeight, MeanVariance, RiskParity, CVaR,
      multi-asset backtester with rebalancing

Phase 3.3–3.8: Advanced Derivatives (COMPLETE — 68 tests added)
  └── Binomial CRR, MC (European/Asian/Barrier), Crank-Nicolson PDE + PSOR (American),
      Merton jump-diffusion PIDE, Variance Gamma MC, SABR calibration

Phase 4: Dashboard + Paper Trading (COMPLETE — 190 tests total baseline)
  └── Execution layer (Order, Broker, AlpacaBroker, PaperTrader),
      Streamlit multi-page dashboard (5 pages), paper trading CLI

Phase 4.4–4.7: Advanced Portfolio Allocators (COMPLETE — 38 tests added)
  └── CardinalityAllocator (MIQP/SCIP), MC VaR/CVaR (Cholesky),
      MertonHJB (stochastic control), BlackLitterman (investor views)

Phase 5: Derivatives Lab (COMPLETE — 51 tests)
  └── Black-Scholes pricing, full Greeks, IV solver (NR + Brent), multi-leg strategies,
      VolatilitySurface (cubic spline)

Phase 5.1–5.3: ML Models (COMPLETE — 33 tests added)
  └── Ridge/Lasso classifiers, LSTM (PyTorch sliding-window), ModelEnsemble (soft-voting)

Phase 6: Dashboard Completion (COMPLETE — 433 tests total)
  └── 4_Options_Pricing.py (model comparison, Greeks, SABR smile)
      5_Market_Data.py (OHLCV, returns, vol surface heatmap)
      12 chart helpers in charts.py

Phase 7: Advanced Research (NEXT)
  └── Phase B: GARCH volatility forecasting (arch library) — COMPLETE
  └── Phase C: Kalman filter + cointegration pairs trading — COMPLETE
  └── Phase D: HMM regime detection — COMPLETE
  └── Phase E: Tradier live options chain + real IV surface
  └── Phase F: Alpaca WebSocket live event loop + LiveTrader
  └── Phase G: Fama-French 3/5-factor attribution
  └── Phase H: FinRL reinforcement learning portfolio agent
```

---

## Dashboard Page Development

New pages go in `src/quant_dashboard/pages/` with naming `N_Page_Name.py`.

Each page must:
- Use only chart helpers from `components/charts.py` (testable without Streamlit)
- Accept all numeric inputs via `st.sidebar` / `st.number_input`
- Handle exceptions with `st.error()` — never crash the app

---

## Skill Maintenance Cadence

After each phase completes:
1. Update `.claude/skills/quant-dev/SKILL.md` — add new patterns, remove obsolete ones
2. Create phase-specific skill if the new module has a unique testing/validation pattern
3. Run `npx gitnexus analyze` — update AGENTS.md/CLAUDE.md
4. Update this SOP with new phase status

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
