# quant-lab — Project Handoff Document

**Date:** 2026-05-06  
**State:** All Phases 0–6 + Phase 7 (A–H) complete. 546 tests passing.

---

## 1. What Was Built

quant-lab is an institutional-grade quantitative finance monorepo with:

| Layer | What's in it |
|-------|-------------|
| **Data** | Multi-source OHLCV fetcher (yfinance → Alpaca fallback), parquet cache, OHLCV schema validation, FamaFrenchFetcher (monthly 3-factor), TradierClient (live options chain) |
| **Indicators** | NumPy-vectorized: SMA, EMA, RSI (Wilder), MACD, Bollinger Bands, ATR |
| **Strategies** | Momentum (12-1 cross-sectional), Bollinger mean-reversion, SMA/EMA cross, RSI MR, PairsStrategy (Kalman spread), RegimeFilteredStrategy (HMM gate) |
| **Backtesting** | Vectorized engine (signal T → fill T+1 OPEN), walk-forward with embargo, multi-asset |
| **Analytics** | Sharpe, Sortino, Calmar, max drawdown, PSR, IC, turnover |
| **Risk** | Kelly criterion, fractional Kelly, volatility targeting |
| **Portfolio** | 8 allocators: EqualWeight, MeanVariance, RiskParity (ERC), CVaR LP, Cardinality MIQP (SCIP), MC VaR/CVaR (Cholesky), Merton HJB (stochastic control), Black-Litterman |
| **Derivatives** | BSM + full Greeks, IV solver (NR+Brent), multi-leg strategies, Binomial CRR, MC (European/Asian/Barrier), Crank-Nicolson PDE + PSOR, Merton jump-diffusion PIDE, Variance Gamma MC, SABR calibration, VolatilitySurface, GarchVolatilityModel |
| **ML** | XGBoost, Ridge/Lasso classifiers, LSTM (PyTorch sliding-window), ModelEnsemble (soft-voting); PurgedKFold; FeatureStore |
| **Factor** | FactorModel OLS attribution (alpha, betas, R², t-stats) against Fama-French 3-factor |
| **Regime** | GaussianHMM RegimeClassifier (n_regimes, predict_proba), RegimeFilteredStrategy wrapper |
| **Live** | AlpacaBarStream (WebSocket), LiveTrader (rolling buffer → strategy signal), AlpacaBroker paper trading |
| **RL** | PortfolioEnv (gymnasium Env, log-return reward), RLPortfolioAgent (PPO via stable-baselines3) |
| **Dashboard** | 6-page Streamlit app: Equity Curve, Portfolio Weights, ML Signals, Options Pricing, Market Data + Vol Surface, Regime Analysis |

---

## 2. Current Test Count

```
546 tests — all passing
Run: & "..\.venv\Scripts\python.exe" -m pytest tests/ -q --tb=short
Lint: & "..\.venv\Scripts\ruff.exe" check src/ tests/  →  0 errors
```

---

## 3. Environment Setup

```powershell
cd "C:\Users\marko_babic\Desktop\Projects\Quant\quant-lab"
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[ci]"
cp .env.example .env   # fill in API keys
```

Required `.env` keys:
- `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` — for Alpaca broker and data fallback
- `ALPHAVANTAGE_API_KEY` — for Alpha Vantage (stub, not yet active)
- `POLYGON_API_KEY` — for Polygon (stub, not yet active)

Phase 7 will also need:
- `TRADIER_API_KEY` / `TRADIER_BASE_URL` — for live options chain

---

## 4. Known Technical Debt

| Item | File | Severity |
|------|------|----------|
| Alpha Vantage fetcher is docstring-only stub | `src/quantcore/data/fetcher.py` | Low |
| Polygon fetcher is docstring-only stub | `src/quantcore/data/fetcher.py` | Low |
| SABR formula degenerate at `F ≈ K` (special-case present, but not battle-tested) | `src/alpha_engine/derivatives/volatility/sabr.py` | Low |
| VolatilitySurface needs `from_garch_forecasts()` (Phase B) | `src/alpha_engine/derivatives/volatility/surface.py` | Medium |
| VolatilitySurface needs `from_options_chain()` (Phase E) | `src/alpha_engine/derivatives/volatility/surface.py` | Medium |
| Phase 7 dependencies not installed | `pyproject.toml` extras | Medium |

---

## 5. Architecture Decisions

**Why `shift(1)` everywhere in signals:**  
Avoids look-ahead bias. Signal for day T uses only data available at end of day T-1.

**Why `BacktestEngine` fills at T+1 OPEN:**  
Realistic for EOD strategies — you see the signal at close, execute at next open.

**Why CVXPY/SCIP for cardinality:**  
MIQP is NP-hard; SCIP (open-source MIP solver) provides exact solutions for small N (≤50 assets).

**Why PyTorch for LSTM (not Keras/TF):**  
Lighter dependency, easier to mock in tests, compatible with `scikit-learn`-style API via `ModelTrainer`.

**Why Streamlit for dashboard:**  
Fastest iteration for research dashboards. Not intended for production user-facing deployment.

---

## 6. Phase 7 — Completed

All Phase 7 sub-phases are complete:

| Phase | Module | Status |
|-------|--------|--------|
| A | Infrastructure (GitNexus, README, CHANGELOG, SOPs) | ✅ |
| B | GARCH volatility forecasting | ✅ 21 tests |
| C | Kalman filter + Pairs strategy | ✅ 30 tests |
| D | HMM Regime classifier + Streamlit page | ✅ 18 tests |
| E | Tradier REST options client | ✅ 10 tests |
| F | AlpacaBarStream + LiveTrader | ✅ 12 tests |
| G | FamaFrenchFetcher + FactorModel OLS | ✅ 11 tests |
| H | PortfolioEnv (gymnasium) + RLPortfolioAgent (PPO) | ✅ 11 tests |

---

## 7. Claude Skills Available

Skills live in `.claude/skills/` and `c:\Users\marko_babic\.agents\skills\`:

| Skill | When to load |
|-------|-------------|
| `quant-dev` | Any new module implementation |
| `quant-backtest-validation` | Validating backtest results |
| `quant-data-pipeline` | Data fetcher or feature engineering changes |
| `gitnexus/gitnexus-impact-analysis` | Before editing any existing function |
| `brainstorming` | Design decisions, architecture choices |
| `writing-plans` | Planning multi-phase work |
| `test-driven-development` | TDD cycle enforcement |
| `executing-plans` | Running a written plan |
| `kaizen` | Continuous improvement reviews |
| `software-architecture` | Module structure decisions |

---

## 8. GitNexus State

```
Last indexed: HEAD (8e658ea → 5074d9f → e288943 → current)
Nodes: 1,064 | Edges: 2,703 | Flows: 16
```

Run `npx gitnexus analyze` from `quant-lab/` after any commit.  
AGENTS.md and CLAUDE.md are auto-regenerated.

---

## Contact / Ownership

Repository: `https://github.com/babicmarko33/quant-lab`  
Owner: Marko Babic  
License: MIT
