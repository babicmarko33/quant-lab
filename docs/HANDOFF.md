# quant-lab — Project Handoff Document

**Date:** 2026-05-06  
**State:** All Phases 0–6 complete. 433 tests passing. Phase 7 ready to begin.

---

## 1. What Was Built

quant-lab is an institutional-grade quantitative finance monorepo with:

| Layer | What's in it |
|-------|-------------|
| **Data** | Multi-source OHLCV fetcher (yfinance → Alpaca fallback), parquet cache, OHLCV schema validation |
| **Indicators** | NumPy-vectorized: SMA, EMA, RSI (Wilder), MACD, Bollinger Bands, ATR |
| **Strategies** | Momentum (12-1 cross-sectional), Bollinger mean-reversion, SMA/EMA cross, RSI MR |
| **Backtesting** | Vectorized engine (signal T → fill T+1 OPEN), walk-forward with embargo, multi-asset |
| **Analytics** | Sharpe, Sortino, Calmar, max drawdown, PSR, IC, turnover |
| **Risk** | Kelly criterion, fractional Kelly, volatility targeting |
| **Portfolio** | 8 allocators: EqualWeight, MeanVariance, RiskParity (ERC), CVaR LP, Cardinality MIQP (SCIP), MC VaR/CVaR (Cholesky), Merton HJB (stochastic control), Black-Litterman |
| **Derivatives** | BSM + full Greeks, IV solver (NR+Brent), multi-leg strategies, Binomial CRR, MC (European/Asian/Barrier), Crank-Nicolson PDE + PSOR, Merton jump-diffusion PIDE, Variance Gamma MC, SABR calibration, VolatilitySurface |
| **ML** | XGBoost, Ridge/Lasso classifiers, LSTM (PyTorch sliding-window), ModelEnsemble (soft-voting); PurgedKFold; FeatureStore |
| **Execution** | Alpaca REST paper trading, signal deduplication, mock-testable Broker ABC |
| **Dashboard** | 5-page Streamlit app: Equity Curve, Portfolio Weights, ML Signals, Options Pricing, Market Data + Vol Surface |

---

## 2. Current Test Count

```
433 tests — all passing
Run: pytest -q --tb=short
Lint: ruff check src/ tests/  →  0 errors
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

## 6. Phase 7 — What Comes Next

Phase 7 is the Advanced Research track. Install dependencies first:

```powershell
pip install "arch>=6.3" "hmmlearn>=0.3" "statsmodels>=0.14" "filterpy>=1.4" "websocket-client>=1.7" "stable-baselines3>=2.3" "gymnasium>=0.29"
```

Phases in order:

| Phase | Module | Key files to create |
|-------|--------|-------------------|
| B | GARCH | `src/alpha_engine/derivatives/volatility/garch.py` |
| C | Kalman/Pairs | `src/quantcore/signals/cointegration.py`, `src/quantcore/signals/kalman_filter.py`, `src/alpha_engine/strategies/pairs.py` |
| D | HMM Regime | `src/alpha_engine/regime/hmm_classifier.py`, `src/alpha_engine/strategies/regime_filtered.py` |
| E | Tradier Options | `src/quantcore/data/tradier_client.py` |
| F | Live Event Loop | `src/alpha_engine/execution/alpaca_stream.py`, `src/alpha_engine/execution/live_trader.py` |
| G | Fama-French | `src/quantcore/data/fama_french.py`, `src/alpha_engine/analytics/factor_model.py` |
| H | FinRL | `src/alpha_ml/rl/portfolio_env.py`, `src/alpha_ml/rl/rl_agent.py` |

All phases use strict TDD: write failing test first, show RED, implement, show GREEN, commit.

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
