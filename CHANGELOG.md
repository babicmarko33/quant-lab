# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

## [0.5.0] — 2026-05-06

### Added — Phase 5: Derivatives Lab (51 tests)
- `src/alpha_engine/derivatives/options/black_scholes.py` — BSM analytical pricing (`bsm_price`) + full Greeks (`bsm_greeks`: Delta, Gamma, Theta, Vega, Rho)
- `src/alpha_engine/derivatives/options/implied_vol.py` — IV solver: Newton-Raphson + Brent's method fallback
- `src/alpha_engine/derivatives/options/strategies.py` — Multi-leg P&L at expiry: covered call, protective put, straddle, strangle, iron condor
- `src/alpha_engine/derivatives/volatility/surface.py` — `VolatilitySurface`: cubic spline interpolation on (strike, expiry) grid

### Added — Infrastructure overhaul
- `CONTRIBUTING.md` — TDD workflow, commit conventions, PR guidelines
- `.github/dependabot.yml` — Weekly automated dependency updates (pip + GitHub Actions)
- `docs/sops/execution-and-paper-trading.md` — New SOP
- `docs/sops/portfolio-allocation.md` — New SOP
- `docs/sops/ml-signal-pipeline.md` — New SOP
- Restructured `pyproject.toml` extras: `ci`, `deep-learning`; declared `xgboost`, `cvxpy` in `ml`
- CI now installs `.[ci]` (covers all test dependencies); Python 3.11/3.12 matrix
- Pre-commit: ruff bumped to v0.4.4, mypy to v1.10.0, added `check-toml` hook

---

## [0.4.0] — 2026-05-05

### Added — Phase 4: Dashboard + Paper Trading (190 tests)
- `src/alpha_engine/execution/` — `Order`, `Broker` ABC, `AlpacaBroker`, `PaperTrader`
- `src/quant_dashboard/` — Streamlit 3-page app: Equity Curve, Portfolio, ML Signal
- `src/quant_dashboard/components/charts.py` — 5 pure Plotly chart factories
- `scripts/run_paper_trader.py` — CLI for paper trading with strategy selection

---

## [0.3.0] — 2026-05-04

### Added — Phase 3: Portfolio Optimization (162 tests)
- `src/alpha_engine/portfolio/` — `EqualWeightAllocator`, `MeanVarianceAllocator` (max Sharpe / min variance), `RiskParityAllocator` (Spinu 2013 ERC), `CVaRAllocator` (Rockafellar-Uryasev LP)
- `src/alpha_engine/backtest/multi_asset.py` — `run_portfolio_backtest` with configurable rebalancing frequency

---

## [0.2.0] — 2026-05-03

### Added — Phase 2: AlphaML (129 tests)
- `src/alpha_ml/validation/purged_kfold.py` — `PurgedKFold` (López de Prado 2018)
- `src/alpha_ml/features/feature_store.py` — `FeatureStore` with winsorization + z-score
- `src/alpha_ml/models/xgboost_model.py` — `XGBoostModel` binary classifier
- `src/alpha_ml/pipeline.py` — End-to-end `MLSignalPipeline`

---

## [0.1.0] — 2026-05-02

### Added — Phase 1: AlphaEngine (87 tests)
- `src/alpha_engine/backtest/engine.py` — Vectorized backtesting engine
- `src/alpha_engine/backtest/walk_forward.py` — Walk-forward validation
- `src/alpha_engine/strategies/` — `Strategy` ABC, `MomentumStrategy`, `BollingerMeanReversionStrategy`
- `src/alpha_engine/analytics/performance.py` — Sharpe, Sortino, Calmar, max drawdown, etc.
- `src/alpha_engine/risk/position_sizing.py` — Kelly criterion, volatility targeting

---

## [0.0.1] — 2026-05-01

### Added — Phase 0: Infrastructure
- Repository scaffolding: `pyproject.toml`, `ruff`, `pre-commit`, `.github/workflows/ci.yml`
- `src/quantcore/data/fetcher.py` — Multi-source OHLCV fetcher (yfinance → Alpaca → Polygon)
- `src/quantcore/indicators/technical.py` — SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- `src/quantcore/features/pipeline.py` — Feature engineering pipeline
- `docs/sops/` — 5 SOPs: project-workflow, backtesting-protocol, data-handling, git-workflow, research-methodology
