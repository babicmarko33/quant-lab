# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

## [1.4.0] — 2026-05-06

### Added — Phase E: Tradier Options Client (10 tests)
- `src/quantcore/data/tradier_client.py` — `TradierClient` with `get_expirations`, `get_option_chain`, `get_quote`; `OptionQuote` dataclass (strike, greeks, mid_iv); `OptionChain.to_dataframe()` — all HTTP mocked in tests; sandbox/production URL selection; token from env or constructor

---

## [1.3.0] — 2026-05-06

### Added — Phase D: HMM Regime Detection (18 tests)
- `src/alpha_engine/regime/hmm_classifier.py` — `RegimeClassifier` using hmmlearn 0.3 GaussianHMM; `fit`, `predict`, `predict_proba`; multi-feature (return + abs_return); 2–4 regimes supported (11 tests)
- `src/alpha_engine/regime/regime_filtered_strategy.py` — `RegimeFilteredStrategy` wraps any `Strategy`, zeroes signals in non-active regimes (7 tests)
- `src/quant_dashboard/pages/6_Regime_Analysis.py` — 3-tab dashboard: Regime Timeline with vrect bands + posterior probabilities, Per-Regime statistics, Filtered vs Unfiltered backtest comparison
- `regime_timeline_fig` + `regime_vol_bar_fig` chart helpers added to `charts.py`

---

## [1.2.0] — 2026-05-06

### Added — Phase C: Kalman/Pairs Trading (30 tests)
- `src/quantcore/signals/cointegration.py` — `engle_granger_test`, `find_cointegrated_pairs`, `hedge_ratio` via statsmodels (13 tests)
- `src/quantcore/signals/kalman_filter.py` — `KalmanPairFilter` dynamic hedge ratio + z-score via state-space model (9 tests)
- `src/alpha_engine/strategies/pairs.py` — `PairsStrategy` entry/exit on Kalman z-score (8 tests)

---

## [1.1.0] — 2026-05-06

### Added — Phase B: GARCH Volatility Forecasting (21 tests)
- `src/alpha_engine/derivatives/volatility/garch.py` — `fit_garch`, `forecast_volatility`, `garch_vol_forecast`; GARCH(p,q) via arch 8.0; returns/variance decimal-scaling; stationarity check (14 tests)
- `VolatilitySurface.from_garch_forecasts()` — builds term-structure surface from GARCH daily vol forecasts (7 tests)
- `arch>=6.3`, `statsmodels>=0.14` added to `pyproject.toml` `[phase7]` extras

---

## [1.0.0] — 2026-05-06

### Added — Phase 6: Dashboard Completion
- `4_Options_Pricing.py` — 3 tabs: model comparison (BSM/Binomial/MC/PDE), Greeks sensitivity, SABR smile + calibration
- `5_Market_Data.py` — 3 tabs: OHLCV candlestick + volume, returns analysis, SABR vol surface heatmap
- `charts.py` extended with 7 new chart helpers: `payoff_fig`, `greeks_sensitivity_fig`, `model_comparison_bar_fig`, `ohlcv_fig`, `rolling_vol_fig`, `vol_surface_heatmap_fig`, `sabr_smile_fig`
- **Total: 433 tests passing** (all phases complete)

---

## [0.9.0] — 2026-05-06

### Added — Phase 5.1–5.3: ML Model Additions (33 tests)
- `src/alpha_ml/models/linear.py` — `RidgeClassifier`, `LassoClassifier` via LogisticRegression (13 tests)
- `src/alpha_ml/models/lstm.py` — `LSTMClassifier` PyTorch sliding-window sequence-to-one binary classifier (9 tests)
- `src/alpha_ml/models/ensemble.py` — `ModelEnsemble` soft-voting over arbitrary `ModelTrainer` instances (11 tests)

---

## [0.8.0] — 2026-05-06

### Added — Phase 4.4–4.7: Advanced Portfolio Allocators (38 tests)
- `src/alpha_engine/portfolio/cardinality.py` — `CardinalityAllocator` MIQP via CVXPY/SCIP (8 tests)
- `src/alpha_engine/portfolio/mc_risk.py` — `mc_var`, `mc_cvar` with Cholesky decomposition for correlated assets (10 tests)
- `src/alpha_engine/portfolio/merton_hjb.py` — `merton_optimal_weight`, `merton_value_function` via Hamilton-Jacobi-Bellman (13 tests)
- `src/alpha_engine/portfolio/black_litterman.py` — `BlackLittermanAllocator` posterior returns from investor views (7 tests)

---

## [0.7.0] — 2026-05-06

### Added — Phase 3.3–3.8: Advanced Derivatives Pricing (68 tests)
- `src/alpha_engine/derivatives/options/monte_carlo.py` — `mc_european`, `mc_asian`, `mc_barrier` with antithetic + control variates (16 tests)
- `src/alpha_engine/derivatives/options/binomial.py` — `binomial_price` CRR model, European + American early exercise (10 tests)
- `src/alpha_engine/derivatives/options/pde.py` — `pde_european` Crank-Nicolson; `pde_american` PSOR for linear complementarity (15 tests)
- `src/alpha_engine/derivatives/options/levy.py` — `merton_price` jump-diffusion PIDE series; `vg_price` Variance Gamma MC (14 tests)
- `src/alpha_engine/derivatives/volatility/sabr.py` — `sabr_vol` (Hagan et al. 2002); `sabr_calibrate` RMSE minimization (13 tests)

---

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
