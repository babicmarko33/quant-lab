# quant-lab

**Institutional-grade quantitative finance monorepo** — systematic alpha research, ML-driven strategies, portfolio optimisation, and real-time analytics.

[![CI](https://github.com/babicmarko33/quant-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/babicmarko33/quant-lab/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![241 tests](https://img.shields.io/badge/tests-241%20passing-brightgreen)](https://github.com/babicmarko33/quant-lab/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Architecture

```
quant-lab/
├── src/
│   ├── quantcore/              # Shared data, indicators, feature engineering
│   │   ├── data/               # Multi-source fetcher (yfinance -> Alpaca -> Polygon)
│   │   ├── indicators/         # Vectorized: SMA, EMA, RSI, MACD, BB, ATR
│   │   └── features/           # Feature pipeline with look-ahead bias protection
│   ├── alpha_engine/           # Strategy development & backtesting
│   │   ├── strategies/         # Momentum (12-1), Bollinger mean-reversion; Strategy registry
│   │   ├── backtest/           # Vectorized engine + walk-forward + multi-asset backtester
│   │   ├── analytics/          # Sharpe, Sortino, PSR, IC, turnover
│   │   ├── risk/               # Kelly criterion, fractional Kelly, volatility targeting
│   │   ├── portfolio/          # EqualWeight, MeanVariance, RiskParity, CVaR allocators
│   │   └── execution/          # Order, Broker ABC, AlpacaBroker, PaperTrader
│   ├── alpha_ml/               # ML signal generation
│   │   ├── features/           # FeatureStore: winsorize + z-score + target labels
│   │   ├── models/             # ModelTrainer ABC + XGBoost binary classifier
│   │   ├── validation/         # PurgedKFold (Lopez de Prado), cross_val_predict_purged
│   │   └── pipeline.py         # End-to-end OHLCV -> signals -> BacktestResult
│   └── quant_dashboard/        # Streamlit multi-page analytics dashboard
│       ├── components/charts.py# Pure Plotly chart factories (testable without Streamlit)
│       └── pages/              # 1_Equity_Curve, 2_Portfolio, 3_ML_Signal
├── tests/                      # 190+ tests; strict TDD red-green-refactor
├── scripts/                    # run_backtest.py, run_paper_trader.py
├── notebooks/                  # Research notebooks
├── docs/
│   ├── sops/                   # 8 Standard Operating Procedures
│   └── plans/                  # Phase implementation plans
└── data/                       # Local parquet cache (gitignored)
```

---

## Key Capabilities

| Module | Highlights |
|--------|-----------|
| **Data** | Auto-fallback chain (yfinance -> Alpaca), local parquet caching, full OHLCV |
| **Indicators** | NumPy-vectorized: SMA, EMA, RSI (Wilder), MACD, Bollinger Bands, ATR |
| **Backtesting** | Signal T -> fill T+1 OPEN, slippage/commission modeled, zero look-ahead |
| **Walk-Forward** | Anchored folds with embargo (Lopez de Prado ch7), mean OOS Sharpe |
| **ML Pipeline** | XGBoost + purged k-fold CV + PSR / IC evaluation |
| **Portfolio** | MV, Risk Parity (Spinu 2013), CVaR LP (Rockafellar-Uryasev 2000) |
| **Execution** | Alpaca paper trading; signal dedup; mock-testable broker interface |
| **Dashboard** | 3-page Streamlit app: equity curve, portfolio weights, ML feature importance |

---

## Quick Start

```bash
git clone https://github.com/babicmarko33/quant-lab.git
cd quant-lab

# Full install (all test dependencies)
pip install -e ".[ci]"

# Copy and populate environment variables
cp .env.example .env

# Run all tests (190 passing)
pytest -m "not integration" -q

# Launch the dashboard
streamlit run src/quant_dashboard/app.py
```

---

## Usage Examples

### Run a Backtest

```python
from quantcore.data.fetcher import fetch_ohlcv
from alpha_engine.strategies.momentum import MomentumStrategy

spy = fetch_ohlcv("SPY", start="2020-01-01")
result = MomentumStrategy().run(spy)
print(f"Sharpe: {result.sharpe:.2f}  MaxDD: {result.max_drawdown:.1%}  Total: {result.total_return:.1%}")
```

### Portfolio Allocation

```python
import pandas as pd
from alpha_engine.portfolio import RiskParityAllocator

returns = pd.DataFrame({t: fetch_ohlcv(t, start="2021-01-01")["close"].pct_change()
                        for t in ["SPY", "TLT", "GLD"]}).dropna()
weights = RiskParityAllocator().fit(returns)
print(weights)
```

### XGBoost ML Signal

```python
from alpha_ml.pipeline import MLSignalPipeline
from alpha_ml.models.xgboost_model import XGBoostModel

pipeline = MLSignalPipeline(model=XGBoostModel(), train_ratio=0.6, horizon=5)
result = pipeline.run(fetch_ohlcv("SPY", start="2019-01-01"))
print(f"OOS Sharpe: {result.sharpe:.2f}")
```

### Paper Trading CLI

```bash
# Signal-only (no Alpaca keys needed)
python scripts/run_paper_trader.py --ticker SPY --strategy momentum

# Live paper submission (requires ALPACA_API_KEY + ALPACA_SECRET_KEY in .env)
python scripts/run_paper_trader.py --ticker SPY --strategy bollinger_mr --qty 10
```

---

## Development

```bash
# Lint + format check
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/

# Full test suite
pytest -m "not integration" -q

# Coverage report
pytest --cov=src --cov-report=term-missing -m "not integration" -q

# Pre-commit hooks
pre-commit run --all-files
```

---

## Documentation

| Document | Location |
|----------|----------|
| Project workflow (TDD, phases) | [docs/sops/project-workflow.md](docs/sops/project-workflow.md) |
| Backtesting protocol | [docs/sops/backtesting-protocol.md](docs/sops/backtesting-protocol.md) |
| Research methodology | [docs/sops/research-methodology.md](docs/sops/research-methodology.md) |
| ML signal pipeline | [docs/sops/ml-signal-pipeline.md](docs/sops/ml-signal-pipeline.md) |
| Portfolio allocation | [docs/sops/portfolio-allocation.md](docs/sops/portfolio-allocation.md) |
| Execution & paper trading | [docs/sops/execution-and-paper-trading.md](docs/sops/execution-and-paper-trading.md) |
| Git workflow | [docs/sops/git-workflow.md](docs/sops/git-workflow.md) |
| Data handling | [docs/sops/data-handling.md](docs/sops/data-handling.md) |

---

## Roadmap

- [x] **Phase 0** — Infrastructure: pyproject, CI/CD, data fetcher, indicators, features
- [x] **Phase 1** — AlphaEngine: vectorized backtester, momentum + Bollinger, walk-forward, risk (87 tests)
- [x] **Phase 2** — AlphaML: XGBoost + purged k-fold CV + MLSignalPipeline (129 tests)
- [x] **Phase 3** — Portfolio: MeanVariance, RiskParity, CVaR, multi-asset backtester (162 tests)
- [x] **Phase 4** — Dashboard + Paper Trading: Streamlit app, AlpacaBroker, PaperTrader (190 tests)
- [x] **Phase 5** — Derivatives Lab: Black-Scholes, Greeks, implied volatility, vol surface (241 tests)
- [ ] **Phase 6** — Production: live execution, monitoring, alerting

---

## License

MIT

