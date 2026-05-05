# 🧪 quant-lab

**Institutional-grade quantitative finance monorepo** — systematic alpha research, ML-driven strategies, and real-time analytics.

[![CI](https://github.com/babicmarko33/quant-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/babicmarko33/quant-lab/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Architecture

```
quant-lab/
├── src/
│   ├── quantcore/          # Shared data, indicators, feature engineering
│   │   ├── data/           # Multi-source fetcher (yfinance → Alpaca → Polygon)
│   │   ├── indicators/     # Vectorized technical analysis (SMA, EMA, RSI, MACD, BB, ATR)
│   │   ├── features/       # Feature pipeline for strategies & ML
│   │   └── utils/          # Shared utilities
│   ├── alpha_engine/       # Strategy development & backtesting
│   │   ├── strategies/     # Strategy implementations (momentum, mean-reversion, stat-arb)
│   │   ├── backtest/       # Event-driven backtester with walk-forward validation
│   │   ├── risk/           # Position sizing, drawdown control, VaR/CVaR
│   │   └── execution/      # Paper trading & live execution adapters
│   ├── alpha_ml/           # ML models for financial prediction
│   │   ├── models/         # Gradient boosting, LSTM, Transformers
│   │   ├── training/       # Walk-forward training with purged k-fold CV
│   │   └── evaluation/     # PSR/DSR, calibration, feature importance
│   └── quant_dashboard/    # Streamlit real-time analytics
├── tests/                  # Comprehensive test suite
├── notebooks/              # Research notebooks (Jupyter/Marimo)
├── docs/sops/              # Standard operating procedures
└── data/                   # Local cache (gitignored)
```

## Key Features

- **Multi-source data pipeline** — Automatic fallback chain with parquet caching
- **Vectorized indicators** — NumPy-optimized technical analysis (zero Python loops)
- **Feature engineering** — Configurable pipeline with look-ahead bias protection
- **Event-driven backtester** — Walk-forward validation, transaction costs, slippage modeling
- **ML integration** — Purged k-fold CV, Probabilistic/Deflated Sharpe Ratio
- **Risk management** — Kelly criterion sizing, drawdown control, correlation-aware allocation
- **Production execution** — Alpaca paper trading with live order management

## Quick Start

```bash
# Clone
git clone https://github.com/babicmarko33/quant-lab.git
cd quant-lab

# Install (editable mode with dev dependencies)
pip install -e ".[dev]"

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Run tests
pytest -x --tb=short

# Run with coverage
pytest --cov=src --cov-report=term-missing
```

## Usage

### Fetch Market Data

```python
from quantcore.data.fetcher import fetch_ohlcv

# Automatic provider fallback + local parquet caching
spy = fetch_ohlcv("SPY", start="2020-01-01")
print(spy.tail())
```

### Compute Indicators

```python
from quantcore.indicators.technical import rsi, macd, bollinger_bands

rsi_14 = rsi(spy["close"], window=14)
macd_df = macd(spy["close"])
bb = bollinger_bands(spy["close"], window=20, num_std=2.0)
```

### Build Feature Matrix

```python
from quantcore.features.pipeline import build_features, add_target

features = build_features(spy, sma_windows=[10, 20, 50], rsi_window=14)
dataset = add_target(features, spy["close"], horizon=5, target_type="direction")
# Drop NaN rows (lookback warmup) before training
dataset_clean = dataset.dropna()
```

## Development

```bash
# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Type check
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

## Research Methodology

This project follows a rigorous quantitative research workflow:

1. **Hypothesis formation** — State testable prediction with clear null hypothesis
2. **Data validation** — Check for survivorship bias, look-ahead bias, data snooping
3. **Walk-forward testing** — Purged k-fold CV to prevent information leakage
4. **Statistical validation** — Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR)
5. **Out-of-sample verification** — Hold-out period that was never touched during development

## Roadmap

- [x] QuantCore data pipeline + indicators + feature engineering
- [ ] AlphaEngine: Event-driven backtester with walk-forward
- [ ] AlphaEngine: Momentum, mean-reversion, and stat-arb strategies
- [ ] AlphaML: Gradient boosting + LSTM models
- [ ] Risk: Kelly criterion + correlation-aware portfolio allocation
- [ ] Dashboard: Streamlit real-time P&L and analytics
- [ ] Execution: Alpaca paper trading integration

## License

MIT
