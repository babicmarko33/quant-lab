# Phase 4: Streamlit Dashboard + Paper Trading Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a multi-page Streamlit analytics dashboard wired to real market data and an Alpaca paper-trading execution layer.

**Architecture:** Three self-contained layers — (1) `quant_dashboard/` multi-page Streamlit app for interactive research; (2) `alpha_engine/execution/` Alpaca adapter that submits/tracks paper orders; (3) `scripts/run_paper_trader.py` CLI that runs a live strategy loop. All testable in isolation via mocks/fakes — no live Alpaca account required for CI.

**Tech Stack:** Streamlit ≥1.28, Plotly ≥5.18, alpaca-trade-api 3.2 (REST + websocket), pandas 3.x, pytest-mock for unit tests.

---

## Task 1: Install dashboard dependencies

**Files:**
- Modify: `pyproject.toml` (dependencies section)

**Step 1: Install Streamlit + Plotly into the venv**

```powershell
pip install "streamlit>=1.28" "plotly>=5.18"
```

**Step 2: Verify**

```powershell
python -c "import streamlit; import plotly; print('ok')"
```

**Step 3: Commit**

```
chore(deps): install streamlit and plotly for Phase 4
```

---

## Task 2: Alpaca execution adapter — order model + interface

**Files:**
- Create: `src/alpha_engine/execution/__init__.py`
- Create: `src/alpha_engine/execution/order.py`
- Test: `tests/test_alpha_engine/test_execution.py`

**Domain:**  
An `Order` dataclass carries everything needed to submit a market order:
- `symbol: str`, `side: str` (`"buy"` / `"sell"`), `qty: float`, `order_id: str | None`
- `status: str` defaults to `"pending"` — set to `"filled"` / `"cancelled"` by the broker adapter

**Step 1: Write failing tests**

```python
# tests/test_alpha_engine/test_execution.py
from alpha_engine.execution.order import Order

def test_order_defaults():
    o = Order(symbol="SPY", side="buy", qty=10)
    assert o.status == "pending"
    assert o.order_id is None

def test_order_side_validation():
    with pytest.raises(ValueError):
        Order(symbol="SPY", side="short", qty=10)

def test_order_qty_positive():
    with pytest.raises(ValueError):
        Order(symbol="SPY", side="buy", qty=-5)

def test_order_repr_contains_symbol():
    o = Order(symbol="AAPL", side="sell", qty=5)
    assert "AAPL" in repr(o)
```

**Step 2: Run — confirm RED**

```powershell
pytest tests/test_alpha_engine/test_execution.py -q
```

**Step 3: Implement `Order`**

```python
# src/alpha_engine/execution/order.py
from __future__ import annotations
from dataclasses import dataclass, field
import uuid

VALID_SIDES = {"buy", "sell"}

@dataclass
class Order:
    symbol: str
    side: str
    qty: float
    order_id: str | None = None
    status: str = "pending"

    def __post_init__(self) -> None:
        if self.side not in VALID_SIDES:
            raise ValueError(f"side must be one of {VALID_SIDES}, got '{self.side}'")
        if self.qty <= 0:
            raise ValueError(f"qty must be positive, got {self.qty}")
        if self.order_id is None:
            object.__setattr__(self, "order_id", str(uuid.uuid4()))
```

**Step 4: Run — confirm GREEN**

```powershell
pytest tests/test_alpha_engine/test_execution.py -q
```

**Step 5: ruff + commit**

```
feat(execution): Order dataclass with validation
```

---

## Task 3: Alpaca broker adapter (interface + paper implementation)

**Files:**
- Create: `src/alpha_engine/execution/broker.py`
- Create: `src/alpha_engine/execution/alpaca_broker.py`
- Modify: `src/alpha_engine/execution/__init__.py`
- Test: `tests/test_alpha_engine/test_execution.py` (extend)

**Domain:**  
Abstract `Broker` ABC with one method:
```python
def submit(self, order: Order) -> Order:  # returns filled/rejected order copy
```
`AlpacaBroker` calls `alpaca_trade_api.REST.submit_order(...)` and returns an updated `Order`.
For testing: `FakeBroker` that immediately fills all orders — lives in `tests/` only.

**Step 1: Add failing tests**

```python
# Extend test_execution.py
from unittest.mock import MagicMock, patch
from alpha_engine.execution.broker import Broker
from alpha_engine.execution.alpaca_broker import AlpacaBroker

def test_broker_is_abstract():
    with pytest.raises(TypeError):
        Broker()  # type: ignore[abstract]

def test_alpaca_broker_submit_buy(monkeypatch):
    """AlpacaBroker.submit() returns order with status='filled'."""
    mock_api = MagicMock()
    mock_api.submit_order.return_value = MagicMock(id="abc123", status="filled")
    with patch("alpha_engine.execution.alpaca_broker.tradeapi.REST", return_value=mock_api):
        broker = AlpacaBroker(api_key="k", secret_key="s", base_url="https://paper-api.alpaca.markets")
        order = Order(symbol="SPY", side="buy", qty=1)
        filled = broker.submit(order)
    assert filled.status == "filled"
    assert filled.order_id == "abc123"
```

**Step 2: Run — confirm RED**

**Step 3: Implement `Broker` ABC and `AlpacaBroker`**

```python
# src/alpha_engine/execution/broker.py
from abc import ABC, abstractmethod
from alpha_engine.execution.order import Order

class Broker(ABC):
    @abstractmethod
    def submit(self, order: Order) -> Order: ...

# src/alpha_engine/execution/alpaca_broker.py
import alpaca_trade_api as tradeapi
from alpha_engine.execution.broker import Broker
from alpha_engine.execution.order import Order
import dataclasses

class AlpacaBroker(Broker):
    def __init__(self, api_key: str, secret_key: str, base_url: str) -> None:
        self._api = tradeapi.REST(api_key, secret_key, base_url)

    def submit(self, order: Order) -> Order:
        resp = self._api.submit_order(
            symbol=order.symbol,
            qty=order.qty,
            side=order.side,
            type="market",
            time_in_force="day",
        )
        return dataclasses.replace(order, order_id=str(resp.id), status=str(resp.status))
```

**Step 4: Run — confirm GREEN**

**Step 5: ruff + commit**

```
feat(execution): Broker ABC + AlpacaBroker paper trading adapter
```

---

## Task 4: Paper trader loop — signal-to-order wiring

**Files:**
- Create: `src/alpha_engine/execution/paper_trader.py`
- Test: `tests/test_alpha_engine/test_paper_trader.py`

**Domain:**  
`PaperTrader` takes a `Strategy`, a `Broker`, and a live prices DataFrame.
On `run(df)`:
1. Calls `strategy.generate_signals(df)` → last signal value (`+1`, `-1`, `0`)
2. If signal changed from previous → submit order
3. Returns list of submitted `Order`s

**Step 1: Write failing tests**

```python
# tests/test_alpha_engine/test_paper_trader.py
import pandas as pd, numpy as np, pytest
from unittest.mock import MagicMock
from alpha_engine.execution.paper_trader import PaperTrader
from alpha_engine.execution.order import Order

@pytest.fixture
def rising_df():
    """100 days of steadily rising prices — momentum signal = +1."""
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = np.linspace(100, 150, n)
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1e6}, index=dates)

def test_paper_trader_submits_buy_on_long_signal(rising_df):
    from alpha_engine.strategies.momentum import MomentumStrategy
    strategy = MomentumStrategy()
    fake_broker = MagicMock()
    fake_broker.submit.return_value = Order(symbol="SPY", side="buy", qty=1, status="filled")
    trader = PaperTrader(symbol="SPY", qty_per_trade=1, broker=fake_broker)
    orders = trader.run(rising_df, strategy)
    # Momentum on 300 rising bars should generate a long signal → buy submitted
    assert any(o.side == "buy" for o in orders)

def test_paper_trader_no_duplicate_orders(rising_df):
    """Same signal on two consecutive runs → no new order second time."""
    from alpha_engine.strategies.momentum import MomentumStrategy
    strategy = MomentumStrategy()
    fake_broker = MagicMock()
    fake_broker.submit.return_value = Order(symbol="SPY", side="buy", qty=1, status="filled")
    trader = PaperTrader(symbol="SPY", qty_per_trade=1, broker=fake_broker)
    trader.run(rising_df, strategy)
    orders2 = trader.run(rising_df, strategy)  # same data → same signal → no new order
    assert len(orders2) == 0
```

**Step 2: Run — confirm RED**

**Step 3: Implement `PaperTrader`**

```python
# src/alpha_engine/execution/paper_trader.py
from __future__ import annotations
import pandas as pd
from alpha_engine.execution.broker import Broker
from alpha_engine.execution.order import Order
from alpha_engine.strategies.base import Strategy

class PaperTrader:
    def __init__(self, symbol: str, qty_per_trade: float, broker: Broker) -> None:
        self.symbol = symbol
        self.qty_per_trade = qty_per_trade
        self._broker = broker
        self._last_signal: int = 0

    def run(self, df: pd.DataFrame, strategy: Strategy) -> list[Order]:
        signals = strategy.generate_signals(df)
        current_signal = int(signals.iloc[-1])
        submitted: list[Order] = []
        if current_signal != self._last_signal:
            if current_signal == 1:
                order = Order(symbol=self.symbol, side="buy", qty=self.qty_per_trade)
                submitted.append(self._broker.submit(order))
            elif current_signal == -1:
                order = Order(symbol=self.symbol, side="sell", qty=self.qty_per_trade)
                submitted.append(self._broker.submit(order))
            self._last_signal = current_signal
        return submitted
```

**Step 4: Run — confirm GREEN**

**Step 5: ruff + commit**

```
feat(execution): PaperTrader — signal-to-order wiring with dedup
```

---

## Task 5: Dashboard — Equity Curve page

**Files:**
- Create: `src/quant_dashboard/pages/1_Equity_Curve.py`
- Create: `src/quant_dashboard/components/charts.py`
- Test: `tests/test_dashboard/test_charts.py`

**Domain:**  
`charts.py` contains pure-Python/Plotly helper functions that return `go.Figure` objects. These are fully testable without Streamlit. The Streamlit page file calls `st.plotly_chart(fig)`.

Chart helpers:
- `equity_curve_fig(equity: pd.Series, title: str) -> go.Figure` — line chart with drawdown shading
- `returns_distribution_fig(returns: pd.Series) -> go.Figure` — histogram with VaR line

**Step 1: Write failing tests**

```python
# tests/test_dashboard/test_charts.py
import pandas as pd, numpy as np
import pytest
from quant_dashboard.components.charts import equity_curve_fig, returns_distribution_fig

@pytest.fixture
def sample_equity():
    idx = pd.date_range("2023-01-01", periods=252, freq="B")
    return pd.Series(np.cumprod(1 + np.random.default_rng(0).normal(0.0005, 0.01, 252)), index=idx)

def test_equity_curve_fig_returns_figure(sample_equity):
    import plotly.graph_objects as go
    fig = equity_curve_fig(sample_equity, title="Test")
    assert isinstance(fig, go.Figure)

def test_equity_curve_fig_has_drawdown_trace(sample_equity):
    fig = equity_curve_fig(sample_equity, title="Test")
    trace_names = [t.name for t in fig.data]
    assert any("drawdown" in (name or "").lower() for name in trace_names)

def test_returns_distribution_fig_returns_figure(sample_equity):
    import plotly.graph_objects as go
    returns = sample_equity.pct_change().dropna()
    fig = returns_distribution_fig(returns)
    assert isinstance(fig, go.Figure)
```

**Step 2: Run — confirm RED**

```powershell
pytest tests/test_dashboard/ -q
```

**Step 3: Create `tests/test_dashboard/__init__.py`** (empty)

**Step 4: Implement `charts.py`**

```python
# src/quant_dashboard/components/__init__.py  (empty)
# src/quant_dashboard/components/charts.py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def equity_curve_fig(equity: pd.Series, title: str = "Equity Curve") -> go.Figure:
    returns = equity.pct_change().fillna(0)
    rolling_max = equity.cummax()
    drawdown = (equity / rolling_max) - 1

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=equity.index, y=equity.values, name="Equity", line=dict(color="#00B4D8")), row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name="Drawdown",
                              fill="tozeroy", fillcolor="rgba(220,50,50,0.2)",
                              line=dict(color="rgba(220,50,50,0.8)")), row=2, col=1)
    fig.update_layout(title=title, template="plotly_dark", height=500,
                      legend=dict(orientation="h"), margin=dict(l=40, r=20, t=50, b=20))
    return fig

def returns_distribution_fig(returns: pd.Series, confidence: float = 0.95) -> go.Figure:
    var = float(np.percentile(returns.dropna(), (1 - confidence) * 100))
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=returns.values, nbinsx=60, name="Returns",
                                marker_color="#90E0EF", opacity=0.75))
    fig.add_vline(x=var, line_dash="dash", line_color="red",
                  annotation_text=f"VaR {confidence:.0%}: {var:.2%}")
    fig.update_layout(title="Return Distribution", template="plotly_dark",
                      xaxis_title="Daily Return", yaxis_title="Count",
                      margin=dict(l=40, r=20, t=50, b=20))
    return fig
```

**Step 5: Implement `1_Equity_Curve.py` page**

```python
# src/quant_dashboard/pages/1_Equity_Curve.py
"""Equity Curve & Drawdown analysis page."""
import streamlit as st
import pandas as pd
import numpy as np
from quant_dashboard.components.charts import equity_curve_fig, returns_distribution_fig
from alpha_engine.strategies.momentum import MomentumStrategy
from alpha_engine.strategies.mean_reversion import BollingerMeanReversionStrategy
from alpha_engine.strategies import REGISTRY
from quantcore.data.fetcher import DataFetcher

st.set_page_config(page_title="Equity Curve", layout="wide")
st.title("Equity Curve & Drawdown")

ticker = st.sidebar.text_input("Ticker", value="SPY")
period = st.sidebar.selectbox("Period", ["1y", "2y", "5y"], index=1)
strategy_name = st.sidebar.selectbox("Strategy", list(REGISTRY.keys()))

@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    return DataFetcher().fetch(ticker, period=period)

with st.spinner("Loading data…"):
    df = load_data(ticker, period)

if df.empty:
    st.error("No data returned. Check ticker symbol.")
else:
    strategy_cls = REGISTRY[strategy_name]
    result = strategy_cls().run(df)
    equity = result.equity_curve
    returns = result.returns

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{result.total_return:.1%}")
    col2.metric("Sharpe Ratio", f"{result.sharpe:.2f}")
    col3.metric("Max Drawdown", f"{result.max_drawdown:.1%}")
    col4.metric("Calmar Ratio", f"{result.calmar:.2f}")

    st.plotly_chart(equity_curve_fig(equity, title=f"{ticker} — {strategy_name}"), use_container_width=True)
    st.plotly_chart(returns_distribution_fig(returns), use_container_width=True)
```

**Step 6: Run tests — confirm GREEN**

**Step 7: ruff + commit**

```
feat(dashboard): equity curve + drawdown Plotly charts
```

---

## Task 6: Dashboard — Portfolio Allocation page

**Files:**
- Create: `src/quant_dashboard/pages/2_Portfolio.py`
- Extend: `src/quant_dashboard/components/charts.py` (add `weights_bar_fig`, `correlation_heatmap_fig`)
- Test: extend `tests/test_dashboard/test_charts.py`

**New chart helpers:**
- `weights_bar_fig(weights: pd.Series, title: str) -> go.Figure` — horizontal bar chart
- `correlation_heatmap_fig(returns_df: pd.DataFrame) -> go.Figure` — Plotly heatmap

**Step 1: Write failing tests**

```python
def test_weights_bar_fig(sample_equity):
    from quant_dashboard.components.charts import weights_bar_fig
    import plotly.graph_objects as go
    weights = pd.Series({"SPY": 0.5, "TLT": 0.3, "GLD": 0.2})
    fig = weights_bar_fig(weights, "Weights")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0

def test_correlation_heatmap_fig():
    from quant_dashboard.components.charts import correlation_heatmap_fig
    import plotly.graph_objects as go
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(0, 0.01, (100, 3)), columns=["SPY", "TLT", "GLD"])
    fig = correlation_heatmap_fig(df)
    assert isinstance(fig, go.Figure)
```

**Step 2: Implement chart helpers, then the page**

```python
# Add to charts.py:
def weights_bar_fig(weights: pd.Series, title: str = "Portfolio Weights") -> go.Figure:
    fig = go.Figure(go.Bar(x=weights.values, y=weights.index.tolist(),
                            orientation="h", marker_color="#00B4D8"))
    fig.update_layout(title=title, template="plotly_dark",
                      xaxis_tickformat=".0%", margin=dict(l=40, r=20, t=50, b=20))
    return fig

def correlation_heatmap_fig(returns_df: pd.DataFrame) -> go.Figure:
    corr = returns_df.corr()
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns.tolist(),
                                y=corr.index.tolist(), colorscale="RdBu",
                                zmid=0, text=corr.round(2).values, texttemplate="%{text}"))
    fig.update_layout(title="Return Correlation", template="plotly_dark",
                      margin=dict(l=60, r=20, t=50, b=20))
    return fig
```

```python
# src/quant_dashboard/pages/2_Portfolio.py
"""Portfolio allocation & correlation analysis page."""
import streamlit as st
import pandas as pd
from quant_dashboard.components.charts import weights_bar_fig, correlation_heatmap_fig
from alpha_engine.portfolio import EqualWeightAllocator, MeanVarianceAllocator, RiskParityAllocator, CVaRAllocator
from quantcore.data.fetcher import DataFetcher

ALLOCATORS = {
    "Equal Weight": EqualWeightAllocator(),
    "Max Sharpe": MeanVarianceAllocator(objective="max_sharpe"),
    "Min Variance": MeanVarianceAllocator(objective="min_variance"),
    "Risk Parity": RiskParityAllocator(),
    "CVaR (95%)": CVaRAllocator(alpha=0.95),
}

st.set_page_config(page_title="Portfolio", layout="wide")
st.title("Portfolio Allocation")

tickers_input = st.sidebar.text_input("Tickers (comma-separated)", "SPY,TLT,GLD,QQQ")
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
period = st.sidebar.selectbox("Period", ["1y", "2y", "5y"], index=1)
alloc_name = st.sidebar.selectbox("Allocator", list(ALLOCATORS.keys()))

@st.cache_data(ttl=3600)
def load_returns(tickers: tuple, period: str) -> pd.DataFrame:
    fetcher = DataFetcher()
    closes = {}
    for t in tickers:
        df = fetcher.fetch(t, period=period)
        if not df.empty:
            closes[t] = df["close"]
    close_df = pd.DataFrame(closes).dropna()
    return close_df.pct_change().dropna()

with st.spinner("Loading returns…"):
    returns_df = load_returns(tuple(tickers), period)

if returns_df.empty or len(returns_df.columns) < 2:
    st.error("Need at least 2 valid tickers with data.")
else:
    alloc = ALLOCATORS[alloc_name]
    weights = alloc.fit(returns_df)
    st.subheader(f"Weights — {alloc_name}")
    st.plotly_chart(weights_bar_fig(weights), use_container_width=True)
    st.subheader("Correlation Matrix")
    st.plotly_chart(correlation_heatmap_fig(returns_df), use_container_width=True)
```

**Step 3: Run — confirm GREEN**

**Step 4: ruff + commit**

```
feat(dashboard): portfolio allocation page with weights + correlation heatmap
```

---

## Task 7: Dashboard — ML Signal page

**Files:**
- Create: `src/quant_dashboard/pages/3_ML_Signal.py`
- Extend: `src/quant_dashboard/components/charts.py` (add `feature_importance_fig`)
- Test: extend `tests/test_dashboard/test_charts.py`

**New chart helper:**
- `feature_importance_fig(importance: pd.Series) -> go.Figure` — horizontal bar, sorted descending

**Step 1: Write failing tests**

```python
def test_feature_importance_fig():
    from quant_dashboard.components.charts import feature_importance_fig
    import plotly.graph_objects as go
    importance = pd.Series({"rsi": 0.4, "macd": 0.35, "bb_pct": 0.25})
    fig = feature_importance_fig(importance)
    assert isinstance(fig, go.Figure)
    # Bars should be sorted descending
    bar_trace = fig.data[0]
    assert bar_trace.x[0] >= bar_trace.x[-1]
```

**Step 2: Implement `feature_importance_fig`**

```python
def feature_importance_fig(importance: pd.Series) -> go.Figure:
    sorted_imp = importance.sort_values(ascending=True)  # ascending for horizontal bar visual
    fig = go.Figure(go.Bar(x=sorted_imp.values, y=sorted_imp.index.tolist(),
                            orientation="h", marker_color="#48CAE4"))
    fig.update_layout(title="Feature Importance (XGBoost Gain)",
                      template="plotly_dark", xaxis_tickformat=".0%",
                      margin=dict(l=120, r=20, t=50, b=20))
    return fig
```

**Step 3: Implement `3_ML_Signal.py` page**

```python
# src/quant_dashboard/pages/3_ML_Signal.py
"""ML Signal (XGBoost) analysis page."""
import streamlit as st
import pandas as pd
from quant_dashboard.components.charts import equity_curve_fig, feature_importance_fig
from alpha_ml.pipeline import MLSignalPipeline
from alpha_ml.models.xgboost_model import XGBoostModel
from quantcore.data.fetcher import DataFetcher

st.set_page_config(page_title="ML Signal", layout="wide")
st.title("XGBoost ML Signal")

ticker = st.sidebar.text_input("Ticker", value="SPY")
period = st.sidebar.selectbox("Period", ["2y", "5y"], index=1)
horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 21, 5)
train_ratio = st.sidebar.slider("Train Ratio", 0.5, 0.8, 0.6, step=0.05)

@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    return DataFetcher().fetch(ticker, period=period)

with st.spinner("Loading data…"):
    df = load_data(ticker, period)

if df.empty:
    st.error("No data returned.")
else:
    with st.spinner("Training XGBoost model…"):
        model = XGBoostModel()
        pipeline = MLSignalPipeline(model=model, train_ratio=train_ratio, horizon=horizon)
        result = pipeline.run(df)
        importance = model.feature_importance()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", f"{result.total_return:.1%}")
    col2.metric("Sharpe Ratio", f"{result.sharpe:.2f}")
    col3.metric("Max Drawdown", f"{result.max_drawdown:.1%}")

    st.plotly_chart(equity_curve_fig(result.equity_curve, title=f"{ticker} — XGBoost OOS"), use_container_width=True)
    st.plotly_chart(feature_importance_fig(importance), use_container_width=True)
```

**Step 4: Run — confirm GREEN**

**Step 5: ruff + commit**

```
feat(dashboard): ML signal page with XGBoost feature importance
```

---

## Task 8: Dashboard — main app entry point

**Files:**
- Create: `src/quant_dashboard/app.py`
- Modify: `src/quant_dashboard/__init__.py`

**Goal:** Landing page with project overview and navigation guidance.

**Step 1: Implement `app.py`**

```python
# src/quant_dashboard/app.py
"""quant-lab Streamlit dashboard — landing page."""
import streamlit as st

st.set_page_config(page_title="quant-lab", page_icon="chart_with_upward_trend", layout="wide")

st.title("quant-lab Analytics Dashboard")
st.markdown("""
**Institutional-grade quantitative research platform**

Use the sidebar to navigate between modules:

| Page | Description |
|---|---|
| Equity Curve | Backtest results, drawdown analysis |
| Portfolio | Multi-asset allocation weights + correlation |
| ML Signal | XGBoost predictions, feature importance |
""")
st.info("Start with the sidebar — select a page above.")
```

**Step 2: Add launch script to pyproject.toml**

```toml
[project.scripts]
quant-dashboard = "quant_dashboard.app:main"
```

Or use the simpler approach: document `streamlit run src/quant_dashboard/app.py` in README.

**Step 3: Update README** with dashboard launch instructions

```markdown
## Dashboard

```bash
# Install dashboard dependencies
pip install -e ".[dashboard]"

# Launch
streamlit run src/quant_dashboard/app.py
```
```

**Step 4: ruff + commit**

```
feat(dashboard): main app entry point + README launch instructions
```

---

## Task 9: Live paper trading CLI

**Files:**
- Create: `scripts/run_paper_trader.py`

**Goal:** CLI that fetches last N days of prices for a ticker, runs a strategy, and submits paper orders via Alpaca. No test required (it's a CLI script) — but add `if __name__ == "__main__"` guard.

**Step 1: Implement**

```python
#!/usr/bin/env python
"""Paper trading CLI — fetch latest OHLCV and submit orders via Alpaca paper API.

Usage:
    python scripts/run_paper_trader.py --ticker SPY --strategy momentum

Requires .env:
    ALPACA_API_KEY=...
    ALPACA_SECRET_KEY=...
    ALPACA_BASE_URL=https://paper-api.alpaca.markets
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, "src")

from dotenv import load_dotenv  # type: ignore[import-untyped]  # noqa: E402

load_dotenv()

from alpha_engine.execution.alpaca_broker import AlpacaBroker
from alpha_engine.execution.paper_trader import PaperTrader
from alpha_engine.strategies import REGISTRY
from quantcore.data.fetcher import DataFetcher


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper trading loop")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--strategy", default="momentum", choices=list(REGISTRY.keys()))
    parser.add_argument("--qty", type=float, default=1.0)
    parser.add_argument("--period", default="1y")
    args = parser.parse_args()

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    if not api_key or not secret_key:
        print("[warn] ALPACA_API_KEY or ALPACA_SECRET_KEY not set -- skipping live order submission")
        print("[info] Running signal generation only")

    print(f"[info] Fetching {args.ticker} ({args.period})...")
    df = DataFetcher().fetch(args.ticker, period=args.period)
    if df.empty:
        print(f"[error] No data for {args.ticker}")
        return

    strategy = REGISTRY[args.strategy]()
    signals = strategy.generate_signals(df)
    last_signal = int(signals.iloc[-1])
    signal_str = {1: "LONG", -1: "SHORT", 0: "FLAT"}.get(last_signal, "UNKNOWN")
    print(f"[signal] {args.ticker} current signal: {signal_str} ({last_signal})")

    if api_key and secret_key:
        broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, base_url=base_url)
        trader = PaperTrader(symbol=args.ticker, qty_per_trade=args.qty, broker=broker)
        orders = trader.run(df, strategy)
        for o in orders:
            print(f"[order] {o.side.upper()} {o.qty} {o.symbol} → status={o.status}")
        if not orders:
            print("[info] No new orders (signal unchanged)")


if __name__ == "__main__":
    main()
```

**Step 2: Verify it runs without Alpaca keys**

```powershell
python scripts/run_paper_trader.py --ticker SPY --strategy momentum
```

**Step 3: ruff + commit**

```
feat(execution): paper trading CLI script
```

---

## Task 10: Full verification + coverage + push

**Step 1: Run full suite**

```powershell
pytest -m "not integration" -q --tb=short
```

Expected: ≥175 tests passing.

**Step 2: ruff clean**

```powershell
ruff check src/ tests/ scripts/
```

**Step 3: Coverage report**

```powershell
pytest --cov=src --cov-report=term-missing -m "not integration" -q
```

Target: ≥70% overall.

**Step 4: Commit + push**

```
feat(phase4): Streamlit dashboard + Alpaca paper trading -- Phase 4 complete
```

```powershell
git push origin main
npx gitnexus analyze
```

---

## Test File Locations Summary

| File | Tests |
|---|---|
| `tests/test_alpha_engine/test_execution.py` | Order validation, Broker ABC, AlpacaBroker mock |
| `tests/test_alpha_engine/test_paper_trader.py` | Signal-to-order wiring, dedup |
| `tests/test_dashboard/test_charts.py` | Plotly figure assertions |

---

## Key Constraints

- **No live Alpaca calls in CI.** All `AlpacaBroker` tests use `unittest.mock.patch`.
- **Streamlit pages are not unit-tested** (they call `st.xxx` which requires a running server). Only pure chart functions in `components/charts.py` are tested.
- **`Order` is a plain dataclass.** Use `dataclasses.replace()` to update status — do not mutate.
- **ruff**: `tests/**` already has `S101` ignored. Add `S106` pattern if needed for secrets in tests.
- **Windows encoding**: No Unicode chars (→, ✓, σ) in CLI script output strings.
