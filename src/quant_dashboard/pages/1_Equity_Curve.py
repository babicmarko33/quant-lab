"""Equity Curve & Drawdown analysis page."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from alpha_engine.strategies import REGISTRY
from quant_dashboard.components.charts import equity_curve_fig, returns_distribution_fig
from quantcore.data.fetcher import fetch_ohlcv

st.set_page_config(page_title="Equity Curve", layout="wide")
st.title("Equity Curve & Drawdown")

# Sidebar controls
ticker = st.sidebar.text_input("Ticker", value="SPY")
period = st.sidebar.selectbox("Period", ["1y", "2y", "5y"], index=1)
strategy_name = st.sidebar.selectbox("Strategy", list(REGISTRY.keys()))


@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str):  # noqa: ANN201
    years = int(period.rstrip("y"))
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    return fetch_ohlcv(ticker, start=start)


with st.spinner("Loading data..."):
    df = load_data(ticker, period)

if df.empty:
    st.error("No data returned. Check the ticker symbol.")
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

    st.plotly_chart(
        equity_curve_fig(equity, title=f"{ticker} -- {strategy_name}"),
        use_container_width=True,
    )
    st.plotly_chart(returns_distribution_fig(returns), use_container_width=True)
