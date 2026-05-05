"""Portfolio allocation & correlation analysis page."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from alpha_engine.portfolio import CVaRAllocator, EqualWeightAllocator, MeanVarianceAllocator, RiskParityAllocator
from quant_dashboard.components.charts import correlation_heatmap_fig, weights_bar_fig
from quantcore.data.fetcher import fetch_ohlcv

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
def load_returns(tickers: tuple, period: str):  # noqa: ANN201
    years = int(period.rstrip("y"))
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    closes = {}
    for t in tickers:
        df = fetch_ohlcv(t, start=start)
        if not df.empty:
            closes[t] = df["close"]
    import pandas as pd

    close_df = pd.DataFrame(closes).dropna()
    return close_df.pct_change().dropna()


with st.spinner("Loading returns..."):
    returns_df = load_returns(tuple(tickers), period)

if returns_df.empty or len(returns_df.columns) < 2:
    st.error("Need at least 2 valid tickers with data.")
else:
    alloc = ALLOCATORS[alloc_name]
    weights = alloc.fit(returns_df)

    st.subheader(f"Weights -- {alloc_name}")
    col_w, col_c = st.columns(2)
    with col_w:
        st.plotly_chart(weights_bar_fig(weights), use_container_width=True)
    with col_c:
        st.plotly_chart(correlation_heatmap_fig(returns_df), use_container_width=True)

    st.subheader("Weight Table")
    st.dataframe(weights.to_frame("weight").style.format("{:.2%}"))
