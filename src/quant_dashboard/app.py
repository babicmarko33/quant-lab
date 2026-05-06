"""quant-lab Streamlit dashboard -- landing page."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="quant-lab",
    page_icon=":chart_with_upward_trend:",
    layout="wide",
)

st.title("quant-lab Analytics Dashboard")
st.markdown(
    """
**Institutional-grade quantitative research platform**

Use the sidebar to navigate between modules:

| Page | Description |
|---|---|
| Equity Curve | Backtest results, drawdown analysis |
| Portfolio | Multi-asset allocation weights + correlation |
| ML Signal | XGBoost predictions, feature importance |
| Options Pricing | BSM / Binomial / MC / PDE prices + Greeks + SABR smile |
| Market Data | Live OHLCV, return analytics, SABR vol surface |
"""
)
st.info("Select a page from the sidebar to get started.")

st.markdown("---")
st.subheader("Quick Start")
st.code(
    "# Launch this dashboard\nstreamlit run src/quant_dashboard/app.py",
    language="bash",
)
