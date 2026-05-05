"""XGBoost ML Signal analysis page."""

from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from alpha_ml.models.xgboost_model import XGBoostModel
from alpha_ml.pipeline import MLSignalPipeline
from quant_dashboard.components.charts import equity_curve_fig, feature_importance_fig
from quantcore.data.fetcher import fetch_ohlcv

st.set_page_config(page_title="ML Signal", layout="wide")
st.title("XGBoost ML Signal")

ticker = st.sidebar.text_input("Ticker", value="SPY")
period = st.sidebar.selectbox("Period", ["2y", "5y"], index=1)
horizon = st.sidebar.slider("Forecast Horizon (days)", 1, 21, 5)
train_ratio = st.sidebar.slider("Train Ratio", 0.5, 0.8, 0.6, step=0.05)


@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str):  # noqa: ANN201
    years = int(period.rstrip("y"))
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    return fetch_ohlcv(ticker, start=start)


with st.spinner("Loading data..."):
    df = load_data(ticker, period)

if df.empty:
    st.error("No data returned.")
else:
    with st.spinner("Training XGBoost model... (this may take ~10s)"):
        model = XGBoostModel()
        pipeline = MLSignalPipeline(model=model, train_ratio=train_ratio, horizon=horizon)
        result = pipeline.run(df)
        importance = model.feature_importance()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Return", f"{result.total_return:.1%}")
    col2.metric("Sharpe Ratio", f"{result.sharpe:.2f}")
    col3.metric("Max Drawdown", f"{result.max_drawdown:.1%}")

    st.plotly_chart(
        equity_curve_fig(result.equity_curve, title=f"{ticker} -- XGBoost OOS"),
        use_container_width=True,
    )
    st.plotly_chart(feature_importance_fig(importance), use_container_width=True)
