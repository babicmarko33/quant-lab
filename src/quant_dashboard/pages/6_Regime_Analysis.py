"""HMM Regime Analysis dashboard page."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from alpha_engine.regime.hmm_classifier import RegimeClassifier
from alpha_engine.regime.regime_filtered_strategy import RegimeFilteredStrategy
from alpha_engine.strategies.moving_average import MovingAverageCrossStrategy
from quant_dashboard.components.charts import (
    equity_curve_fig,
    regime_timeline_fig,
    regime_vol_bar_fig,
)
from quantcore.data.fetcher import fetch_ohlcv

st.set_page_config(page_title="Regime Analysis", layout="wide")
st.title("HMM Regime Analysis")

# ── Sidebar controls ──────────────────────────────────────────────────────────
ticker = st.sidebar.text_input("Ticker", value="SPY")
period = st.sidebar.selectbox("Period", ["2y", "5y"], index=1)
n_regimes = st.sidebar.slider("Number of Regimes", 2, 4, 2)
active_regime = st.sidebar.slider("Active Regime (for filtered strategy)", 0, 3, 0)


@st.cache_data(ttl=3600)
def load_data(ticker: str, period: str) -> pd.DataFrame:  # noqa: ANN201
    years = int(period.rstrip("y"))
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    return fetch_ohlcv(ticker, start=start)


with st.spinner("Loading data..."):
    df = load_data(ticker, period)

if df.empty:
    st.error("No data returned.")
    st.stop()

returns = df["close"].pct_change().fillna(0.0)

with st.spinner("Fitting HMM classifier..."):
    clf = RegimeClassifier(n_regimes=n_regimes)
    clf.fit(returns)
    regimes = clf.predict(returns)
    proba = clf.predict_proba(returns)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Regime Timeline", "Regime Statistics", "Filtered Backtest"])

# ── Tab 1: Timeline ───────────────────────────────────────────────────────────
with tab1:
    st.plotly_chart(
        regime_timeline_fig(df["close"], regimes, n_regimes=n_regimes, title=f"{ticker} — Regime Timeline"),
        use_container_width=True,
    )

    # Posterior probabilities heatmap
    st.subheader("Posterior Probabilities")
    proba_df = pd.DataFrame(
        proba,
        index=df.index,
        columns=[f"P(Regime {r})" for r in range(n_regimes)],
    )
    st.line_chart(proba_df, use_container_width=True)

# ── Tab 2: Statistics ─────────────────────────────────────────────────────────
with tab2:
    rows = []
    for r in range(n_regimes):
        mask = regimes == r
        r_ret = returns[mask]
        rows.append({
            "regime": r,
            "count": int(mask.sum()),
            "pct_time": float(mask.mean()),
            "mean_ret": float(r_ret.mean()),
            "ann_vol": float(r_ret.std() * np.sqrt(252)),
            "sharpe": float(r_ret.mean() / r_ret.std() * np.sqrt(252)) if r_ret.std() > 0 else 0.0,
        })
    stats_df = pd.DataFrame(rows)

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            stats_df.style.format({
                "pct_time": "{:.1%}",
                "mean_ret": "{:.4f}",
                "ann_vol": "{:.2%}",
                "sharpe": "{:.2f}",
            }),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            regime_vol_bar_fig(stats_df, title=f"{ticker} — Regime Volatility"),
            use_container_width=True,
        )

# ── Tab 3: Filtered Backtest ──────────────────────────────────────────────────
with tab3:
    st.info(
        f"Comparing unfiltered MA cross vs regime-filtered (active=Regime {active_regime}) on {ticker}."
    )
    active_regime_clamped = min(active_regime, n_regimes - 1)

    with st.spinner("Running backtests..."):
        base_strat = MovingAverageCrossStrategy()
        base_result = base_strat.run(df)

        filtered_strat = RegimeFilteredStrategy(
            inner=MovingAverageCrossStrategy(),
            active_regime=active_regime_clamped,
            n_regimes=n_regimes,
        )
        filtered_result = filtered_strat.run(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Unfiltered MA Cross")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Return", f"{base_result.total_return:.1%}")
        m2.metric("Sharpe", f"{base_result.sharpe:.2f}")
        m3.metric("Max DD", f"{base_result.max_drawdown:.1%}")
        st.plotly_chart(
            equity_curve_fig(base_result.equity_curve, title="Unfiltered"),
            use_container_width=True,
        )
    with col2:
        st.subheader(f"Regime-Filtered (active={active_regime_clamped})")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Return", f"{filtered_result.total_return:.1%}")
        m2.metric("Sharpe", f"{filtered_result.sharpe:.2f}")
        m3.metric("Max DD", f"{filtered_result.max_drawdown:.1%}")
        st.plotly_chart(
            equity_curve_fig(filtered_result.equity_curve, title="Regime-Filtered"),
            use_container_width=True,
        )
