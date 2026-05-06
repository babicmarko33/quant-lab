"""Market Data Dashboard Page.

Provides an interactive UI for:
  Tab 1 — Price History: OHLCV candlestick + volume bar
  Tab 2 — Returns Analysis: distribution + rolling annualised vol
  Tab 3 — Vol Surface: synthetic IV surface from SABR across strikes/expiries
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from alpha_engine.derivatives.volatility.sabr import sabr_vol
from quant_dashboard.components.charts import (
    ohlcv_fig,
    returns_distribution_fig,
    rolling_vol_fig,
    sabr_smile_fig,
    vol_surface_heatmap_fig,
)
from quantcore.data.fetcher import fetch_ohlcv

st.set_page_config(page_title="Market Data", layout="wide")
st.title("Market Data Explorer")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Data Settings")
ticker = st.sidebar.text_input("Ticker", value="SPY").upper().strip()
period_choice = st.sidebar.selectbox("Period", ["6m", "1y", "2y", "5y"], index=1)
_years_map = {"6m": 0.5, "1y": 1, "2y": 2, "5y": 5}
start_date = (date.today() - timedelta(days=int(365 * _years_map[period_choice]))).isoformat()


@st.cache_data(ttl=3600)
def _load(ticker: str, start: str) -> pd.DataFrame:
    return fetch_ohlcv(ticker, start=start)


with st.spinner(f"Loading {ticker}…"):
    df = _load(ticker, start_date)

if df.empty:
    st.error(f"No data returned for **{ticker}**. Try a different ticker or period.")
    st.stop()

returns = df["close"].pct_change().dropna()

# ── Summary metrics ───────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Last Close", f"{df['close'].iloc[-1]:.2f}")
m2.metric("Period Return", f"{(df['close'].iloc[-1] / df['close'].iloc[0] - 1):.1%}")
m3.metric("Ann. Vol (1y)", f"{returns.tail(252).std() * np.sqrt(252):.1%}")
m4.metric("Max Drawdown",
          f"{((df['close'] / df['close'].cummax()) - 1).min():.1%}")
m5.metric("Bars", f"{len(df):,}")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Price History", "Returns Analysis", "Vol Surface"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Price History
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader(f"{ticker} — OHLCV")
    st.plotly_chart(ohlcv_fig(df, title=f"{ticker} — {period_choice}"), use_container_width=True)

    with st.expander("Raw Data"):
        st.dataframe(
            df.tail(100).sort_index(ascending=False).style.format({
                "open": "{:.2f}", "high": "{:.2f}", "low": "{:.2f}",
                "close": "{:.2f}", "volume": "{:,.0f}",
            }),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Returns Analysis
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Return Distribution")
    col_l, col_r = st.columns(2)
    confidence = col_l.slider("VaR Confidence", min_value=0.90, max_value=0.99,
                               value=0.95, step=0.01, format="%.2f")
    with col_l:
        st.plotly_chart(returns_distribution_fig(returns, confidence=confidence),
                        use_container_width=True)

    with col_r:
        windows_str = st.multiselect(
            "Rolling Windows (days)", options=[10, 20, 30, 60, 90, 120],
            default=[20, 60, 120],
        )
        windows = windows_str if windows_str else [20, 60, 120]
        st.plotly_chart(rolling_vol_fig(returns, windows=windows), use_container_width=True)

    st.subheader("Descriptive Statistics")
    stats = pd.DataFrame({
        "Mean (ann.)": [returns.mean() * 252],
        "Vol (ann.)": [returns.std() * np.sqrt(252)],
        "Skewness": [float(returns.skew())],
        "Kurtosis": [float(returns.kurtosis())],
        "Min": [float(returns.min())],
        "Max": [float(returns.max())],
    }, index=[ticker])
    st.dataframe(stats.style.format("{:.4f}"))


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Vol Surface (SABR synthetic)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Implied Volatility Surface — SABR Model")
    st.markdown(
        "Constructs a synthetic IV surface across strikes and expiries using the "
        "SABR stochastic-vol model. The ATM vol is estimated from recent realised vol; "
        "all other parameters can be tuned in the sidebar below."
    )

    spot = float(df["close"].iloc[-1])
    atm_vol_est = float(returns.tail(252).std() * np.sqrt(252))

    st.sidebar.markdown("---")
    st.sidebar.header("SABR Surface Parameters")
    sabr_alpha = st.sidebar.number_input("α (ATM vol seed)", value=round(atm_vol_est, 3),
                                          min_value=0.01, step=0.01)
    sabr_beta = st.sidebar.slider("β (CEV)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    sabr_rho = st.sidebar.slider("ρ", min_value=-0.99, max_value=0.99, value=-0.30, step=0.01)
    sabr_nu = st.sidebar.number_input("ν (vol-of-vol)", value=0.40, min_value=0.01, step=0.01)

    # Strike grid: ±40% around spot in 13 steps
    strikes = np.linspace(spot * 0.60, spot * 1.40, 13)
    # Expiry grid: 1M to 2Y
    expiries = np.array([1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5, 2.0])

    with st.spinner("Building SABR surface…"):
        iv_matrix = np.zeros((len(expiries), len(strikes)))
        for i, T in enumerate(expiries):
            F = spot  # simplified: F ≈ S for short maturities
            for j, K in enumerate(strikes):
                iv_matrix[i, j] = sabr_vol(
                    F=F, K=float(K), T=float(T),
                    alpha=sabr_alpha, beta=sabr_beta,
                    rho=sabr_rho, nu=sabr_nu,
                )

    col_surf, col_smile = st.columns([1.4, 1])
    with col_surf:
        st.plotly_chart(
            vol_surface_heatmap_fig(strikes, expiries, iv_matrix,
                                    title=f"{ticker} — SABR IV Surface"),
            use_container_width=True,
        )

    with col_smile:
        expiry_labels = [f"{t*12:.0f}M" for t in expiries]
        chosen_exp_label = st.selectbox("Smile slice — expiry", expiry_labels, index=3)
        chosen_idx = expiry_labels.index(chosen_exp_label)
        chosen_expiry = expiries[chosen_idx]  # noqa: F841

        smile_vols = iv_matrix[chosen_idx]
        st.plotly_chart(
            sabr_smile_fig(
                strikes=strikes,
                market_vols=None,
                model_vols=smile_vols,
                title=f"SABR Smile at T={chosen_exp_label}",
            ),
            use_container_width=True,
        )

    with st.expander("Surface data table"):
        surf_df = pd.DataFrame(
            iv_matrix * 100,
            index=[f"{t*12:.0f}M" for t in expiries],
            columns=[f"{k:.0f}" for k in strikes],
        )
        st.dataframe(surf_df.style.format("{:.1f}%").background_gradient(cmap="viridis"))
