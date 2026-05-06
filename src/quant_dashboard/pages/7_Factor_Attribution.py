"""Factor Attribution dashboard page.

Fetches OHLCV data, downloads Fama-French 3-factor data, runs OLS attribution
via FactorModel, optionally splits by HMM regime via RegimeFactorModel, and
displays alpha, betas, R², t-stats, and residual plots.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alpha_engine.factor.factor_model import FactorModel
from alpha_engine.factor.regime_factor_model import RegimeFactorModel
from alpha_engine.regime.hmm_classifier import RegimeClassifier
from quantcore.data.fama_french import FamaFrenchFetcher
from quantcore.data.fetcher import fetch_ohlcv

st.set_page_config(page_title="Factor Attribution", layout="wide")
st.title("Fama-French Factor Attribution")

# ── Sidebar ───────────────────────────────────────────────────────────────────
ticker = st.sidebar.text_input("Ticker / Portfolio", value="SPY")
period = st.sidebar.selectbox("History", ["3y", "5y", "10y"], index=1)
regime_split = st.sidebar.toggle("Split by HMM Regime", value=False)
n_regimes = st.sidebar.slider("Number of Regimes", 2, 4, 2, disabled=not regime_split)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_price_returns(ticker: str, period: str) -> pd.Series:
    years = int(period.rstrip("y"))
    start = (date.today() - timedelta(days=365 * years)).isoformat()
    df = fetch_ohlcv(ticker, start=start)
    return df["close"].resample("ME").last().pct_change().dropna()


@st.cache_data(ttl=86400)
def load_ff3() -> pd.DataFrame:
    return FamaFrenchFetcher().fetch_3_factor()


with st.spinner("Fetching price data…"):
    try:
        monthly_returns = load_price_returns(ticker, period)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load price data: {exc}")
        st.stop()

with st.spinner("Fetching Fama-French factors…"):
    try:
        ff3 = load_ff3()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to load Fama-French data: {exc}")
        st.stop()

# Align
common = monthly_returns.index.intersection(ff3.index)
if len(common) < 12:
    st.warning("Less than 12 months of overlapping data. Try a longer period.")
    st.stop()

returns_aligned = monthly_returns.loc[common]
ff_aligned = ff3.loc[common]

# ── Overall Factor Model ──────────────────────────────────────────────────────
model = FactorModel()
result = model.fit(returns_aligned, ff_aligned.copy())

st.subheader(f"Overall Attribution — {ticker}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Annualised Alpha", f"{result.alpha:.2%}")
col2.metric("Market Beta", f"{result.betas.get('Mkt_RF', float('nan')):.3f}")
col3.metric("R²", f"{result.r_squared:.3f}")
col4.metric("Observations", len(returns_aligned))

# Betas table
betas_df = pd.DataFrame(
    {"Factor": list(result.betas.keys()), "Beta": list(result.betas.values()),
     "t-stat": [result.t_stats.get(f, float("nan")) for f in result.betas]}
)
st.dataframe(betas_df.style.format({"Beta": "{:.4f}", "t-stat": "{:.2f}"}), use_container_width=True)

# Residual plot
fig_resid = go.Figure()
fig_resid.add_trace(go.Scatter(
    x=list(range(len(result.residuals))),
    y=result.residuals,
    mode="lines",
    name="Residuals",
    line={"color": "royalblue", "width": 1},
))
fig_resid.add_hline(y=0, line_dash="dash", line_color="gray")
fig_resid.update_layout(
    title="OLS Residuals (excess return unexplained by factors)",
    xaxis_title="Month",
    yaxis_title="Residual",
    height=280,
    margin={"t": 40, "b": 30},
)
st.plotly_chart(fig_resid, use_container_width=True)

# ── Regime-split attribution ──────────────────────────────────────────────────
if regime_split:
    st.divider()
    st.subheader(f"Regime-Split Attribution ({n_regimes} regimes)")

    with st.spinner("Fitting HMM and running per-regime OLS…"):
        # Use daily returns for HMM then map to monthly regime
        daily_ret = returns_aligned  # fallback: use monthly if no daily available
        clf = RegimeClassifier(n_regimes=n_regimes)
        clf.fit(returns_aligned)
        regime_labels = clf.predict(returns_aligned)

        rfm = RegimeFactorModel()
        regime_results = rfm.fit(returns_aligned, ff_aligned.copy(), regime_labels)
        summary_df = RegimeFactorModel.summary(regime_results)

    st.dataframe(summary_df.style.format("{:.4f}"), use_container_width=True)

    # Alpha comparison bar chart
    fig_alpha = go.Figure()
    regimes_list = sorted(regime_results.keys())
    alphas = [regime_results[r].alpha for r in regimes_list]
    colors = ["green" if a > 0 else "crimson" for a in alphas]
    fig_alpha.add_trace(go.Bar(
        x=[f"Regime {r}" for r in regimes_list],
        y=alphas,
        marker_color=colors,
        name="Alpha",
    ))
    fig_alpha.update_layout(
        title="Annualised Alpha by Regime",
        yaxis_title="Alpha",
        height=300,
        margin={"t": 40, "b": 30},
    )
    st.plotly_chart(fig_alpha, use_container_width=True)

    # Market beta by regime
    fig_beta = go.Figure()
    mkt_betas = [regime_results[r].betas.get("Mkt_RF", 0.0) for r in regimes_list]
    fig_beta.add_trace(go.Bar(
        x=[f"Regime {r}" for r in regimes_list],
        y=mkt_betas,
        marker_color="steelblue",
        name="Market Beta",
    ))
    fig_beta.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="β=1")
    fig_beta.update_layout(
        title="Market Beta (Mkt-RF) by Regime",
        yaxis_title="Beta",
        height=300,
        margin={"t": 40, "b": 30},
    )
    st.plotly_chart(fig_beta, use_container_width=True)
