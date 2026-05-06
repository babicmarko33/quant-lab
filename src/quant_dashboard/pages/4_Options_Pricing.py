"""Options Pricing Dashboard Page.

Provides an interactive UI for:
  Tab 1 — Model Comparison: BSM / Binomial / Monte Carlo / PDE prices side-by-side
  Tab 2 — Greeks: delta, gamma, vega, theta, rho sensitivity to spot
  Tab 3 — Vol Smile: SABR model smile + calibration to user-supplied quotes
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from alpha_engine.derivatives.options.binomial import binomial_price
from alpha_engine.derivatives.options.black_scholes import bsm_greeks, bsm_price
from alpha_engine.derivatives.options.monte_carlo import mc_european
from alpha_engine.derivatives.options.pde import pde_european
from alpha_engine.derivatives.volatility.sabr import sabr_calibrate, sabr_vol
from quant_dashboard.components.charts import (
    greeks_sensitivity_fig,
    model_comparison_bar_fig,
    payoff_fig,
    sabr_smile_fig,
)

st.set_page_config(page_title="Options Pricing", layout="wide")
st.title("Options Pricing Lab")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Contract Parameters")
S = st.sidebar.number_input("Spot (S)", value=100.0, min_value=0.01, step=1.0)
K = st.sidebar.number_input("Strike (K)", value=100.0, min_value=0.01, step=1.0)
T = st.sidebar.slider("Expiry T (years)", min_value=0.05, max_value=5.0, value=1.0, step=0.05)
r = st.sidebar.slider("Risk-free Rate r", min_value=0.0, max_value=0.20, value=0.05, step=0.005, format="%.3f")
sigma = st.sidebar.slider("Volatility σ", min_value=0.01, max_value=1.0, value=0.20, step=0.01, format="%.2f")
q = st.sidebar.slider("Dividend Yield q", min_value=0.0, max_value=0.15, value=0.0, step=0.005, format="%.3f")
option_type = st.sidebar.radio("Option Type", ["call", "put"])

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Model Comparison", "Greeks", "Vol Smile (SABR)"])

# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Multi-Model Price Comparison")

    with st.spinner("Pricing options across models…"):
        bsm = bsm_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
        binomial = binomial_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q,
                                  option_type=option_type, n_steps=500, american=False)
        mc, mc_se = mc_european(S=S, K=K, T=T, r=r, sigma=sigma, q=q,
                                 option_type=option_type, n_paths=100_000, seed=42)
        pde = pde_european(S=S, K=K, T=T, r=r, sigma=sigma, q=q,
                           option_type=option_type, n_s=200, n_t=200)

    model_prices = {
        "BSM (exact)": bsm,
        "Binomial CRR (500)": binomial,
        "Monte Carlo (100k)": mc,
        "PDE Crank-Nicolson": pde,
    }

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BSM", f"{bsm:.4f}")
    c2.metric("Binomial CRR", f"{binomial:.4f}", delta=f"{binomial - bsm:+.4f}")
    c3.metric("Monte Carlo", f"{mc:.4f}", delta=f"{mc - bsm:+.4f}",
              help=f"MC std error: ±{mc_se:.4f}")
    c4.metric("PDE C-N", f"{pde:.4f}", delta=f"{pde - bsm:+.4f}")

    st.plotly_chart(model_comparison_bar_fig(model_prices), use_container_width=True)

    # Payoff diagram
    st.subheader("Payoff at Expiry")
    strike_range = np.linspace(max(S * 0.4, 1.0), S * 1.6, 300)
    phi = 1.0 if option_type == "call" else -1.0
    payoffs = np.maximum(phi * (strike_range - K), 0.0) - bsm
    st.plotly_chart(
        payoff_fig(strike_range, payoffs, spot=S,
                   title=f"{option_type.capitalize()} Payoff (net of premium)"),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — Greeks
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Greeks")

    greeks = bsm_greeks(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)

    gc1, gc2, gc3, gc4, gc5 = st.columns(5)
    gc1.metric("Delta (Δ)", f"{greeks['delta']:.4f}")
    gc2.metric("Gamma (Γ)", f"{greeks['gamma']:.4f}")
    gc3.metric("Vega (ν)", f"{greeks['vega']:.4f}", help="Per 1% move in σ")
    gc4.metric("Theta (Θ)", f"{greeks['theta']:.4f}", help="Per calendar day")
    gc5.metric("Rho (ρ)", f"{greeks['rho']:.4f}", help="Per 1% move in r")

    st.subheader("Greeks vs Spot")
    spot_range = np.linspace(max(S * 0.5, 1.0), S * 1.5, 200)
    with st.spinner("Computing sensitivities…"):
        greek_curves: dict[str, np.ndarray] = {"Delta": [], "Gamma": [], "Vega": []}  # type: ignore[assignment]
        for s_val in spot_range:
            g = bsm_greeks(S=float(s_val), K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
            greek_curves["Delta"].append(g["delta"])
            greek_curves["Gamma"].append(g["gamma"])
            greek_curves["Vega"].append(g["vega"] / 100)  # normalise to per-1% move

    for key in greek_curves:
        greek_curves[key] = np.array(greek_curves[key])  # type: ignore[assignment]

    st.plotly_chart(
        greeks_sensitivity_fig(spot_range, greek_curves, param_label="Spot Price",
                               title="Δ, Γ, ν vs Spot"),
        use_container_width=True,
    )

    st.subheader("Greeks vs Volatility")
    vol_range = np.linspace(0.05, 0.80, 200)
    with st.spinner("Computing vol sensitivities…"):
        vol_curves: dict[str, np.ndarray] = {"Delta": [], "Gamma": [], "Vega": []}  # type: ignore[assignment]
        for v_val in vol_range:
            g = bsm_greeks(S=S, K=K, T=T, r=r, sigma=float(v_val), q=q, option_type=option_type)
            vol_curves["Delta"].append(g["delta"])
            vol_curves["Gamma"].append(g["gamma"])
            vol_curves["Vega"].append(g["vega"] / 100)

    for key in vol_curves:
        vol_curves[key] = np.array(vol_curves[key])  # type: ignore[assignment]

    st.plotly_chart(
        greeks_sensitivity_fig(vol_range * 100, vol_curves, param_label="Volatility (%)",
                               title="Δ, Γ, ν vs Implied Volatility"),
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — SABR Vol Smile
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("SABR Volatility Smile")

    st.markdown("**SABR Parameters** — enter manually or calibrate to market quotes below.")
    col_a, col_b, col_c, col_d = st.columns(4)
    sabr_alpha = col_a.number_input("α (initial vol)", value=float(sigma), min_value=0.001, step=0.01)
    sabr_beta = col_b.slider("β (CEV)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    sabr_rho = col_c.slider("ρ (correlation)", min_value=-0.99, max_value=0.99, value=-0.30, step=0.01)
    sabr_nu = col_d.number_input("ν (vol-of-vol)", value=0.40, min_value=0.001, step=0.01)

    F = S  # use spot as forward (simplified: q=0 or near-date)
    smile_strikes = np.linspace(max(F * 0.7, 1.0), F * 1.3, 60)

    model_vols = np.array([
        sabr_vol(F=F, K=float(k), T=T, alpha=sabr_alpha, beta=sabr_beta,
                 rho=sabr_rho, nu=sabr_nu)
        for k in smile_strikes
    ])

    # Market quotes input (optional)
    st.markdown("**Optional: paste market quotes to calibrate SABR**")
    market_input = st.text_area(
        "Market quotes — one per line: `strike,implied_vol`  (e.g. `90,0.22`)",
        value="",
        height=120,
    )

    market_strikes_cal: np.ndarray | None = None
    market_vols_cal: np.ndarray | None = None
    if market_input.strip():
        try:
            rows = [line.split(",") for line in market_input.strip().splitlines() if "," in line]
            market_strikes_cal = np.array([float(r[0]) for r in rows])
            market_vols_cal = np.array([float(r[1]) for r in rows])

            with st.spinner("Calibrating SABR…"):
                alpha_fit, rho_fit, nu_fit = sabr_calibrate(
                    strikes=market_strikes_cal,
                    market_vols=market_vols_cal,
                    F=F, T=T, beta=sabr_beta,
                )
            st.success(f"Calibrated: α={alpha_fit:.4f}, ρ={rho_fit:.4f}, ν={nu_fit:.4f}")
            model_vols = np.array([
                sabr_vol(F=F, K=float(k), T=T, alpha=alpha_fit, beta=sabr_beta,
                         rho=rho_fit, nu=nu_fit)
                for k in smile_strikes
            ])
            # Align market vols onto the same x-axis for plotting
            mkt_dense = np.array([
                sabr_vol(F=F, K=float(k), T=T, alpha=alpha_fit, beta=sabr_beta,
                         rho=rho_fit, nu=nu_fit)
                for k in market_strikes_cal
            ])
            mkt_dense = market_vols_cal  # raw market quotes for scatter
        except Exception as exc:
            st.error(f"Could not parse market quotes: {exc}")

    st.plotly_chart(
        sabr_smile_fig(
            strikes=smile_strikes,
            market_vols=market_vols_cal if market_strikes_cal is not None else None,
            model_vols=model_vols,
            title=f"SABR Smile  |  F={F:.0f}, T={T:.2f}y, β={sabr_beta:.2f}",
        ),
        use_container_width=True,
    )

    with st.expander("SABR formula parameters"):
        st.json({
            "F": F, "T": T,
            "alpha": sabr_alpha, "beta": sabr_beta,
            "rho": sabr_rho, "nu": sabr_nu,
        })
