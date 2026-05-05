from __future__ import annotations

import numpy as np
from scipy.stats import norm

VALID_TYPES = {"call", "put"}


def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> float:
    """Black-Scholes-Merton option price with continuous dividend yield q."""
    if option_type not in VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if option_type == "call":
        return float(
            S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        )
    return float(
        K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    )


def bsm_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
) -> dict[str, float]:
    """First-order Greeks + Gamma for a BSM option.

    Returns a dict with keys: delta, gamma, theta, vega, rho.
    - theta is per calendar day (divided by 252).
    - vega is per 1% change in implied vol.
    - rho is per 1% change in interest rate.
    """
    if option_type not in VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    n_d1 = norm.pdf(d1)
    exp_qT = np.exp(-q * T)
    exp_rT = np.exp(-r * T)

    gamma = float(exp_qT * n_d1 / (S * sigma * sqrt_T))
    vega = float(S * exp_qT * n_d1 * sqrt_T / 100)  # per 1% vol

    if option_type == "call":
        delta = float(exp_qT * norm.cdf(d1))
        theta = float(
            (
                -S * n_d1 * sigma * exp_qT / (2 * sqrt_T)
                - r * K * exp_rT * norm.cdf(d2)
                + q * S * exp_qT * norm.cdf(d1)
            )
            / 252
        )
        rho = float(K * T * exp_rT * norm.cdf(d2) / 100)
    else:
        delta = float(exp_qT * (norm.cdf(d1) - 1))
        theta = float(
            (
                -S * n_d1 * sigma * exp_qT / (2 * sqrt_T)
                + r * K * exp_rT * norm.cdf(-d2)
                - q * S * exp_qT * norm.cdf(-d1)
            )
            / 252
        )
        rho = float(-K * T * exp_rT * norm.cdf(-d2) / 100)

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}
