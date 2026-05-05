from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from alpha_engine.derivatives.options.black_scholes import bsm_greeks, bsm_price


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Compute implied volatility using Newton-Raphson with Brent's method fallback.

    Parameters
    ----------
    market_price : float
        Observed market price of the option.
    S, K, T, r : float
        Spot price, strike, time-to-expiry (years), risk-free rate.
    option_type : str
        'call' or 'put'.
    q : float
        Continuous dividend yield.
    tol : float
        Convergence tolerance.
    max_iter : int
        Maximum Newton-Raphson iterations before Brent fallback.

    Returns
    -------
    float
        Implied volatility (annualised).

    Raises
    ------
    ValueError
        If market_price is at or below intrinsic value (no valid IV exists).
    """
    intrinsic: float
    if option_type == "call":
        intrinsic = max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
    else:
        intrinsic = max(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))

    if market_price <= intrinsic + 1e-10:
        raise ValueError(
            f"market_price {market_price:.6f} is below intrinsic {intrinsic:.6f}; "
            "no valid implied volatility exists."
        )

    # Newton-Raphson
    sigma = 0.20
    for _ in range(max_iter):
        price = bsm_price(S, K, T, r, sigma, option_type=option_type, q=q)
        # vega from bsm_greeks is per 1% vol, convert back to per unit
        vega_pct = bsm_greeks(S, K, T, r, sigma, option_type=option_type, q=q)["vega"]
        vega = vega_pct * 100  # per unit vol
        if abs(vega) < 1e-12:
            break
        step = (price - market_price) / vega
        sigma -= step
        sigma = max(1e-8, min(sigma, 10.0))
        if abs(price - market_price) < tol:
            return float(sigma)

    # Brent's method fallback (guaranteed convergence on bracket)
    def objective(s: float) -> float:
        return bsm_price(S, K, T, r, s, option_type=option_type, q=q) - market_price

    lo, hi = 1e-6, 10.0
    f_lo = objective(lo)
    f_hi = objective(hi)
    if f_lo * f_hi > 0:
        raise ValueError(
            "Could not bracket implied volatility. "
            "Market price may be outside the BSM model range."
        )
    return float(brentq(objective, lo, hi, xtol=tol, maxiter=500))
