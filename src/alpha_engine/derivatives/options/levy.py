"""Lévy Process Options Pricing (Phase 3.7).

Merton Jump-Diffusion (1976)
----------------------------
Closed-form series expansion. The stock price follows:
    dS/S = (r - q - λk̄) dt + σ dW + (J-1) dN
where N is a Poisson process with intensity λ, J = exp(Y), Y ~ N(μ_J, σ_J²),
and k̄ = E[J-1] = exp(μ_J + σ_J²/2) - 1.

Price = Σ_{n=0}^{N} w_n * BSM(S, K, T, r_n, q, σ_n)
where:
    λ' = λ * (1 + k̄)
    w_n = exp(-λ'T) * (λ'T)^n / n!
    σ_n = sqrt(σ² + n*σ_J²/T)
    r_n = r - λ*k̄ + n*ln(1+k̄)/T

Variance Gamma (Madan-Seneta 1990)
-----------------------------------
Monte Carlo simulation via subordinated Brownian motion:
    X(t) = θ*G(t) + σ*W(G(t))
where G(t) ~ Gamma(t/ν, ν) is the random time change.
The risk-neutral drift correction ω = (1/ν)*ln(1 - θ*ν - σ²*ν/2)
ensures E[exp(X(T))] = exp((r-q)*T).
    S_T = S * exp((r - q + ω)*T + X(T))
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

VALID_TYPES = {"call", "put"}
_N_TERMS = 50   # Merton series truncation


def _bsm_single(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
) -> float:
    """Single BSM price, handles degenerate σ→0 case."""
    if T <= 0:
        return max((1.0 if option_type == "call" else -1.0) * (S * math.exp(-q * T) - K * math.exp(-r * T)), 0.0)
    if sigma <= 1e-10:
        fwd = S * math.exp((r - q) * T)
        if option_type == "call":
            return math.exp(-r * T) * max(fwd - K, 0.0)
        return math.exp(-r * T) * max(K - fwd, 0.0)

    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)


def merton_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    lam: float = 1.0,
    mu_j: float = -0.1,
    sigma_j: float = 0.15,
    n_terms: int = _N_TERMS,
) -> float:
    """Price a European option under Merton (1976) jump-diffusion.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
        Standard BSM parameters.
    option_type : str
        'call' or 'put'.
    lam : float
        Poisson jump intensity (average jumps per year, default 1.0).
    mu_j : float
        Mean of log-jump size (default -0.1).
    sigma_j : float
        Volatility of log-jump size (default 0.15).
    n_terms : int
        Series truncation (default 50).

    Returns
    -------
    float
        Merton jump-diffusion option price.
    """
    if option_type not in VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    k_bar = math.exp(mu_j + 0.5 * sigma_j**2) - 1.0   # E[J-1]
    lam_prime = lam * (1.0 + k_bar)                     # risk-neutral intensity
    log1pk = math.log(1.0 + k_bar) if k_bar > -1 else mu_j

    price = 0.0
    log_lam_T = math.log(lam_prime * T) if lam_prime * T > 0 else -math.inf
    log_factorial = 0.0   # log(n!)

    for n in range(n_terms):
        if n > 0:
            log_factorial += math.log(n)

        # Poisson weight: exp(-λ'T) * (λ'T)^n / n!
        # When n=0 the (λ'T)^0 term is always 1 regardless of λ'→0
        n_log_lam_T = 0.0 if n == 0 else (n * log_lam_T if log_lam_T > -math.inf else -math.inf)
        log_w = -lam_prime * T + n_log_lam_T - log_factorial
        w = math.exp(log_w)

        if w < 1e-15:
            break

        sigma_n = math.sqrt(sigma**2 + n * sigma_j**2 / T) if T > 0 else sigma
        r_n = r - lam * k_bar + n * log1pk / T

        price += w * _bsm_single(S, K, T, r_n, q, sigma_n, option_type)

    return price


def vg_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    theta: float = -0.1,
    nu: float = 0.2,
    n_paths: int = 100_000,
    seed: int | None = None,
) -> float:
    """Price a European option under the Variance Gamma process (MC).

    Parameters
    ----------
    S, K, T, r, sigma, q : float
        Standard parameters.
    option_type : str
        'call' or 'put'.
    theta : float
        Drift of Brownian motion in VG (skewness parameter, default -0.1).
    nu : float
        Variance rate of Gamma subordinator (kurtosis parameter, default 0.2).
    n_paths : int
        Monte Carlo paths (default 100_000).
    seed : int | None
        Random seed.

    Returns
    -------
    float
        VG European option price.
    """
    if option_type not in VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    # Risk-neutral drift correction: ω = (1/ν)*ln(1 - θ*ν - σ²*ν/2)
    arg = 1.0 - theta * nu - 0.5 * sigma**2 * nu
    if arg <= 0:
        raise ValueError(
            "VG parameters violate no-arbitrage condition: "
            "1 - theta*nu - 0.5*sigma²*nu must be positive."
        )
    omega = math.log(arg) / nu

    rng = np.random.default_rng(seed)

    # Gamma subordinator G ~ Gamma(T/ν, ν)
    shape = T / nu
    scale = nu
    G = rng.gamma(shape, scale, size=n_paths)

    # Subordinated BM: X = θ*G + σ*sqrt(G)*Z
    Z = rng.standard_normal(n_paths)
    X = theta * G + sigma * np.sqrt(G) * Z

    # Terminal asset price
    S_T = S * np.exp((r - q + omega) * T + X)

    phi = 1.0 if option_type == "call" else -1.0
    payoffs = np.maximum(phi * (S_T - K), 0.0)
    return float(np.exp(-r * T) * payoffs.mean())
