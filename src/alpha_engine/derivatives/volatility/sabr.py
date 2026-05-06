"""SABR Stochastic Volatility Model (Phase 3.8).

Hagan, Kumar, Lesniewski, Woodward (2002) — "Managing Smile Risk"

The SABR model:
    dF = σ * F^β * dW₁
    dσ = ν * σ * dW₂
    ⟨dW₁, dW₂⟩ = ρ dt

where F is the forward price, σ (alpha) the initial vol, β the CEV exponent,
ρ the correlation, and ν (nu) the vol-of-vol.

Hagan et al. derived an approximate closed-form implied Black vol:

    σ_B(F, K) = (α / (FK)^((1-β)/2) / D(z)) * [1 + correction terms] * T

where:
    z = (ν/α) * (FK)^((1-β)/2) * ln(F/K)
    χ(z) = ln[(√(1-2ρz+z²) + z - ρ) / (1-ρ)]
    D(z) = z / χ(z)   (= 1 at ATM)
    correction = [(1-β)²/24 * α²/(FK)^(1-β) + ρβνα/(4*(FK)^((1-β)/2)) + (2-3ρ²)/24 * ν²] * T

Calibration: minimise RMSE between SABR vols and market vols over (α, ρ, ν).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize


def sabr_vol(
    F: float,
    K: float,
    T: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Compute SABR implied Black volatility using Hagan et al. (2002) formula.

    Parameters
    ----------
    F : float
        Forward price.
    K : float
        Strike.
    T : float
        Time to expiry (years).
    alpha : float
        Initial volatility (σ in the SABR equations).
    beta : float
        CEV exponent ∈ [0, 1]. β=0: normal, β=1: log-normal.
    rho : float
        Correlation ∈ (-1, 1).
    nu : float
        Vol-of-vol (ν > 0).

    Returns
    -------
    float
        SABR implied Black volatility.
    """
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    if not (-1.0 < rho < 1.0):
        raise ValueError(f"rho must be in (-1, 1), got {rho}")

    if F <= 0 or K <= 0:
        raise ValueError("F and K must be positive")

    one_minus_beta = 1.0 - beta
    FK = F * K
    FK_mid = FK ** (one_minus_beta / 2.0)

    # Correction term (shared between ATM and non-ATM branches)
    term1 = one_minus_beta**2 / 24.0 * alpha**2 / (FK ** one_minus_beta)
    term2 = rho * beta * nu * alpha / (4.0 * FK_mid)
    term3 = (2.0 - 3.0 * rho**2) / 24.0 * nu**2
    correction = 1.0 + (term1 + term2 + term3) * T

    # ATM case (F ≈ K): use limiting form directly
    if abs(F - K) < 1e-8 * F:
        return float((alpha / (F**one_minus_beta)) * correction)

    log_FK = math.log(F / K)

    # Log-expansion denominator: 1 + (1-β)²/24 * ln²(F/K) + (1-β)⁴/1920 * ln⁴(F/K)
    log_expansion = 1.0 + (one_minus_beta**2 / 24.0) * log_FK**2 + (one_minus_beta**4 / 1920.0) * log_FK**4

    # z/χ(z) skew factor — z/χ(z) is in the NUMERATOR (amplifies smile)
    z = (nu / alpha) * FK_mid * log_FK
    sqrt_term = math.sqrt(max(1.0 - 2.0 * rho * z + z * z, 0.0))
    denom_chi = sqrt_term + z - rho
    if denom_chi <= 0:
        denom_chi = 1e-14
    chi_z = math.log(denom_chi / (1.0 - rho))
    skew = z / chi_z if abs(chi_z) > 1e-14 else 1.0

    # Full formula: σ_B = [α / ((FK)^((1-β)/2) * log_expansion)] * skew * correction
    prefactor = alpha / (FK_mid * log_expansion)

    return float(prefactor * skew * correction)


def sabr_calibrate(
    strikes: np.ndarray,
    market_vols: np.ndarray,
    F: float,
    T: float,
    beta: float = 0.5,
) -> tuple[float, float, float]:
    """Calibrate SABR parameters (α, ρ, ν) to market implied vols.

    Minimises the sum of squared errors between SABR and market vols.
    β (beta) is treated as fixed (typically 0.5 for equities).

    Parameters
    ----------
    strikes : np.ndarray
        Array of option strikes.
    market_vols : np.ndarray
        Corresponding market implied Black vols.
    F : float
        Forward price.
    T : float
        Time to expiry.
    beta : float
        Fixed CEV exponent (default 0.5).

    Returns
    -------
    (alpha, rho, nu) : tuple[float, float, float]
        Calibrated SABR parameters.
    """
    def objective(params: np.ndarray) -> float:
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or not (-1.0 < rho < 1.0):
            return 1e10
        try:
            model_vols = np.array([
                sabr_vol(F=F, K=k, T=T, alpha=alpha, beta=beta, rho=rho, nu=nu)
                for k in strikes
            ])
        except Exception:
            return 1e10
        return float(np.sum((model_vols - market_vols) ** 2))

    # Initial guess: use ATM vol to seed alpha
    atm_vol = market_vols[np.argmin(np.abs(strikes - F))]
    alpha0 = atm_vol * F**(1.0 - beta)
    x0 = np.array([alpha0, -0.2, 0.3])

    bounds = [(1e-6, None), (-0.999, 0.999), (1e-6, None)]
    result = minimize(
        objective, x0, method="L-BFGS-B", bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500},
    )

    # Try a second starting point if convergence is poor
    if result.fun > 1e-6:
        for rho_init in [0.0, 0.3, -0.5]:
            res2 = minimize(
                objective,
                [alpha0, rho_init, 0.4],
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500},
            )
            if res2.fun < result.fun:
                result = res2

    alpha_fit, rho_fit, nu_fit = result.x
    return float(alpha_fit), float(np.clip(rho_fit, -0.999, 0.999)), float(nu_fit)
