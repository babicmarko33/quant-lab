"""Merton's Continuous-Time Portfolio Problem — HJB Solution (Phase 4.6).

Merton (1969, 1971): Optimal portfolio for an agent maximising expected
CRRA utility of terminal wealth W(T).

Setup:
    dW = [r*W + w*(μ-r)*W - c*W] dt + w*σ*W dB
where w = fraction in risky asset, c = consumption rate (set to 0 here),
r = risk-free rate, μ = risky asset drift, σ = volatility.

Utility: U(W) = W^γ / γ,  γ ∈ (-∞, 0) ∪ (0, 1)

HJB solution (Merton, 1971):
    w* = (μ - r) / (σ² * (1 - γ))      ← optimal investment fraction
    c* = 0 (no consumption, terminal utility only)

Value function:
    V(W, t) = W^γ / γ * exp(A(T-t))
where:
    A = γ * [r + (μ-r)² / (2*σ²*(1-γ))]
      = γ * [r + w* * (μ-r) / 2]   (equivalent form)

Note: V(W, T) = U(W) = W^γ / γ  (terminal condition, A=0 at t=T).
"""

from __future__ import annotations

import math


def merton_optimal_weight(
    mu: float,
    r: float,
    sigma: float,
    gamma: float,
) -> float:
    """Compute Merton's optimal risky-asset investment fraction.

    Parameters
    ----------
    mu : float
        Expected return of the risky asset (drift).
    r : float
        Risk-free rate.
    sigma : float
        Volatility of the risky asset (σ > 0).
    gamma : float
        CRRA risk-aversion parameter. Must be in (-∞, 0) ∪ (0, 1).
        γ close to 1 → nearly risk-neutral; γ close to 0 → log utility limit.

    Returns
    -------
    float
        Optimal portfolio weight w* = (μ-r) / (σ²*(1-γ)).
    """
    if gamma == 0.0 or gamma >= 1.0:
        raise ValueError(
            f"gamma must be in (-inf, 0) or (0, 1) for CRRA utility, got {gamma}"
        )
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    return float((mu - r) / (sigma**2 * (1.0 - gamma)))


def merton_value_function(
    W: float,
    T: float,
    mu: float,
    r: float,
    sigma: float,
    gamma: float,
    t: float = 0.0,
) -> float:
    """Compute Merton's value function V(W, t) for the terminal wealth problem.

    V(W, t) = (W^γ / γ) * exp(A * (T - t))

    where A = γ * [r + (μ-r)² / (2*σ²*(1-γ))]

    Parameters
    ----------
    W : float
        Current wealth (W > 0).
    T : float
        Terminal time (horizon).
    mu, r, sigma, gamma : float
        See merton_optimal_weight.
    t : float
        Current time (default 0.0).

    Returns
    -------
    float
        Value function V(W, t).
    """
    if gamma == 0.0 or gamma >= 1.0:
        raise ValueError(
            f"gamma must be in (-inf, 0) or (0, 1) for CRRA utility, got {gamma}"
        )
    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if W <= 0.0:
        raise ValueError(f"W must be positive, got {W}")

    tau = T - t  # time remaining

    # Exponent A: γ * [r + (μ-r)² / (2σ²(1-γ))]
    A = gamma * (r + (mu - r) ** 2 / (2.0 * sigma**2 * (1.0 - gamma)))

    # Utility component: W^γ / γ
    utility = (W**gamma) / gamma

    return float(utility * math.exp(A * tau))
