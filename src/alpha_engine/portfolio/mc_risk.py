"""Monte Carlo VaR and CVaR with Cholesky-correlated returns (Phase 4.5).

Simulation method:
    1. Estimate mean (μ) and covariance (Σ) from historical returns.
    2. Decompose Σ = L*Lᵀ via Cholesky.
    3. Simulate n_paths portfolio returns:
           r_sim = μ + L * Z,  Z ~ N(0, I)
           r_portfolio = w · r_sim  (dot product per path)
    4. VaR_α  = -quantile(r_portfolio, 1 - α)
       CVaR_α = -mean(r_portfolio[r_portfolio ≤ -VaR_α])

The Cholesky decomposition preserves the full correlation structure, making
this a proper multivariate MC risk measure (unlike historical simulation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _simulate_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
    n_paths: int,
    seed: int | None,
) -> np.ndarray:
    """Simulate portfolio returns using Cholesky decomposition."""
    assets = list(returns.columns)
    w = weights.loc[assets].values  # raises KeyError if mismatch

    mu = returns.mean().values
    Sigma = returns.cov().values

    # Cholesky decomposition: Sigma = L @ L.T
    L = np.linalg.cholesky(Sigma)

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((len(assets), n_paths))   # (k, n_paths)
    asset_returns = mu[:, None] + L @ Z               # (k, n_paths)
    return w @ asset_returns                           # (n_paths,)


def mc_var(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95,
    n_paths: int = 100_000,
    horizon: int = 1,
    seed: int | None = None,
) -> float:
    """Compute Monte Carlo Value-at-Risk (VaR).

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns. Used to estimate μ and Σ.
    weights : pd.Series
        Portfolio weights indexed by asset names.
    confidence : float
        VaR confidence level, e.g. 0.95 or 0.99 (default 0.95).
    n_paths : int
        Number of MC scenarios (default 100_000).
    horizon : int
        Holding period in days (default 1). Scales by sqrt(horizon).
    seed : int | None
        Random seed.

    Returns
    -------
    float
        VaR as a positive number (loss convention).
    """
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    port_returns = _simulate_portfolio_returns(returns, weights, n_paths, seed)
    if horizon > 1:
        port_returns *= np.sqrt(horizon)

    loss_quantile = np.quantile(port_returns, 1.0 - confidence)
    return float(-loss_quantile)


def mc_cvar(
    returns: pd.DataFrame,
    weights: pd.Series,
    confidence: float = 0.95,
    n_paths: int = 100_000,
    horizon: int = 1,
    seed: int | None = None,
) -> float:
    """Compute Monte Carlo Conditional VaR (CVaR / Expected Shortfall).

    CVaR is the expected loss in the worst (1-confidence) fraction of scenarios.
    CVaR >= VaR always holds.

    Parameters
    ----------
    See mc_var for parameter descriptions.

    Returns
    -------
    float
        CVaR as a positive number (loss convention).
    """
    if not (0 < confidence < 1):
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    port_returns = _simulate_portfolio_returns(returns, weights, n_paths, seed)
    if horizon > 1:
        port_returns *= np.sqrt(horizon)

    var_threshold = np.quantile(port_returns, 1.0 - confidence)
    tail = port_returns[port_returns <= var_threshold]
    return float(-tail.mean())
