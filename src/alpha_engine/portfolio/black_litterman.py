"""Black-Litterman Portfolio Allocation (Phase 4.7).

The Black-Litterman model blends market equilibrium returns (implied by
market capitalisation weights) with investor views to produce posterior
expected returns, which are then fed into mean-variance optimisation.

Algorithm
---------
1. Compute equilibrium excess returns:
       Π = δ * Σ * w_mkt
   where δ is the risk-aversion coefficient, Σ the covariance matrix,
   and w_mkt the market-cap weights.

2. If no views: posterior mean = Π, weights = w_mkt (normalised).

3. With views matrix P (k×n), view returns Q (k,), and view uncertainty Ω (k×k):
       posterior_mean = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]
       posterior_cov  = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹

4. Optimal weights via unconstrained MV:
       w* = (δ * Σ)⁻¹ * posterior_mean
   Then normalise to sum to 1 (long-only clamp to 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.portfolio.allocator import Allocator


class BlackLittermanAllocator(Allocator):
    """Black-Litterman portfolio allocator.

    Parameters
    ----------
    market_caps : pd.Series
        Market capitalisation for each asset (same index as returns columns).
    tau : float
        Uncertainty scaling of prior covariance (default 0.05).
    delta : float
        Risk-aversion coefficient (default 2.5).
    P : np.ndarray | None
        Views matrix (k × n_assets). None = no views.
    Q : np.ndarray | None
        View returns vector (k,).
    omega : np.ndarray | None
        View uncertainty covariance (k × k). None = proportional to τΣ.
    """

    def __init__(
        self,
        market_caps: pd.Series,
        tau: float = 0.05,
        delta: float = 2.5,
        P: np.ndarray | None = None,
        Q: np.ndarray | None = None,
        omega: np.ndarray | None = None,
    ) -> None:
        self.market_caps = market_caps
        self.tau = tau
        self.delta = delta
        self.P = P
        self.Q = Q
        self.omega = omega

    def fit(self, returns: pd.DataFrame) -> pd.Series:
        assets = list(returns.columns)

        # Align market caps to returns columns — raises KeyError if mismatch
        mkt_caps = self.market_caps.loc[assets]
        w_mkt = mkt_caps / mkt_caps.sum()

        Sigma = returns.cov().values  # n × n annualised-scale covariance
        n = len(assets)

        # Step 1: Equilibrium excess returns Π = δ * Σ * w_mkt
        Pi = self.delta * Sigma @ w_mkt.values

        if self.P is None or self.Q is None:
            # No views: optimal weights ≈ market-cap weights
            mu_bl = Pi
        else:
            P = np.asarray(self.P)
            Q = np.asarray(self.Q)
            Omega = (
                np.asarray(self.omega)
                if self.omega is not None
                else np.diag(np.diag(self.tau * P @ Sigma @ P.T))
            )

            # Step 3: Posterior mean (BL formula)
            tau_sigma_inv = np.linalg.inv(self.tau * Sigma)
            omega_inv = np.linalg.inv(Omega)
            M_inv = tau_sigma_inv + P.T @ omega_inv @ P
            M = np.linalg.inv(M_inv)
            mu_bl = M @ (tau_sigma_inv @ Pi + P.T @ omega_inv @ Q)

        # Step 4: Unconstrained MV weights
        w_raw = np.linalg.solve(self.delta * Sigma, mu_bl)
        # Clamp negatives and normalise
        w_raw = np.clip(w_raw, 0.0, None)
        total = w_raw.sum()
        if total <= 0:
            # Fallback to equal weight
            w_raw = np.ones(n) / n
        else:
            w_raw /= total

        return pd.Series(w_raw, index=assets)
