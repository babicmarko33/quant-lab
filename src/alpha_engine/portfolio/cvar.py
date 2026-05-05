"""CVaR (Conditional Value-at-Risk / Expected Shortfall) portfolio allocator.

Minimizes the portfolio's Expected Shortfall at confidence level alpha.
CVaR is a coherent risk measure: convex, sub-additive, translation invariant.

Uses the Rockafellar-Uryasev (2000) linear programming formulation:
    CVaR_alpha(w) = VaR + (1/(1-alpha)*T) * sum_t(max(loss_t - VaR, 0))

This is reformulated as a convex QP/LP solvable with cvxpy.

Reference:
    Rockafellar, R.T., Uryasev, S. (2000).
    Optimization of Conditional Value-at-Risk.
    Journal of Risk, 2(3), 21-41.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.portfolio.allocator import Allocator


class CVaRAllocator(Allocator):
    """CVaR-minimizing portfolio allocator.

    Parameters
    ----------
    alpha : float
        Confidence level in (0, 1). Default 0.95 (95% CVaR).
    """

    def __init__(self, alpha: float = 0.95) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha

    def fit(self, returns: pd.DataFrame) -> pd.Series:
        """Minimize CVaR at confidence level alpha (long-only).

        Uses cvxpy's disciplined convex programming to formulate the
        Rockafellar-Uryasev LP.
        """
        import cvxpy as cp

        losses = -returns.values  # shape (n_obs, n_assets) — positive = loss
        n_obs, n = losses.shape

        # Decision variables
        w = cp.Variable(n, nonneg=True)          # portfolio weights
        var = cp.Variable()                      # VaR threshold (zeta)
        z = cp.Variable(n_obs, nonneg=True)          # auxiliary: excess losses above VaR

        # CVaR = VaR + 1/((1-alpha)*n_obs) * sum(z)
        cvar = var + (1.0 / ((1.0 - self.alpha) * n_obs)) * cp.sum(z)

        constraints = [
            cp.sum(w) == 1.0,                       # fully invested
            z >= losses @ w - var,                  # z >= loss - VaR (or 0)
        ]

        prob = cp.Problem(cp.Minimize(cvar), constraints)
        prob.solve(solver=cp.CLARABEL, verbose=False)

        if w.value is None:
            # Fallback to equal weight if solver fails
            weights = np.full(n, 1.0 / n)
        else:
            weights = np.clip(w.value, 0.0, 1.0)
            if weights.sum() < 1e-10:
                weights = np.full(n, 1.0 / n)
            else:
                weights = weights / weights.sum()

        return pd.Series(weights, index=returns.columns, name="weight")
