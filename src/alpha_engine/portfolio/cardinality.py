"""Cardinality-Constrained Portfolio Optimisation (Phase 4.4).

Solves the Mixed Integer Quadratic Program (MIQP):

    min  w' Σ w
    s.t. Σ w_i = 1
         w_i >= 0          ∀ i
         w_i <= z_i        ∀ i      (big-M = 1, weight ≤ selection indicator)
         z_i ∈ {0, 1}      ∀ i
         Σ z_i <= K                 (at most K assets)

This is the classic Markowitz minimum-variance portfolio with a cardinality
constraint enforced via binary selection variables.

Solver: CVXPY with SCIP (open-source MILP/MIQP solver bundled with CVXPY).
Falls back to GLPK_MI if SCIP is not available.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from alpha_engine.portfolio.allocator import Allocator


class CardinalityAllocator(Allocator):
    """Minimum-variance allocator with a cardinality (max-assets) constraint.

    Solves a Mixed Integer Quadratic Program (MIQP) via CVXPY.

    Parameters
    ----------
    max_assets : int
        Maximum number of non-zero weight assets (K in the MIQP).
    """

    def __init__(self, max_assets: int = 5) -> None:
        self.max_assets = max_assets

    def fit(self, returns: pd.DataFrame) -> pd.Series:
        n = returns.shape[1]
        assets = list(returns.columns)

        if self.max_assets <= 0:
            raise ValueError(f"max_assets must be >= 1, got {self.max_assets}")

        Sigma = returns.cov().values  # (n, n)
        k = min(self.max_assets, n)

        # Decision variables
        w = cp.Variable(n, nonneg=True)
        z = cp.Variable(n, boolean=True)

        # Objective: minimise portfolio variance
        objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma)))

        constraints = [
            cp.sum(w) == 1.0,          # fully invested
            w <= z,                    # weight only if selected
            cp.sum(z) <= k,            # at most K assets
        ]

        problem = cp.Problem(objective, constraints)

        # Try solvers in preference order
        for solver in [cp.SCIP, cp.GLPK_MI, cp.CBC]:
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status in {"optimal", "optimal_inaccurate"} and w.value is not None:
                    break
            except (cp.SolverError, Exception):
                continue

        if w.value is None:
            # Fallback: equal weight on top-K by Sharpe
            sharpe = returns.mean() / returns.std()
            top_k = sharpe.nlargest(k).index
            weights = np.zeros(n)
            for i, a in enumerate(assets):
                if a in top_k:
                    weights[i] = 1.0 / k
            return pd.Series(weights, index=assets)

        raw = np.clip(w.value, 0.0, None)
        total = raw.sum()
        if total > 0:
            raw /= total
        return pd.Series(raw, index=assets)
