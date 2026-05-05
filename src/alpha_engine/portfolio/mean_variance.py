"""Mean-Variance (Markowitz) portfolio allocator.

Supports two objectives:
  1. ``max_sharpe``   — maximize Sharpe ratio (tangency portfolio)
  2. ``min_variance`` — minimize portfolio variance (global minimum variance)

Both are solved via scipy.optimize.minimize with:
  - Equality constraint: weights sum to 1
  - Bound constraint: weights in [0, 1] (long-only)

Reference:
    Markowitz, H. (1952). Portfolio Selection.
    Journal of Finance, 7(1), 77-91.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from alpha_engine.portfolio.allocator import Allocator

_MIN_WEIGHT = 0.0
_MAX_WEIGHT = 1.0


def _portfolio_variance(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(weights @ cov @ weights)


def _neg_sharpe(weights: np.ndarray, mean_returns: np.ndarray, cov: np.ndarray, rf: float) -> float:
    port_return = float(weights @ mean_returns)
    port_vol = float(np.sqrt(weights @ cov @ weights))
    if port_vol < 1e-12:
        return 0.0
    return -(port_return - rf) / port_vol


class MeanVarianceAllocator(Allocator):
    """Markowitz mean-variance portfolio optimizer.

    Parameters
    ----------
    objective : str
        'max_sharpe' (default) or 'min_variance'.
    rf : float
        Annual risk-free rate for Sharpe computation. Default 0.0.
    annualization : int
        Trading days per year for annualizing returns/covariance. Default 252.
    """

    def __init__(
        self,
        objective: str = "max_sharpe",
        rf: float = 0.0,
        annualization: int = 252,
    ) -> None:
        if objective not in ("max_sharpe", "min_variance"):
            raise ValueError(f"objective must be 'max_sharpe' or 'min_variance', got '{objective}'")
        self.objective = objective
        self.rf = rf
        self.annualization = annualization

    def fit(self, returns: pd.DataFrame) -> pd.Series:
        """Optimize portfolio weights."""
        n = len(returns.columns)
        mean_ret = returns.mean().values * self.annualization
        cov = returns.cov().values * self.annualization

        # Initial guess: equal weight
        w0 = np.full(n, 1.0 / n)
        bounds = [(_MIN_WEIGHT, _MAX_WEIGHT)] * n
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

        if self.objective == "max_sharpe":
            result = minimize(
                _neg_sharpe,
                w0,
                args=(mean_ret, cov, self.rf),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 1000},
            )
        else:  # min_variance
            result = minimize(
                _portfolio_variance,
                w0,
                args=(cov,),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 1000},
            )

        weights = np.clip(result.x, 0.0, 1.0)
        # Re-normalize to exactly sum to 1
        weights = weights / weights.sum()
        return pd.Series(weights, index=returns.columns, name="weight")
