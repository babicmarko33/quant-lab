"""Risk Parity portfolio allocator.

Each asset contributes an equal share of total portfolio risk.
The risk contribution of asset i is:
    RC_i = w_i * (Sigma * w)_i

Risk parity requires: RC_i = RC_j for all i, j
Equivalently: RC_i = total_risk / n

Optimization objective (Spinu 2013):
    Minimize: sum_i [ w_i * (Sigma*w)_i - target_rc * log(w_i) ]

Reference:
    Maillard, S., Roncalli, T., Teiletche, J. (2010).
    The Properties of Equally Weighted Risk Contributions Portfolios.
    Journal of Portfolio Management, 36(4), 60-70.

    Spinu, F. (2013). An Algorithm for Computing Risk Parity Weights.
    SSRN 2297383.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from alpha_engine.portfolio.allocator import Allocator


def _risk_parity_objective(weights: np.ndarray, cov: np.ndarray) -> float:
    """Spinu (2013) objective for risk parity.

    Minimizing this produces equal risk contributions.
    f(w) = 0.5 * w' * Sigma * w - (1/n) * sum(log(w))
    """
    n = len(weights)
    port_var = float(weights @ cov @ weights)
    log_sum = float(np.sum(np.log(weights)))
    return 0.5 * port_var - (1.0 / n) * log_sum


def _risk_parity_gradient(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    n = len(weights)
    return cov @ weights - (1.0 / n) / weights


class RiskParityAllocator(Allocator):
    """Equal risk contribution (risk parity) portfolio allocator.

    Parameters
    ----------
    annualization : int
        Trading days per year. Used to annualize covariance. Default 252.
    """

    def __init__(self, annualization: int = 252) -> None:
        self.annualization = annualization

    def fit(self, returns: pd.DataFrame) -> pd.Series:
        """Compute risk parity weights."""
        n = len(returns.columns)
        cov = returns.cov().values * self.annualization

        # Start from equal weights — strictly positive (required by log objective)
        w0 = np.full(n, 1.0 / n)
        bounds = [(1e-6, None)] * n  # Strictly positive (log constraint)

        result = minimize(
            _risk_parity_objective,
            w0,
            args=(cov,),
            jac=_risk_parity_gradient,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 5000},
        )

        weights = result.x
        weights = np.clip(weights, 0.0, None)
        weights = weights / weights.sum()  # Normalize to sum = 1
        return pd.Series(weights, index=returns.columns, name="weight")
