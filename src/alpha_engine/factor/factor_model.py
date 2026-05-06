"""OLS-based Fama-French factor attribution model.

Regresses portfolio excess returns on factor returns to estimate alpha,
factor betas, and R².

Usage::

    model = FactorModel()
    result = model.fit(portfolio_returns, ff_factors_df)
    print(result.alpha, result.betas, result.r_squared)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorResult:
    """Output of a Fama-French factor regression.

    Attributes
    ----------
    alpha:
        Annualised Jensen's alpha (intercept × 12 for monthly data).
    betas:
        Dict mapping factor name → estimated beta.
    r_squared:
        OLS R² (coefficient of determination).
    residuals:
        OLS residuals aligned to input index.
    t_stats:
        Dict mapping coefficient name → t-statistic (intercept key: ``"alpha"``).
    """

    alpha: float
    betas: dict[str, float]
    r_squared: float
    residuals: np.ndarray
    t_stats: dict[str, float]


class FactorModel:
    """Ordinary-least-squares Fama-French factor attribution.

    Regresses ``(returns - RF)`` on the factor columns present in
    ``factors`` DataFrame.

    The intercept is annualised (×12) to give Jensen's alpha.
    """

    def fit(self, returns: pd.Series, factors: pd.DataFrame) -> FactorResult:
        """Run OLS factor regression.

        Parameters
        ----------
        returns:
            Portfolio or asset monthly returns (decimal). Must share index
            (or be alignable) with ``factors``.
        factors:
            DataFrame with factor columns.  Must include ``RF`` for
            excess-return calculation if present; otherwise raw returns
            are used.

        Returns
        -------
        FactorResult
        """
        # Align on common index
        common = returns.index.intersection(factors.index)
        ret = returns.loc[common]
        fac = factors.loc[common].copy()

        # Compute excess returns
        rf = fac.pop("RF") if "RF" in fac.columns else pd.Series(0.0, index=common)
        excess = ret - rf

        # Build design matrix [1, f1, f2, ...]
        factor_names = list(fac.columns)
        design_matrix = np.column_stack([np.ones(len(excess)), fac.values])
        y = excess.values

        # OLS via lstsq
        coeffs, residuals_ss, rank, _ = np.linalg.lstsq(design_matrix, y, rcond=None)
        y_hat = design_matrix @ coeffs
        residuals = y - y_hat

        # R²
        ss_tot = np.sum((y - y.mean()) ** 2)
        ss_res = np.sum(residuals ** 2)
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # t-stats via OLS sandwich estimator
        n, k = design_matrix.shape
        sigma2 = ss_res / max(n - k, 1)
        xtx_inv = np.linalg.pinv(design_matrix.T @ design_matrix)
        se = np.sqrt(np.diag(xtx_inv) * sigma2)
        t_vals = coeffs / np.where(se > 0, se, np.nan)

        betas = dict(zip(factor_names, coeffs[1:].tolist(), strict=False))
        t_stats = {"alpha": float(t_vals[0])}
        t_stats.update(dict(zip(factor_names, t_vals[1:].tolist(), strict=False)))

        # Annualise alpha (monthly data → ×12)
        alpha_annualised = float(coeffs[0]) * 12

        return FactorResult(
            alpha=alpha_annualised,
            betas=betas,
            r_squared=r_squared,
            residuals=residuals,
            t_stats=t_stats,
        )
