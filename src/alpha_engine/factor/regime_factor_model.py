"""Regime-aware Fama-French factor attribution.

Splits portfolio returns by HMM regime label and runs a separate OLS
FactorModel for each regime, revealing how alpha and betas shift across
market states (e.g. bull / bear, low-vol / high-vol).

Usage::

    model = RegimeFactorModel()
    results = model.fit(monthly_returns, ff_factors_df, regime_labels)
    df = RegimeFactorModel.summary(results)
    print(df)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_engine.factor.factor_model import FactorModel, FactorResult


@dataclass
class RegimeFactorResult:
    """Factor attribution result for a single regime.

    Attributes
    ----------
    regime:
        Integer regime label.
    alpha:
        Annualised Jensen's alpha for this regime.
    betas:
        Factor betas dict (same keys as FactorModel.fit output).
    r_squared:
        OLS R² for this regime's subset.
    t_stats:
        t-statistics dict.
    residuals:
        OLS residuals for this regime's observations.
    n_obs:
        Number of observations in this regime.
    """

    regime: int
    alpha: float
    betas: dict[str, float]
    r_squared: float
    t_stats: dict[str, float]
    residuals: np.ndarray
    n_obs: int


class RegimeFactorModel:
    """Run FactorModel independently for each HMM regime.

    Accepts pre-computed regime labels (e.g. from
    :class:`~alpha_engine.regime.hmm_classifier.RegimeClassifier`) so the
    class itself has no dependency on the HMM fitting step.
    """

    def fit(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
        regimes: np.ndarray,
    ) -> dict[int, RegimeFactorResult]:
        """Fit a separate OLS FactorModel for each regime.

        Parameters
        ----------
        returns:
            Portfolio monthly returns (decimal). Must share index with
            ``factors``.
        factors:
            Fama-French factor DataFrame (must include ``RF`` column).
        regimes:
            Integer array of regime labels, one per row. Must have the
            same length as ``returns``.

        Returns
        -------
        dict mapping regime label → :class:`RegimeFactorResult`.
        """
        if len(regimes) != len(returns):
            raise ValueError(
                f"regimes length ({len(regimes)}) must match returns length ({len(returns)})"
            )

        regime_labels = sorted({int(r) for r in regimes})
        inner = FactorModel()
        results: dict[int, RegimeFactorResult] = {}

        for label in regime_labels:
            mask = np.asarray(regimes) == label
            subset_returns = returns.iloc[mask]
            subset_factors = factors.iloc[mask]

            fr: FactorResult = inner.fit(subset_returns, subset_factors.copy())

            results[label] = RegimeFactorResult(
                regime=label,
                alpha=fr.alpha,
                betas=fr.betas,
                r_squared=fr.r_squared,
                t_stats=fr.t_stats,
                residuals=fr.residuals,
                n_obs=int(mask.sum()),
            )

        return results

    @staticmethod
    def summary(results: dict[int, RegimeFactorResult]) -> pd.DataFrame:
        """Convert a results dict to a tidy summary DataFrame.

        Each row is one regime.  Columns include ``regime``, ``alpha``,
        ``r_squared``, ``n_obs``, and one column per factor beta.
        """
        rows = []
        for label, rfr in sorted(results.items()):
            row: dict = {
                "regime": label,
                "alpha": rfr.alpha,
                "r_squared": rfr.r_squared,
                "n_obs": rfr.n_obs,
            }
            for factor, beta in rfr.betas.items():
                row[f"beta_{factor}"] = beta
            for name, t in rfr.t_stats.items():
                row[f"t_{name}"] = t
            rows.append(row)
        return pd.DataFrame(rows).set_index("regime")
