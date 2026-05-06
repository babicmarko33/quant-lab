"""HMM-based market regime classifier.

Fits a Gaussian Hidden Markov Model to log-return features and classifies
each observation into one of ``n_regimes`` regimes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install hmmlearn>=0.3: pip install 'hmmlearn>=0.3'") from exc

_MIN_OBS = 20


class RegimeClassifier:
    """Gaussian HMM regime classifier.

    Parameters
    ----------
    n_regimes:
        Number of hidden states (regimes). Default 2 (bull/bear or low/high vol).
    n_iter:
        Maximum EM iterations for fitting.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_regimes: int = 2,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.random_state = random_state
        self._model: GaussianHMM | None = None

    def fit(self, returns: pd.Series) -> RegimeClassifier:
        """Fit the HMM to ``returns``.

        Parameters
        ----------
        returns:
            Daily decimal return series.

        Raises
        ------
        ValueError
            If series is too short (< ``n_regimes * 10``).
        """
        min_obs = max(_MIN_OBS, self.n_regimes * 10)
        if len(returns) < min_obs:
            raise ValueError(
                f"returns must have at least {min_obs} observations, got {len(returns)}"
            )

        X = self._build_features(returns)

        self._model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._model.fit(X)
        return self

    def predict(self, returns: pd.Series) -> np.ndarray:
        """Predict regime labels for ``returns``.

        Returns
        -------
        np.ndarray of int labels with shape ``(len(returns),)``.
        """
        self._check_fitted()
        X = self._build_features(returns)
        return self._model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, returns: pd.Series) -> np.ndarray:
        """Posterior regime probabilities.

        Returns
        -------
        np.ndarray of shape ``(len(returns), n_regimes)``.
        """
        self._check_fitted()
        X = self._build_features(returns)
        _, posteriors = self._model.score_samples(X)  # type: ignore[union-attr]
        return posteriors

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "RegimeClassifier is not fitted. Call fit() first."
            )

    @staticmethod
    def _build_features(returns: pd.Series) -> np.ndarray:
        """Build feature matrix: [return, abs_return] as proxy for vol."""
        r = returns.values.astype(float).reshape(-1, 1)
        abs_r = np.abs(r)
        return np.hstack([r, abs_r])
