"""Linear classification baselines for alpha_ml.

Implements Ridge and Lasso classifiers following the ModelTrainer ABC.
Both use logistic regression with the appropriate penalty (L2 for Ridge,
L1 for Lasso) via scikit-learn's LogisticRegression so that:
  - predict()        → {0, 1} class labels
  - predict_proba()  → (n, 2) probability matrix summing to 1
  - feature_importance() → pd.Series of |coefficient| magnitudes

Ridge (L2) never zeroes coefficients; Lasso (L1) encourages sparsity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from alpha_ml.models.base import ModelTrainer


class _LinearClassifierBase(ModelTrainer):
    """Shared logic for Ridge and Lasso classifiers."""

    def __init__(self, alpha: float = 1.0, penalty: str = "l2", max_iter: int = 1000) -> None:
        self.alpha = alpha
        self.penalty = penalty
        self.max_iter = max_iter
        self._model: LogisticRegression | None = None
        self._feature_names: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        # C = 1 / alpha in sklearn's convention
        self._feature_names = list(X.columns)
        solver = "saga" if self.penalty == "l1" else "lbfgs"
        self._model = LogisticRegression(
            penalty=self.penalty,
            C=1.0 / self.alpha if self.alpha > 0 else 1e12,
            solver=solver,
            max_iter=self.max_iter,
            random_state=42,
        )
        self._model.fit(X.values, y.values)

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X.values).astype(int)  # type: ignore[union-attr]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict_proba(X.values)  # type: ignore[union-attr]

    def feature_importance(self) -> pd.Series:
        self._check_fitted()
        assert self._feature_names is not None
        coef = self._model.coef_  # type: ignore[union-attr]
        # Binary case: shape (1, n_features); multi-class: (n_classes, n_features)
        importance = np.abs(coef).mean(axis=0)
        return pd.Series(importance, index=self._feature_names)


class RidgeClassifier(_LinearClassifierBase):
    """L2-regularised logistic regression (Ridge) classifier.

    Parameters
    ----------
    alpha : float
        Regularisation strength (higher = more regularised, default 1.0).
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000) -> None:
        super().__init__(alpha=alpha, penalty="l2", max_iter=max_iter)


class LassoClassifier(_LinearClassifierBase):
    """L1-regularised logistic regression (Lasso) classifier.

    High alpha values encourage sparse coefficients (some features zeroed out).

    Parameters
    ----------
    alpha : float
        Regularisation strength (higher = more sparse, default 1.0).
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000) -> None:
        super().__init__(alpha=alpha, penalty="l1", max_iter=max_iter)
