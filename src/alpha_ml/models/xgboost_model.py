"""XGBoost classifier wrapper implementing the ModelTrainer interface.

Wraps xgboost.XGBClassifier with a clean API:
  - Raises RuntimeError("Model not fitted") before fit is called
  - feature_importance() returns normalized gain-based importances
  - predict_proba() always returns shape (n, 2) for binary classification
  - Deterministic with random_state
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from alpha_ml.models.base import ModelTrainer


class XGBoostPredictor(ModelTrainer):
    """XGBoost binary classifier with gain-based feature importance.

    Parameters
    ----------
    n_estimators : int
        Number of boosting rounds. Default 100.
    max_depth : int
        Maximum tree depth. Default 4.
    learning_rate : float
        Step size shrinkage. Default 0.1.
    random_state : int
        Random seed for reproducibility. Default 42.
    **kwargs
        Additional keyword arguments passed to XGBClassifier.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        random_state: int = 42,
        **kwargs: object,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._kwargs = kwargs
        self._model: xgb.XGBClassifier | None = None
        self._feature_names: list[str] = []

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() before predict().")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train XGBoost classifier on (X, y).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Integer target labels {0, 1}.
        """
        self._feature_names = list(X.columns)
        self._model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            eval_metric="logloss",
            verbosity=0,
            **self._kwargs,
        )
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions as int array."""
        self._check_fitted()
        assert self._model is not None
        return self._model.predict(X).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability matrix of shape (n_samples, 2)."""
        self._check_fitted()
        assert self._model is not None
        return self._model.predict_proba(X)

    def feature_importance(self) -> pd.Series:
        """Return normalized gain-based feature importances.

        Returns
        -------
        pd.Series
            Importances indexed by feature name, summing to 1.0.
        """
        self._check_fitted()
        assert self._model is not None
        raw = self._model.feature_importances_
        total = raw.sum()
        normalized = raw / total if total > 0 else raw
        return pd.Series(normalized, index=self._feature_names, name="importance")
