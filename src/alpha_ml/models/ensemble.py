"""Model Ensemble for alpha_ml (Phase 5.3).

Soft voting ensemble: averages predict_proba outputs from N base models.
Implements the ModelTrainer ABC so it can be used anywhere a single model is expected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_ml.models.base import ModelTrainer


class ModelEnsemble(ModelTrainer):
    """Soft-voting ensemble of ModelTrainer instances.

    Predictions are the argmax of averaged class probabilities.
    Feature importances are the mean of all members' importances.

    Parameters
    ----------
    models : list[ModelTrainer]
        Base models. Must have at least one element.
    """

    def __init__(self, models: list[ModelTrainer]) -> None:
        if not models:
            raise ValueError("ModelEnsemble requires at least one base model.")
        self.models = models
        self._fitted = False

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Ensemble is not fitted. Call fit() first.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        for model in self.models:
            model.fit(X, y)
        self._fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.predict_proba(X).argmax(axis=1).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        probas = np.stack([m.predict_proba(X) for m in self.models], axis=0)
        return probas.mean(axis=0)

    def feature_importance(self) -> pd.Series:
        self._check_fitted()
        imps = np.stack([m.feature_importance().values for m in self.models], axis=0)
        index = self.models[0].feature_importance().index
        return pd.Series(imps.mean(axis=0), index=index)
