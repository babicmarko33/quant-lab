"""Abstract base class for all ML model trainers."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class ModelTrainer(ABC):
    """Abstract interface for supervised ML models in the alpha_ml pipeline.

    All concrete implementations must support:
      - fit(X, y)         → train on labeled data
      - predict(X)        → return class predictions (int array)
      - predict_proba(X)  → return probability matrix (n × n_classes)
      - feature_importance() → return pd.Series of feature importances
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train model on (X, y)."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions as integer array."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability matrix of shape (n_samples, n_classes)."""

    @abstractmethod
    def feature_importance(self) -> pd.Series:
        """Return feature importances as a Series indexed by feature name."""
