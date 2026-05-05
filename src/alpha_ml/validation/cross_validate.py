"""Purged cross-validation for ML model evaluation.

Generates fully OOS predictions by training on each fold's train set
and predicting on the test set. Uses PurgedKFold to prevent look-ahead.

Reference:
    López de Prado (2018). Advances in Financial Machine Learning.
    Chapter 7: Cross-Validation in Finance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from alpha_ml.models.base import ModelTrainer
from alpha_ml.validation.purged_kfold import PurgedKFold


@dataclass
class CVResult:
    """Results from a purged cross-validation run.

    Attributes
    ----------
    oos_predictions : pd.Series
        Class predictions for every row, assembled from OOS folds.
    oos_probabilities : pd.DataFrame
        Probability matrix (n_samples × 2), assembled from OOS folds.
    fold_accuracies : list[float]
        Accuracy per fold.
    oos_accuracy : float
        Mean OOS accuracy across folds.
    feature_importance : pd.Series
        Mean feature importance across folds.
    """

    oos_predictions: pd.Series
    oos_probabilities: pd.DataFrame
    fold_accuracies: list[float]
    oos_accuracy: float
    feature_importance: pd.Series


def cross_val_predict_purged(
    model: ModelTrainer,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    purge_window: int = 5,
    embargo: int = 2,
) -> CVResult:
    """Evaluate model using purged k-fold, returning full OOS predictions.

    For each fold:
      1. Train model on purged train set
      2. Predict on test set
      3. Accumulate predictions and feature importances

    Parameters
    ----------
    model : ModelTrainer
        Any concrete ModelTrainer. A fresh copy is cloned per fold.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    n_splits : int
        Number of CV folds.
    purge_window : int
        Bars to purge around test fold boundary (set to prediction horizon).
    embargo : int
        Bars to embargo after test fold.

    Returns
    -------
    CVResult
    """
    import copy

    pkf = PurgedKFold(n_splits=n_splits, purge_window=purge_window, embargo=embargo)

    pred_array = np.zeros(len(X), dtype=int)
    proba_array = np.zeros((len(X), 2))
    fold_accuracies: list[float] = []
    fold_importances: list[pd.Series] = []

    for train_idx, test_idx in pkf.split(X):
        # Clone model to avoid state bleed across folds
        fold_model = copy.deepcopy(model)

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        fold_model.fit(X_train, y_train)

        preds = fold_model.predict(X_test)
        probas = fold_model.predict_proba(X_test)

        pred_array[test_idx] = preds
        proba_array[test_idx] = probas

        fold_acc = float(accuracy_score(y_test, preds))
        fold_accuracies.append(fold_acc)

        fold_importances.append(fold_model.feature_importance())

    # Aggregate predictions into Series/DataFrame with original index
    oos_predictions = pd.Series(pred_array, index=X.index, name="prediction")
    oos_probabilities = pd.DataFrame(proba_array, index=X.index, columns=[0, 1])

    # Average feature importance across folds
    importance_df = pd.DataFrame(fold_importances)
    mean_importance = importance_df.mean()

    return CVResult(
        oos_predictions=oos_predictions,
        oos_probabilities=oos_probabilities,
        fold_accuracies=fold_accuracies,
        oos_accuracy=float(np.mean(fold_accuracies)),
        feature_importance=mean_importance,
    )
