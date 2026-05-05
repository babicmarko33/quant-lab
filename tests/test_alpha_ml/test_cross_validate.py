"""Tests for purged cross-validation evaluation."""

import numpy as np
import pandas as pd
import pytest

from alpha_ml.models.xgboost_model import XGBoostPredictor
from alpha_ml.validation.cross_validate import CVResult, cross_val_predict_purged


@pytest.fixture
def sample_X() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame(rng.standard_normal((n, 5)), columns=["f1", "f2", "f3", "f4", "f5"])


@pytest.fixture
def sample_y(sample_X: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(7)
    signal = (sample_X["f1"] + rng.standard_normal(len(sample_X)) * 0.5) > 0
    return signal.astype(int).rename("target")


class TestCrossValPredict:
    def test_returns_cv_result_type(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        model = XGBoostPredictor(n_estimators=10)
        result = cross_val_predict_purged(model, sample_X, sample_y, n_splits=3)
        assert isinstance(result, CVResult)

    def test_oos_predictions_full_coverage(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        """OOS predictions must cover all input rows."""
        model = XGBoostPredictor(n_estimators=10)
        result = cross_val_predict_purged(model, sample_X, sample_y, n_splits=3)
        assert len(result.oos_predictions) == len(sample_X)

    def test_oos_predictions_are_binary(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        model = XGBoostPredictor(n_estimators=10)
        result = cross_val_predict_purged(model, sample_X, sample_y, n_splits=3)
        assert set(result.oos_predictions.unique()).issubset({0, 1})

    def test_oos_probabilities_shape(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        model = XGBoostPredictor(n_estimators=10)
        result = cross_val_predict_purged(model, sample_X, sample_y, n_splits=3)
        assert result.oos_probabilities.shape == (len(sample_X), 2)

    def test_oos_accuracy_in_01(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        model = XGBoostPredictor(n_estimators=10)
        result = cross_val_predict_purged(model, sample_X, sample_y, n_splits=3)
        assert 0.0 <= result.oos_accuracy <= 1.0

    def test_fold_accuracies_list(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        n_splits = 4
        model = XGBoostPredictor(n_estimators=10)
        result = cross_val_predict_purged(model, sample_X, sample_y, n_splits=n_splits)
        assert len(result.fold_accuracies) == n_splits
        assert all(0.0 <= a <= 1.0 for a in result.fold_accuracies)

    def test_feature_importance_series(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        model = XGBoostPredictor(n_estimators=10)
        result = cross_val_predict_purged(model, sample_X, sample_y, n_splits=3)
        assert isinstance(result.feature_importance, pd.Series)
        assert len(result.feature_importance) == sample_X.shape[1]

    def test_oos_accuracy_matches_fold_mean(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        model = XGBoostPredictor(n_estimators=10)
        result = cross_val_predict_purged(model, sample_X, sample_y, n_splits=3)
        expected = float(np.mean(result.fold_accuracies))
        assert abs(result.oos_accuracy - expected) < 1e-10
