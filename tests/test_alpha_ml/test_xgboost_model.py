"""Tests for ModelTrainer ABC and XGBoostPredictor."""

import numpy as np
import pandas as pd
import pytest

from alpha_ml.models.base import ModelTrainer
from alpha_ml.models.xgboost_model import XGBoostPredictor


@pytest.fixture
def sample_X() -> pd.DataFrame:
    """Simple 5-feature synthetic feature matrix."""
    rng = np.random.default_rng(0)
    n = 300
    return pd.DataFrame(
        rng.standard_normal((n, 5)),
        columns=["f1", "f2", "f3", "f4", "f5"],
    )


@pytest.fixture
def sample_y(sample_X: pd.DataFrame) -> pd.Series:
    """Binary labels with slight signal from first feature."""
    rng = np.random.default_rng(1)
    signal = (sample_X["f1"] + rng.standard_normal(len(sample_X)) * 0.5) > 0
    return signal.astype(int).rename("target")


class TestModelTrainerABC:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            ModelTrainer()  # type: ignore[abstract]


class TestXGBoostPredictor:
    def test_fit_predict_returns_array(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        model = XGBoostPredictor(n_estimators=10, max_depth=3)
        model.fit(sample_X, sample_y)
        preds = model.predict(sample_X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(sample_X)

    def test_predictions_are_binary(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        model = XGBoostPredictor(n_estimators=10)
        model.fit(sample_X, sample_y)
        preds = model.predict(sample_X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shape(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        model = XGBoostPredictor(n_estimators=10)
        model.fit(sample_X, sample_y)
        proba = model.predict_proba(sample_X)
        assert proba.shape == (len(sample_X), 2)

    def test_predict_proba_sums_to_one(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        model = XGBoostPredictor(n_estimators=10)
        model.fit(sample_X, sample_y)
        proba = model.predict_proba(sample_X)
        row_sums = proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_in_01(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        model = XGBoostPredictor(n_estimators=10)
        model.fit(sample_X, sample_y)
        proba = model.predict_proba(sample_X)
        assert (proba >= 0).all()
        assert (proba <= 1).all()

    def test_feature_importance_is_series(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        model = XGBoostPredictor(n_estimators=10)
        model.fit(sample_X, sample_y)
        imp = model.feature_importance()
        assert isinstance(imp, pd.Series)
        assert len(imp) == sample_X.shape[1]
        assert list(imp.index) == list(sample_X.columns)

    def test_feature_importance_sums_to_one(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        model = XGBoostPredictor(n_estimators=10)
        model.fit(sample_X, sample_y)
        imp = model.feature_importance()
        assert abs(imp.sum() - 1.0) < 1e-6

    def test_not_fitted_predict_raises(self, sample_X: pd.DataFrame) -> None:
        model = XGBoostPredictor()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(sample_X)

    def test_not_fitted_proba_raises(self, sample_X: pd.DataFrame) -> None:
        model = XGBoostPredictor()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(sample_X)

    def test_reproducible_with_seed(self, sample_X: pd.DataFrame, sample_y: pd.Series) -> None:
        m1 = XGBoostPredictor(n_estimators=20, random_state=42)
        m2 = XGBoostPredictor(n_estimators=20, random_state=42)
        m1.fit(sample_X, sample_y)
        m2.fit(sample_X, sample_y)
        assert np.array_equal(m1.predict(sample_X), m2.predict(sample_X))
