"""Tests for Phase 5.3: Model Ensemble."""
import numpy as np
import pandas as pd
import pytest

from alpha_ml.models.ensemble import ModelEnsemble
from alpha_ml.models.linear import LassoClassifier, RidgeClassifier


@pytest.fixture
def dataset():
    rng = np.random.default_rng(99)
    n = 200
    X = pd.DataFrame(
        rng.standard_normal((n, 3)),
        columns=["f1", "f2", "f3"],
    )
    y = pd.Series((X["f1"] - X["f2"] > 0).astype(int), name="target")
    return X, y


class TestModelEnsemble:
    def test_predict_shape(self, dataset):
        X, y = dataset
        ens = ModelEnsemble([RidgeClassifier(alpha=1.0), LassoClassifier(alpha=0.1)])
        ens.fit(X, y)
        preds = ens.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_values_binary(self, dataset):
        X, y = dataset
        ens = ModelEnsemble([RidgeClassifier(), LassoClassifier()])
        ens.fit(X, y)
        preds = ens.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, dataset):
        X, y = dataset
        ens = ModelEnsemble([RidgeClassifier(), LassoClassifier()])
        ens.fit(X, y)
        proba = ens.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, dataset):
        X, y = dataset
        ens = ModelEnsemble([RidgeClassifier(), LassoClassifier()])
        ens.fit(X, y)
        proba = ens.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_feature_importance_index(self, dataset):
        X, y = dataset
        ens = ModelEnsemble([RidgeClassifier(), LassoClassifier()])
        ens.fit(X, y)
        imp = ens.feature_importance()
        assert list(imp.index) == list(X.columns)

    def test_single_model_ensemble_equals_model(self, dataset):
        """Ensemble of one model produces identical probabilities."""
        X, y = dataset
        single = RidgeClassifier(alpha=1.0)
        single.fit(X, y)
        ens = ModelEnsemble([RidgeClassifier(alpha=1.0)])
        ens.fit(X, y)
        np.testing.assert_allclose(ens.predict_proba(X), single.predict_proba(X), atol=1e-6)

    def test_predict_before_fit_raises(self):
        ens = ModelEnsemble([RidgeClassifier()])
        X = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(RuntimeError, match="not fitted"):
            ens.predict(X)

    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ModelEnsemble([])

    def test_three_model_ensemble(self, dataset):
        X, y = dataset
        ens = ModelEnsemble([
            RidgeClassifier(alpha=0.1),
            RidgeClassifier(alpha=1.0),
            LassoClassifier(alpha=0.01),
        ])
        ens.fit(X, y)
        acc = (ens.predict(X) == y.values).mean()
        assert acc > 0.55

    def test_proba_is_average_of_members(self, dataset):
        """Ensemble proba = mean of member probas."""
        X, y = dataset
        m1 = RidgeClassifier(alpha=0.1)
        m2 = LassoClassifier(alpha=0.1)
        m1.fit(X, y)
        m2.fit(X, y)
        expected = (m1.predict_proba(X) + m2.predict_proba(X)) / 2.0

        ens = ModelEnsemble([RidgeClassifier(alpha=0.1), LassoClassifier(alpha=0.1)])
        ens.fit(X, y)
        np.testing.assert_allclose(ens.predict_proba(X), expected, atol=1e-6)
