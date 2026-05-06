"""Tests for Phase 5.1: Linear (Ridge / Lasso) classification baseline."""
import numpy as np
import pandas as pd
import pytest

from alpha_ml.models.linear import LassoClassifier, RidgeClassifier


@pytest.fixture
def binary_dataset():
    """Synthetic binary classification dataset (2 features, 200 samples)."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame(
        rng.standard_normal((n, 3)),
        columns=["rsi", "sma_ratio", "volume_z"],
    )
    # Simple linear rule so model can learn it
    logit = X["rsi"] - X["sma_ratio"]
    y = pd.Series((logit > 0).astype(int), name="target")
    return X, y


class TestRidgeClassifier:
    def test_fit_predict_shape(self, binary_dataset):
        X, y = binary_dataset
        model = RidgeClassifier(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_values_binary(self, binary_dataset):
        X, y = binary_dataset
        model = RidgeClassifier(alpha=1.0)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, binary_dataset):
        X, y = binary_dataset
        model = RidgeClassifier(alpha=1.0)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, binary_dataset):
        X, y = binary_dataset
        model = RidgeClassifier(alpha=1.0)
        model.fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_feature_importance_index(self, binary_dataset):
        X, y = binary_dataset
        model = RidgeClassifier(alpha=1.0)
        model.fit(X, y)
        imp = model.feature_importance()
        assert list(imp.index) == list(X.columns)

    def test_accuracy_above_chance(self, binary_dataset):
        X, y = binary_dataset
        model = RidgeClassifier(alpha=1.0)
        model.fit(X, y)
        acc = (model.predict(X) == y.values).mean()
        assert acc > 0.6

    def test_predict_before_fit_raises(self):
        model = RidgeClassifier()
        X = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)


class TestLassoClassifier:
    def test_fit_predict_shape(self, binary_dataset):
        X, y = binary_dataset
        model = LassoClassifier(alpha=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_values_binary(self, binary_dataset):
        X, y = binary_dataset
        model = LassoClassifier(alpha=0.01)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, binary_dataset):
        X, y = binary_dataset
        model = LassoClassifier(alpha=0.01)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_lasso_sparsity(self, binary_dataset):
        """High alpha should zero-out some coefficients (sparsity)."""
        X, y = binary_dataset
        model = LassoClassifier(alpha=1.0)
        model.fit(X, y)
        imp = model.feature_importance()
        assert (imp == 0.0).any()

    def test_feature_importance_index(self, binary_dataset):
        X, y = binary_dataset
        model = LassoClassifier(alpha=0.01)
        model.fit(X, y)
        imp = model.feature_importance()
        assert list(imp.index) == list(X.columns)

    def test_predict_before_fit_raises(self):
        model = LassoClassifier()
        X = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)
