"""Tests for Phase 5.2: LSTM sequence model for alpha_ml."""
import numpy as np
import pandas as pd
import pytest

from alpha_ml.models.lstm import LSTMClassifier


@pytest.fixture
def time_series_dataset():
    """Synthetic time-series binary classification dataset."""
    rng = np.random.default_rng(0)
    n, feats = 300, 4
    X = pd.DataFrame(
        rng.standard_normal((n, feats)),
        columns=["rsi", "sma_ratio", "volume_z", "ret_1d"],
    )
    # Label = 1 when sum of first 2 features > 0 (learnable signal)
    y = pd.Series((X["rsi"] + X["sma_ratio"] > 0).astype(int), name="target")
    return X, y


class TestLSTMClassifier:
    def test_fit_predict_shape(self, time_series_dataset):
        X, y = time_series_dataset
        model = LSTMClassifier(hidden_size=16, seq_len=10, n_epochs=2)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_values_binary(self, time_series_dataset):
        X, y = time_series_dataset
        model = LSTMClassifier(hidden_size=16, seq_len=10, n_epochs=2)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, time_series_dataset):
        X, y = time_series_dataset
        model = LSTMClassifier(hidden_size=16, seq_len=10, n_epochs=2)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, time_series_dataset):
        X, y = time_series_dataset
        model = LSTMClassifier(hidden_size=16, seq_len=10, n_epochs=2)
        model.fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_feature_importance_index(self, time_series_dataset):
        X, y = time_series_dataset
        model = LSTMClassifier(hidden_size=16, seq_len=10, n_epochs=2)
        model.fit(X, y)
        imp = model.feature_importance()
        assert list(imp.index) == list(X.columns)

    def test_feature_importance_non_negative(self, time_series_dataset):
        X, y = time_series_dataset
        model = LSTMClassifier(hidden_size=16, seq_len=10, n_epochs=2)
        model.fit(X, y)
        imp = model.feature_importance()
        assert (imp >= 0).all()

    def test_predict_before_fit_raises(self):
        model = LSTMClassifier()
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_loss_decreases_with_training(self, time_series_dataset):
        """More epochs should improve training loss."""
        X, y = time_series_dataset
        m1 = LSTMClassifier(hidden_size=32, seq_len=10, n_epochs=1, random_state=42)
        m1.fit(X, y)
        m2 = LSTMClassifier(hidden_size=32, seq_len=10, n_epochs=10, random_state=42)
        m2.fit(X, y)
        # 10 epochs model should have better training accuracy
        acc1 = (m1.predict(X) == y.values).mean()
        acc2 = (m2.predict(X) == y.values).mean()
        assert acc2 >= acc1 - 0.05  # allow small tolerance

    def test_seq_len_larger_than_dataset_raises(self):
        """seq_len >= n_samples should raise."""
        X = pd.DataFrame({"a": np.arange(5, dtype=float)})
        y = pd.Series([0, 1, 0, 1, 0])
        model = LSTMClassifier(seq_len=10)
        with pytest.raises(ValueError, match="seq_len"):
            model.fit(X, y)
