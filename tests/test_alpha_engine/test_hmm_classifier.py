"""Tests for HMM regime classifier. Written before implementation (TDD RED)."""
import numpy as np
import pandas as pd
import pytest

from alpha_engine.regime.hmm_classifier import RegimeClassifier

N = 500


@pytest.fixture
def two_regime_returns() -> pd.Series:
    """Alternating high-vol / low-vol regimes."""
    rng = np.random.default_rng(42)
    r = np.empty(N)
    for i in range(N):
        # Switch every ~100 bars
        high_vol = (i // 100) % 2 == 1
        vol = 0.025 if high_vol else 0.008
        r[i] = rng.normal(0, vol)
    return pd.Series(r, name="returns")


class TestRegimeClassifier:
    def test_instantiation(self):
        clf = RegimeClassifier(n_regimes=2)
        assert isinstance(clf, RegimeClassifier)

    def test_fit_returns_self(self, two_regime_returns):
        clf = RegimeClassifier(n_regimes=2)
        result = clf.fit(two_regime_returns)
        assert result is clf

    def test_predict_shape(self, two_regime_returns):
        clf = RegimeClassifier(n_regimes=2)
        clf.fit(two_regime_returns)
        labels = clf.predict(two_regime_returns)
        assert labels.shape == (N,)

    def test_predict_integer_labels(self, two_regime_returns):
        clf = RegimeClassifier(n_regimes=2)
        clf.fit(two_regime_returns)
        labels = clf.predict(two_regime_returns)
        unique = set(labels)
        assert unique.issubset({0, 1})

    def test_predict_finds_two_regimes(self, two_regime_returns):
        """Should assign at least some observations to each regime."""
        clf = RegimeClassifier(n_regimes=2)
        clf.fit(two_regime_returns)
        labels = clf.predict(two_regime_returns)
        assert len(set(labels)) == 2

    def test_regime_vols_differ(self, two_regime_returns):
        """Regimes should capture distinctly different volatility levels."""
        clf = RegimeClassifier(n_regimes=2)
        clf.fit(two_regime_returns)
        labels = clf.predict(two_regime_returns)
        r = two_regime_returns.values
        vol0 = r[labels == 0].std()
        vol1 = r[labels == 1].std()
        # The two vol levels should differ by at least 50%
        ratio = max(vol0, vol1) / min(vol0, vol1)
        assert ratio > 1.5, f"Vol ratio {ratio:.2f} too small"

    def test_predict_proba_shape(self, two_regime_returns):
        clf = RegimeClassifier(n_regimes=2)
        clf.fit(two_regime_returns)
        proba = clf.predict_proba(two_regime_returns)
        assert proba.shape == (N, 2)

    def test_predict_proba_sums_to_one(self, two_regime_returns):
        clf = RegimeClassifier(n_regimes=2)
        clf.fit(two_regime_returns)
        proba = clf.predict_proba(two_regime_returns)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_three_regimes(self, two_regime_returns):
        clf = RegimeClassifier(n_regimes=3)
        clf.fit(two_regime_returns)
        labels = clf.predict(two_regime_returns)
        assert labels.shape == (N,)

    def test_raises_before_fit(self):
        clf = RegimeClassifier(n_regimes=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            clf.predict(pd.Series([0.01, 0.02]))

    def test_raises_on_too_short(self):
        clf = RegimeClassifier(n_regimes=2)
        short = pd.Series(np.ones(5) * 0.01)
        with pytest.raises(ValueError, match="at least"):
            clf.fit(short)
