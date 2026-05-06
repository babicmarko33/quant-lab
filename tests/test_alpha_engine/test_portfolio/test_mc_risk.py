"""Tests for Phase 4.5: Monte Carlo VaR/CVaR with Cholesky correlation."""
import numpy as np
import pandas as pd
import pytest

from alpha_engine.portfolio.mc_risk import mc_cvar, mc_var


@pytest.fixture
def returns():
    rng = np.random.default_rng(0)
    n, k = 500, 3
    data = rng.standard_normal((n, k)) * 0.01
    return pd.DataFrame(data, columns=["A", "B", "C"])


@pytest.fixture
def weights():
    return pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})


class TestMCVaR:
    def test_var_positive(self, returns, weights):
        var = mc_var(returns, weights, confidence=0.95, n_paths=10_000, seed=42)
        assert var > 0

    def test_var_99_gt_var_95(self, returns, weights):
        """99% VaR should be worse (larger) than 95% VaR."""
        v95 = mc_var(returns, weights, confidence=0.95, n_paths=50_000, seed=7)
        v99 = mc_var(returns, weights, confidence=0.99, n_paths=50_000, seed=7)
        assert v99 > v95

    def test_var_returns_float(self, returns, weights):
        v = mc_var(returns, weights, confidence=0.95, n_paths=1_000, seed=0)
        assert isinstance(v, float)

    def test_invalid_confidence_raises(self, returns, weights):
        with pytest.raises(ValueError, match="confidence"):
            mc_var(returns, weights, confidence=1.5, n_paths=1_000, seed=0)

    def test_var_increases_with_higher_vol(self, weights):
        """Higher volatility → higher VaR."""
        rng = np.random.default_rng(1)
        low = pd.DataFrame(rng.standard_normal((500, 3)) * 0.005, columns=["A", "B", "C"])
        high = pd.DataFrame(rng.standard_normal((500, 3)) * 0.02, columns=["A", "B", "C"])
        v_low = mc_var(low, weights, confidence=0.95, n_paths=50_000, seed=1)
        v_high = mc_var(high, weights, confidence=0.95, n_paths=50_000, seed=1)
        assert v_high > v_low


class TestMCCVaR:
    def test_cvar_ge_var(self, returns, weights):
        """CVaR (Expected Shortfall) >= VaR at same confidence level."""
        var = mc_var(returns, weights, confidence=0.95, n_paths=50_000, seed=42)
        cvar = mc_cvar(returns, weights, confidence=0.95, n_paths=50_000, seed=42)
        assert cvar >= var - 1e-6

    def test_cvar_positive(self, returns, weights):
        cvar = mc_cvar(returns, weights, confidence=0.95, n_paths=10_000, seed=0)
        assert cvar > 0

    def test_cvar_returns_float(self, returns, weights):
        cvar = mc_cvar(returns, weights, confidence=0.95, n_paths=1_000, seed=0)
        assert isinstance(cvar, float)

    def test_cvar_99_gt_cvar_95(self, returns, weights):
        c95 = mc_cvar(returns, weights, confidence=0.95, n_paths=50_000, seed=5)
        c99 = mc_cvar(returns, weights, confidence=0.99, n_paths=50_000, seed=5)
        assert c99 > c95

    def test_weights_mismatch_raises(self, returns):
        bad_w = pd.Series({"X": 0.5, "Y": 0.5})
        with pytest.raises((ValueError, KeyError)):
            mc_var(returns, bad_w, confidence=0.95, n_paths=100, seed=0)
