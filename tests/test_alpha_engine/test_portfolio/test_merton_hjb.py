"""Tests for Phase 4.6: Merton's Continuous-Time Portfolio Problem (HJB).

The Merton problem: find the optimal fraction of wealth to invest in a risky
asset to maximise expected CRRA utility of terminal wealth.

Analytical solution (power utility U(W) = W^γ / γ):
    w* = (μ - r) / (σ² * (1 - γ))

where μ = asset drift, r = risk-free rate, σ = volatility, γ = risk aversion
(γ < 1, γ ≠ 0 for power utility).

merton_optimal_weight(mu, r, sigma, gamma) → float
merton_value_function(W, T, mu, r, sigma, gamma) → float
"""

import pytest

from alpha_engine.portfolio.merton_hjb import merton_optimal_weight, merton_value_function


class TestMertonOptimalWeight:
    def test_basic_case(self):
        """w* = (μ-r) / (σ²*(1-γ)) for power utility."""
        mu, r, sigma, gamma = 0.10, 0.03, 0.20, 0.5
        w = merton_optimal_weight(mu=mu, r=r, sigma=sigma, gamma=gamma)
        expected = (mu - r) / (sigma**2 * (1 - gamma))
        assert abs(w - expected) < 1e-10

    def test_risk_averse_invests_less(self):
        """More risk-averse (lower γ → higher 1-γ) invests a smaller fraction."""
        mu, r, sigma = 0.10, 0.03, 0.20
        w_low_ra = merton_optimal_weight(mu=mu, r=r, sigma=sigma, gamma=0.8)   # less risk-averse
        w_high_ra = merton_optimal_weight(mu=mu, r=r, sigma=sigma, gamma=0.2)  # more risk-averse
        assert w_low_ra > w_high_ra

    def test_zero_excess_return_zero_weight(self):
        """μ = r → no incentive to invest in risky asset."""
        w = merton_optimal_weight(mu=0.05, r=0.05, sigma=0.20, gamma=0.5)
        assert abs(w) < 1e-10

    def test_returns_float(self):
        w = merton_optimal_weight(mu=0.10, r=0.03, sigma=0.20, gamma=0.5)
        assert isinstance(w, float)

    def test_negative_excess_return_short(self):
        """μ < r → optimal to short the risky asset."""
        w = merton_optimal_weight(mu=0.02, r=0.05, sigma=0.20, gamma=0.5)
        assert w < 0

    def test_invalid_gamma_zero_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            merton_optimal_weight(mu=0.10, r=0.03, sigma=0.20, gamma=0.0)

    def test_invalid_gamma_ge_one_raises(self):
        """γ >= 1 is outside power utility domain (log utility is γ→0)."""
        with pytest.raises(ValueError, match="gamma"):
            merton_optimal_weight(mu=0.10, r=0.03, sigma=0.20, gamma=1.0)

    def test_invalid_sigma_raises(self):
        with pytest.raises(ValueError, match="sigma"):
            merton_optimal_weight(mu=0.10, r=0.03, sigma=0.0, gamma=0.5)


class TestMertonValueFunction:
    def test_value_function_positive(self):
        """Value function V(W, T) > 0 for positive wealth."""
        v = merton_value_function(W=1.0, T=1.0, mu=0.10, r=0.03, sigma=0.20, gamma=0.5)
        assert v > 0

    def test_value_function_scales_with_wealth(self):
        """CRRA utility: V(λW) = λ^γ * V(W)."""
        params = dict(T=1.0, mu=0.10, r=0.03, sigma=0.20, gamma=0.5)
        lam = 2.0
        v1 = merton_value_function(W=1.0, **params)
        v2 = merton_value_function(W=lam, **params)
        assert abs(v2 / v1 - lam**0.5) < 1e-6

    def test_value_function_increases_with_wealth(self):
        params = dict(T=1.0, mu=0.10, r=0.03, sigma=0.20, gamma=0.5)
        v1 = merton_value_function(W=1.0, **params)
        v2 = merton_value_function(W=2.0, **params)
        assert v2 > v1

    def test_value_function_increases_with_sharpe(self):
        """Higher Sharpe ratio (same γ) → higher value (better investment opportunity)."""
        base = dict(W=1.0, T=1.0, r=0.03, sigma=0.20, gamma=0.5)
        v_low = merton_value_function(mu=0.05, **base)   # low excess return
        v_high = merton_value_function(mu=0.15, **base)  # high excess return
        assert v_high > v_low

    def test_returns_float(self):
        v = merton_value_function(W=1.0, T=1.0, mu=0.10, r=0.03, sigma=0.20, gamma=0.5)
        assert isinstance(v, float)
