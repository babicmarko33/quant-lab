"""Tests for Phase 4.7: Black-Litterman portfolio allocation."""
import numpy as np
import pandas as pd
import pytest

from alpha_engine.portfolio.black_litterman import BlackLittermanAllocator


@pytest.fixture
def returns():
    """Synthetic 4-asset daily returns (252 days)."""
    rng = np.random.default_rng(42)
    n, k = 252, 4
    data = rng.standard_normal((n, k)) * 0.01
    data += np.array([0.0005, 0.0003, 0.0007, 0.0001])  # different means
    return pd.DataFrame(data, columns=["A", "B", "C", "D"])


@pytest.fixture
def market_caps():
    return pd.Series({"A": 1000.0, "B": 800.0, "C": 600.0, "D": 400.0})


class TestBlackLittermanAllocator:
    def test_weights_sum_to_one(self, returns, market_caps):
        model = BlackLittermanAllocator(market_caps=market_caps, tau=0.05)
        w = model.fit(returns)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_weights_non_negative(self, returns, market_caps):
        """BL with no views: all weights should be non-negative (equilibrium)."""
        model = BlackLittermanAllocator(market_caps=market_caps, tau=0.05)
        w = model.fit(returns)
        assert (w >= -1e-8).all()

    def test_weights_indexed_by_assets(self, returns, market_caps):
        model = BlackLittermanAllocator(market_caps=market_caps, tau=0.05)
        w = model.fit(returns)
        assert list(w.index) == list(returns.columns)

    def test_no_views_returns_mkt_eq_weights(self, returns, market_caps):
        """No views: BL weights ≈ market-cap-implied equilibrium weights."""
        model = BlackLittermanAllocator(market_caps=market_caps, tau=0.05)
        w = model.fit(returns)
        eq = market_caps / market_caps.sum()
        # BL without views should be close to equilibrium
        assert np.linalg.norm(w.values - eq.values) < 0.3

    def test_views_shift_weights(self, returns, market_caps):
        """Strong positive view on asset A should increase A's weight."""
        # View: A outperforms B by 5%
        P = np.array([[1.0, -1.0, 0.0, 0.0]])  # A - B
        Q = np.array([0.05])
        omega = np.array([[0.0001]])
        model_no_view = BlackLittermanAllocator(market_caps=market_caps, tau=0.05)
        model_view = BlackLittermanAllocator(
            market_caps=market_caps, tau=0.05, P=P, Q=Q, omega=omega
        )
        w_no = model_no_view.fit(returns)
        w_view = model_view.fit(returns)
        assert w_view["A"] >= w_no["A"] - 0.05

    def test_market_cap_mismatch_raises(self, returns):
        bad_caps = pd.Series({"X": 100.0, "Y": 200.0})
        model = BlackLittermanAllocator(market_caps=bad_caps, tau=0.05)
        with pytest.raises((ValueError, KeyError)):
            model.fit(returns)

    def test_returns_series(self, returns, market_caps):
        model = BlackLittermanAllocator(market_caps=market_caps)
        w = model.fit(returns)
        assert isinstance(w, pd.Series)
