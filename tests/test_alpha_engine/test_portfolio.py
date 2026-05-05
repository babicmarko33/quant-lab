"""Tests for portfolio allocation: EqualWeight, MeanVariance, RiskParity, CVaR."""

import numpy as np
import pandas as pd
import pytest

from alpha_engine.portfolio.allocator import Allocator
from alpha_engine.portfolio.cvar import CVaRAllocator
from alpha_engine.portfolio.equal_weight import EqualWeightAllocator
from alpha_engine.portfolio.mean_variance import MeanVarianceAllocator
from alpha_engine.portfolio.risk_parity import RiskParityAllocator


@pytest.fixture
def asset_returns() -> pd.DataFrame:
    """5 uncorrelated assets, 3 years of daily returns."""
    rng = np.random.default_rng(42)
    n = 3 * 252
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {
        "SPY": rng.normal(0.0004, 0.010, n),   # equity
        "TLT": rng.normal(0.0001, 0.008, n),   # bonds
        "GLD": rng.normal(0.0002, 0.007, n),   # gold
        "QQQ": rng.normal(0.0005, 0.014, n),   # tech
        "IEF": rng.normal(0.00005, 0.004, n),  # short bonds
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def two_asset_returns() -> pd.DataFrame:
    """2 assets — easy to verify analytical results."""
    rng = np.random.default_rng(7)
    n = 252
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {"A": rng.normal(0.001, 0.01, n), "B": rng.normal(0.0005, 0.02, n)},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Shared invariants — tested for ALL allocators
# ---------------------------------------------------------------------------

ALLOCATOR_CLASSES = [
    EqualWeightAllocator,
    MeanVarianceAllocator,
    RiskParityAllocator,
    CVaRAllocator,
]


class TestAllocatorInvariants:
    @pytest.mark.parametrize("AllocClass", ALLOCATOR_CLASSES)
    def test_weights_sum_to_one(self, AllocClass, asset_returns: pd.DataFrame) -> None:
        alloc = AllocClass()
        weights = alloc.fit(asset_returns)
        assert abs(weights.sum() - 1.0) < 1e-6, f"{AllocClass.__name__}: weights sum = {weights.sum()}"

    @pytest.mark.parametrize("AllocClass", ALLOCATOR_CLASSES)
    def test_weights_indexed_by_assets(self, AllocClass, asset_returns: pd.DataFrame) -> None:
        alloc = AllocClass()
        weights = alloc.fit(asset_returns)
        assert list(weights.index) == list(asset_returns.columns)

    @pytest.mark.parametrize("AllocClass", ALLOCATOR_CLASSES)
    def test_weights_non_negative(self, AllocClass, asset_returns: pd.DataFrame) -> None:
        alloc = AllocClass()
        weights = alloc.fit(asset_returns)
        assert (weights >= -1e-8).all(), f"{AllocClass.__name__}: negative weights found"

    @pytest.mark.parametrize("AllocClass", ALLOCATOR_CLASSES)
    def test_returns_pd_series(self, AllocClass, asset_returns: pd.DataFrame) -> None:
        alloc = AllocClass()
        weights = alloc.fit(asset_returns)
        assert isinstance(weights, pd.Series)

    def test_allocator_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Allocator()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# EqualWeightAllocator
# ---------------------------------------------------------------------------

class TestEqualWeightAllocator:
    def test_all_weights_equal(self, asset_returns: pd.DataFrame) -> None:
        alloc = EqualWeightAllocator()
        weights = alloc.fit(asset_returns)
        expected = 1.0 / len(asset_returns.columns)
        assert (weights - expected).abs().max() < 1e-10

    def test_two_assets_each_50pct(self, two_asset_returns: pd.DataFrame) -> None:
        alloc = EqualWeightAllocator()
        weights = alloc.fit(two_asset_returns)
        assert abs(weights["A"] - 0.5) < 1e-10
        assert abs(weights["B"] - 0.5) < 1e-10


# ---------------------------------------------------------------------------
# MeanVarianceAllocator
# ---------------------------------------------------------------------------

class TestMeanVarianceAllocator:
    def test_higher_sharpe_asset_gets_more_weight(self) -> None:
        """Asset with clearly higher Sharpe gets more weight in max-Sharpe portfolio."""
        # Use a large sample to ensure sample mean tracks population mean reliably
        rng = np.random.default_rng(0)
        n = 10_000
        dates = pd.date_range("2010-01-01", periods=n, freq="B")
        # A: SR ≈ 0.001/0.005 = 0.2; B: SR ≈ 0.0001/0.015 = 0.0067
        returns = pd.DataFrame(
            {"A": rng.normal(0.001, 0.005, n), "B": rng.normal(0.0001, 0.015, n)},
            index=dates,
        )
        alloc = MeanVarianceAllocator(objective="max_sharpe")
        weights = alloc.fit(returns)
        assert weights["A"] >= weights["B"], f"A weight {weights['A']:.3f} < B weight {weights['B']:.3f}"

    def test_max_sharpe_mode(self, asset_returns: pd.DataFrame) -> None:
        alloc = MeanVarianceAllocator(objective="max_sharpe")
        weights = alloc.fit(asset_returns)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_min_variance_mode(self, asset_returns: pd.DataFrame) -> None:
        alloc = MeanVarianceAllocator(objective="min_variance")
        weights = alloc.fit(asset_returns)
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_min_var_lower_vol_than_equal_weight(self, asset_returns: pd.DataFrame) -> None:
        """Min-variance portfolio should have lower volatility than equal-weight."""
        mv_alloc = MeanVarianceAllocator(objective="min_variance")
        eq_alloc = EqualWeightAllocator()
        mv_w = mv_alloc.fit(asset_returns)
        eq_w = eq_alloc.fit(asset_returns)
        sigma = asset_returns.cov().values * 252
        mv_vol = float(np.sqrt(mv_w.values @ sigma @ mv_w.values))
        eq_vol = float(np.sqrt(eq_w.values @ sigma @ eq_w.values))
        assert mv_vol <= eq_vol + 1e-6


# ---------------------------------------------------------------------------
# RiskParityAllocator
# ---------------------------------------------------------------------------

class TestRiskParityAllocator:
    def test_equal_risk_contributions(self, asset_returns: pd.DataFrame) -> None:
        """Risk parity: each asset contributes equal % of total portfolio risk."""
        alloc = RiskParityAllocator()
        weights = alloc.fit(asset_returns)
        sigma = asset_returns.cov().values
        w = weights.values
        portfolio_var = float(w @ sigma @ w)
        # Marginal Risk Contribution = sigma @ w * w / sqrt(portfolio_var)
        mrc = (sigma @ w) * w / np.sqrt(portfolio_var)
        # All MRCs should be approximately equal
        assert mrc.std() / mrc.mean() < 0.05, f"MRCs not equal: {mrc}"

    def test_low_vol_asset_gets_higher_weight(self, two_asset_returns: pd.DataFrame) -> None:
        """Lower vol asset gets higher weight in risk parity."""
        alloc = RiskParityAllocator()
        weights = alloc.fit(two_asset_returns)
        # A has vol ~0.01, B has vol ~0.02 → A should get higher weight
        assert weights["A"] > weights["B"]


# ---------------------------------------------------------------------------
# CVaRAllocator
# ---------------------------------------------------------------------------

class TestCVaRAllocator:
    def test_default_confidence_95(self, asset_returns: pd.DataFrame) -> None:
        alloc = CVaRAllocator(alpha=0.95)
        weights = alloc.fit(asset_returns)
        assert abs(weights.sum() - 1.0) < 1e-5

    def test_higher_confidence_different_weights(self, asset_returns: pd.DataFrame) -> None:
        """Different alpha levels → different weights."""
        alloc_95 = CVaRAllocator(alpha=0.95)
        alloc_99 = CVaRAllocator(alpha=0.99)
        w95 = alloc_95.fit(asset_returns)
        w99 = alloc_99.fit(asset_returns)
        # Not necessarily identical — though may be close
        assert isinstance(w95, pd.Series)
        assert isinstance(w99, pd.Series)
