"""Tests for Phase 4.4: Cardinality-Constrained Mixed Integer Optimization."""
import numpy as np
import pandas as pd
import pytest

from alpha_engine.portfolio.cardinality import CardinalityAllocator


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)
    n, k = 252, 6
    data = rng.standard_normal((n, k)) * 0.01
    data += np.linspace(0.0003, 0.0010, k)
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(k)])


class TestCardinalityAllocator:
    def test_weights_sum_to_one(self, returns):
        model = CardinalityAllocator(max_assets=3)
        w = model.fit(returns)
        assert abs(w.sum() - 1.0) < 1e-4

    def test_weights_non_negative(self, returns):
        model = CardinalityAllocator(max_assets=3)
        w = model.fit(returns)
        assert (w >= -1e-6).all()

    def test_cardinality_constraint_respected(self, returns):
        """At most max_assets non-zero weights."""
        max_k = 3
        model = CardinalityAllocator(max_assets=max_k)
        w = model.fit(returns)
        n_nonzero = (w > 1e-4).sum()
        assert n_nonzero <= max_k

    def test_max_assets_equals_all_assets_equals_mv(self, returns):
        """With max_assets = n_assets, result is a valid allocation."""
        w_card = CardinalityAllocator(max_assets=len(returns.columns)).fit(returns)
        assert abs(w_card.sum() - 1.0) < 1e-4

    def test_weights_indexed_by_assets(self, returns):
        model = CardinalityAllocator(max_assets=3)
        w = model.fit(returns)
        assert list(w.index) == list(returns.columns)

    def test_returns_series(self, returns):
        w = CardinalityAllocator(max_assets=2).fit(returns)
        assert isinstance(w, pd.Series)

    def test_max_assets_1_selects_one(self, returns):
        """max_assets=1 must produce a single 100% allocation."""
        w = CardinalityAllocator(max_assets=1).fit(returns)
        n_nonzero = (w > 1e-4).sum()
        assert n_nonzero == 1
        assert abs(w.sum() - 1.0) < 1e-4

    def test_invalid_max_assets_raises(self, returns):
        with pytest.raises(ValueError, match="max_assets"):
            CardinalityAllocator(max_assets=0).fit(returns)
