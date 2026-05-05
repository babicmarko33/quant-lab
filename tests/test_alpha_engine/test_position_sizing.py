"""Tests for risk/position sizing functions."""

import numpy as np
import pandas as pd

from alpha_engine.risk.position_sizing import (
    fractional_kelly,
    kelly_fraction,
    volatility_target_size,
)


class TestKellyFraction:
    def test_positive_edge_positive_kelly(self) -> None:
        """Edge > 0 should return positive Kelly fraction."""
        k = kelly_fraction(win_rate=0.55, win_loss_ratio=1.5)
        assert k > 0

    def test_fifty_fifty_even_odds_zero_kelly(self) -> None:
        """win_rate=0.5, win_loss=1.0 → exact break-even → Kelly = 0."""
        k = kelly_fraction(win_rate=0.5, win_loss_ratio=1.0)
        assert abs(k) < 1e-10

    def test_negative_edge_negative_or_zero(self) -> None:
        """Negative expected value → Kelly should be ≤ 0 (no bet)."""
        k = kelly_fraction(win_rate=0.4, win_loss_ratio=1.0)
        assert k <= 0

    def test_kelly_formula(self) -> None:
        """Verify Kelly = W - (1-W)/R formula exactly."""
        w, r = 0.6, 2.0
        expected = w - (1 - w) / r
        assert abs(kelly_fraction(w, r) - expected) < 1e-12

    def test_clipped_at_one(self) -> None:
        """Kelly fraction should be clipped to [0, 1]."""
        k = kelly_fraction(win_rate=0.99, win_loss_ratio=100.0)
        assert k <= 1.0


class TestFractionalKelly:
    def test_half_kelly(self) -> None:
        """Half Kelly = 0.5 × full Kelly."""
        full = kelly_fraction(0.6, 2.0)
        half = fractional_kelly(win_rate=0.6, win_loss_ratio=2.0, fraction=0.5)
        assert abs(half - 0.5 * full) < 1e-12

    def test_zero_fraction_zero_size(self) -> None:
        f = fractional_kelly(win_rate=0.6, win_loss_ratio=2.0, fraction=0.0)
        assert f == 0.0

    def test_negative_kelly_returns_zero(self) -> None:
        """When edge is negative, fractional kelly should be 0 (never short via Kelly)."""
        f = fractional_kelly(win_rate=0.3, win_loss_ratio=1.0, fraction=0.5)
        assert f == 0.0


class TestVolatilityTargetSize:
    def test_higher_vol_smaller_size(self) -> None:
        """Higher volatility → smaller position size (same target)."""
        rng = np.random.default_rng(1)
        # low_vol: daily std ≈ 0.005 → annual ≈ 7.9%
        low_vol = pd.Series(rng.normal(0.0, 0.005, 60))
        # high_vol: daily std ≈ 0.030 → annual ≈ 47.6%
        high_vol = pd.Series(rng.normal(0.0, 0.030, 60))
        size_low = volatility_target_size(low_vol, vol_target=0.10)
        size_high = volatility_target_size(high_vol, vol_target=0.10)
        # high_vol annual > vol_target=10% → not capped; low_vol may cap at 1.0
        assert size_low >= size_high

    def test_target_vol_relationship(self) -> None:
        """Position size should be approximately vol_target / realized_vol."""
        returns = pd.Series(np.full(60, 0.01))
        realized_vol = float(returns.std(ddof=1) * np.sqrt(252))
        vol_target = 0.20
        size = volatility_target_size(returns, vol_target=vol_target)
        expected = vol_target / realized_vol
        assert abs(size - min(expected, 1.0)) < 0.01

    def test_capped_at_one(self) -> None:
        """Position size is capped at 1.0 (no leverage beyond fully invested)."""
        very_low_vol = pd.Series(np.full(60, 0.0001))
        size = volatility_target_size(very_low_vol, vol_target=0.20)
        assert size <= 1.0

    def test_insufficient_data_returns_zero(self) -> None:
        """Too few observations → return 0 (no sizing without data)."""
        short_returns = pd.Series([0.01, 0.02])
        size = volatility_target_size(short_returns, vol_target=0.10)
        assert size == 0.0
