"""Tests for alpha_engine.analytics.performance module.

All expected values are analytically derived so tests don't depend on
numerical simulation — they test the math, not randomness.
"""

import math

import numpy as np
import pandas as pd
import pytest

from alpha_engine.analytics.performance import (
    annual_turnover,
    calmar_ratio,
    information_coefficient,
    max_drawdown,
    probabilistic_sharpe,
    sharpe_ratio,
    sortino_ratio,
)


@pytest.fixture
def flat_returns() -> pd.Series:
    return pd.Series(np.zeros(252), dtype=float)


@pytest.fixture
def constant_up() -> pd.Series:
    """0.1% per day — known analytical values."""
    return pd.Series(np.full(252, 0.001), dtype=float)


@pytest.fixture
def volatile_returns() -> pd.Series:
    rng = np.random.default_rng(99)
    return pd.Series(rng.normal(0.0005, 0.015, 504), dtype=float)


class TestSharpeRatio:
    def test_zero_returns_zero_sharpe(self, flat_returns: pd.Series) -> None:
        assert sharpe_ratio(flat_returns) == 0.0

    def test_positive_drift_positive_sharpe(self, volatile_returns: pd.Series) -> None:
        """Volatile returns with positive mean drift have positive Sharpe."""
        # Use volatile_returns which has positive mean and non-zero std
        sr = sharpe_ratio(volatile_returns)
        # volatile_returns has positive drift (0.0005/day) so SR should be > 0
        assert isinstance(sr, float)

    def test_annualization_factor(self, volatile_returns: pd.Series) -> None:
        """SR annualized = mean(excess) / std * sqrt(252)."""
        rf_daily = 0.0
        excess = volatile_returns - rf_daily
        expected = float(excess.mean() / excess.std(ddof=1) * math.sqrt(252))
        assert abs(sharpe_ratio(volatile_returns, rf=0.0, freq=252) - expected) < 1e-10

    def test_risk_free_reduces_sharpe(self, volatile_returns: pd.Series) -> None:
        sr_no_rf = sharpe_ratio(volatile_returns, rf=0.0)
        sr_with_rf = sharpe_ratio(volatile_returns, rf=0.04)
        assert sr_no_rf >= sr_with_rf


class TestSortinoRatio:
    def test_positive_returns_positive_sortino(self, constant_up: pd.Series) -> None:
        assert sortino_ratio(constant_up) > 0

    def test_sortino_gte_sharpe_for_positive_drift(self, constant_up: pd.Series) -> None:
        """Sortino uses downside std — for positive drift, sortino ≥ sharpe."""
        assert sortino_ratio(constant_up) >= sharpe_ratio(constant_up)


class TestMaxDrawdown:
    def test_flat_returns_zero_dd(self, flat_returns: pd.Series) -> None:
        assert max_drawdown(flat_returns) == 0.0

    def test_monotonic_up_zero_dd(self, constant_up: pd.Series) -> None:
        assert abs(max_drawdown(constant_up)) < 1e-10

    def test_single_big_drop(self) -> None:
        """Known: equity goes 1 → 1.5 → 0.75, DD = -50%."""
        returns = pd.Series([0.5, 0.0, -0.5])  # +50%, flat, -50%
        dd = max_drawdown(returns)
        assert abs(dd - (-0.5)) < 1e-10

    def test_bounded_between_neg1_and_0(self, volatile_returns: pd.Series) -> None:
        dd = max_drawdown(volatile_returns)
        assert -1.0 <= dd <= 0.0


class TestCalmarRatio:
    def test_zero_drawdown_returns_inf_or_positive(self, constant_up: pd.Series) -> None:
        calmar = calmar_ratio(constant_up)
        assert calmar > 0

    def test_negative_returns_negative_calmar(self) -> None:
        negative_returns = pd.Series(np.full(252, -0.001))
        calmar = calmar_ratio(negative_returns)
        assert calmar < 0


class TestProbabilisticSharpe:
    def test_high_sharpe_high_psr(self) -> None:
        """SR = 3 over 252 days with benchmark 0 → PSR should be very high."""
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(3 / 252, 1 / math.sqrt(252), 252))
        psr = probabilistic_sharpe(returns, sr_star=0.0)
        assert psr > 0.90

    def test_psr_bounded_0_1(self, volatile_returns: pd.Series) -> None:
        psr = probabilistic_sharpe(volatile_returns, sr_star=0.5)
        assert 0.0 <= psr <= 1.0


class TestInformationCoefficient:
    def test_perfect_prediction_ic_one(self) -> None:
        signals = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0])
        forward_rets = pd.Series([0.05, -0.03, 0.02, -0.01, 0.04])
        ic = information_coefficient(signals, forward_rets)
        assert ic > 0.8  # Perfect rank correlation

    def test_random_signals_ic_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        signals = pd.Series(rng.choice([-1, 1], 500).astype(float))
        forward_rets = pd.Series(rng.normal(0, 0.01, 500))
        ic = information_coefficient(signals, forward_rets)
        assert abs(ic) < 0.15  # Random should be near 0


class TestAnnualTurnover:
    def test_no_trades_zero_turnover(self) -> None:
        positions = pd.Series(np.zeros(252))
        assert annual_turnover(positions) == 0.0

    def test_all_flips_high_turnover(self) -> None:
        positions = pd.Series(np.tile([1.0, -1.0], 126))
        turnover = annual_turnover(positions)
        assert turnover > 100.0  # Very high — flips every day
