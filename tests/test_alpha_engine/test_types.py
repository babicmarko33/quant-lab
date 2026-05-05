"""Tests for BacktestResult dataclass."""

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest.types import BacktestResult


@pytest.fixture
def flat_returns() -> pd.Series:
    """Zero returns — equity stays at 1.0."""
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    return pd.Series(np.zeros(252), index=dates)


@pytest.fixture
def bull_returns() -> pd.Series:
    """Trending up returns ~25% annual."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    return pd.Series(rng.normal(0.001, 0.01, 252), index=dates)


@pytest.fixture
def crash_returns() -> pd.Series:
    """Big drawdown mid-year."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    r = rng.normal(0.0002, 0.008, 252)
    r[100:120] = -0.03  # Crash: -30% over 20 days
    return pd.Series(r, index=dates)


class TestBacktestResult:
    def test_fields_present(self, bull_returns: pd.Series) -> None:
        result = BacktestResult(returns=bull_returns)
        assert result.returns is not None
        assert result.equity_curve is not None
        assert isinstance(result.total_return, float)
        assert isinstance(result.sharpe, float)
        assert isinstance(result.max_drawdown, float)
        assert isinstance(result.calmar, float)
        assert isinstance(result.n_trades, int)

    def test_equity_curve_starts_at_one(self, bull_returns: pd.Series) -> None:
        result = BacktestResult(returns=bull_returns)
        assert abs(result.equity_curve.iloc[0] - 1.0) < 1e-10

    def test_zero_returns_flat_equity(self, flat_returns: pd.Series) -> None:
        result = BacktestResult(returns=flat_returns)
        assert abs(result.total_return) < 1e-10
        assert abs(result.max_drawdown) < 1e-10

    def test_positive_returns_positive_total(self, bull_returns: pd.Series) -> None:
        result = BacktestResult(returns=bull_returns)
        assert result.total_return > 0

    def test_max_drawdown_negative(self, crash_returns: pd.Series) -> None:
        """Max drawdown must be negative (loss from peak)."""
        result = BacktestResult(returns=crash_returns)
        assert result.max_drawdown < 0

    def test_max_drawdown_bounded(self, crash_returns: pd.Series) -> None:
        """Max drawdown is bounded: -1 ≤ DD ≤ 0."""
        result = BacktestResult(returns=crash_returns)
        assert -1.0 <= result.max_drawdown <= 0.0

    def test_sharpe_positive_for_bull(self, bull_returns: pd.Series) -> None:
        result = BacktestResult(returns=bull_returns)
        assert result.sharpe > 0

    def test_repr_contains_key_info(self, bull_returns: pd.Series) -> None:
        result = BacktestResult(returns=bull_returns)
        r = repr(result)
        assert "sharpe" in r.lower() or "BacktestResult" in r
