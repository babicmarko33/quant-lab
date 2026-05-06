"""Tests for cointegration detection module. Written before implementation (TDD RED)."""
import numpy as np
import pandas as pd
import pytest

from quantcore.signals.cointegration import (
    CointegrationResult,
    engle_granger_test,
    find_cointegrated_pairs,
    hedge_ratio,
)

N = 500


@pytest.fixture
def cointegrated_pair() -> tuple[pd.Series, pd.Series]:
    """Genuine cointegrated pair: y = 1.5*x + noise with shared random walk."""
    rng = np.random.default_rng(42)
    x = pd.Series(np.cumsum(rng.standard_normal(N)) + 100.0, name="X")
    y = pd.Series(1.5 * x + rng.standard_normal(N) * 2.0, name="Y")
    return x, y


@pytest.fixture
def non_cointegrated_pair() -> tuple[pd.Series, pd.Series]:
    """Two independent random walks — should NOT be cointegrated."""
    rng = np.random.default_rng(99)
    x = pd.Series(np.cumsum(rng.standard_normal(N)) + 100.0, name="A")
    y = pd.Series(np.cumsum(rng.standard_normal(N)) + 100.0, name="B")
    return x, y


class TestEngleGrangerTest:
    def test_returns_result(self, cointegrated_pair):
        x, y = cointegrated_pair
        result = engle_granger_test(x, y)
        assert isinstance(result, CointegrationResult)

    def test_cointegrated_pair_low_pvalue(self, cointegrated_pair):
        x, y = cointegrated_pair
        result = engle_granger_test(x, y)
        assert result.p_value < 0.05, f"Expected cointegration, p={result.p_value:.4f}"

    def test_cointegrated_pair_is_cointegrated(self, cointegrated_pair):
        x, y = cointegrated_pair
        result = engle_granger_test(x, y)
        assert result.is_cointegrated

    def test_non_cointegrated_pair_high_pvalue(self, non_cointegrated_pair):
        x, y = non_cointegrated_pair
        result = engle_granger_test(x, y)
        # Two independent random walks are rarely cointegrated
        assert result.p_value > 0.05 or not result.is_cointegrated

    def test_hedge_ratio_stored(self, cointegrated_pair):
        x, y = cointegrated_pair
        result = engle_granger_test(x, y)
        assert result.hedge_ratio > 0

    def test_raises_on_mismatched_lengths(self, cointegrated_pair):
        x, y = cointegrated_pair
        with pytest.raises(ValueError, match="same length"):
            engle_granger_test(x, y.iloc[:-10])

    def test_raises_on_too_short(self):
        short = pd.Series(np.ones(20))
        with pytest.raises(ValueError, match="at least 30"):
            engle_granger_test(short, short)


class TestHedgeRatio:
    def test_ols_hedge_ratio_close_to_true(self, cointegrated_pair):
        """OLS hedge ratio should be close to the true ratio of 1.5."""
        x, y = cointegrated_pair
        hr = hedge_ratio(x, y)
        assert abs(hr - 1.5) < 0.2, f"Expected ~1.5, got {hr:.4f}"

    def test_positive_for_cointegrated(self, cointegrated_pair):
        x, y = cointegrated_pair
        assert hedge_ratio(x, y) > 0


class TestFindCointegratedPairs:
    def test_returns_dataframe(self, cointegrated_pair):
        x, y = cointegrated_pair
        prices = pd.DataFrame({"X": x, "Y": y})
        result = find_cointegrated_pairs(prices)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self, cointegrated_pair):
        x, y = cointegrated_pair
        prices = pd.DataFrame({"X": x, "Y": y})
        result = find_cointegrated_pairs(prices)
        assert {"asset_1", "asset_2", "p_value", "hedge_ratio", "is_cointegrated"}.issubset(
            result.columns
        )

    def test_finds_known_pair(self, cointegrated_pair):
        x, y = cointegrated_pair
        prices = pd.DataFrame({"X": x, "Y": y})
        result = find_cointegrated_pairs(prices, max_p_value=0.05)
        cointegrated = result[result["is_cointegrated"]]
        assert len(cointegrated) >= 1

    def test_three_series_produces_three_pairs(self, cointegrated_pair):
        """3 tickers → 3 pair combinations."""
        rng = np.random.default_rng(7)
        x, y = cointegrated_pair
        z = pd.Series(np.cumsum(rng.standard_normal(N)) + 50.0, name="Z")
        prices = pd.DataFrame({"X": x, "Y": y, "Z": z})
        result = find_cointegrated_pairs(prices)
        assert len(result) == 3
