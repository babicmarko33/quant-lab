import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest.compare import ComparisonResult, compare_strategies
from alpha_engine.strategies.rsi_strategy import RSIMeanReversionStrategy
from alpha_engine.strategies.sma_crossover import SMACrossoverStrategy


@pytest.fixture
def price_df() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    n = 400
    trend = np.linspace(100, 180, n)
    noise = rng.normal(0, 1.0, n)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(trend + noise, index=idx)
    return pd.DataFrame({"open": close * 0.998, "close": close}, index=idx)


@pytest.fixture
def strategies():
    return [
        SMACrossoverStrategy(fast=10, slow=40),
        RSIMeanReversionStrategy(window=14),
    ]


class TestCompareStrategies:
    def test_returns_comparison_result(self, price_df, strategies):
        result = compare_strategies(strategies, price_df)
        assert isinstance(result, ComparisonResult)

    def test_result_has_all_strategies(self, price_df, strategies):
        result = compare_strategies(strategies, price_df)
        names = [s.name for s in strategies]
        for name in names:
            assert name in result.summary.index

    def test_summary_has_expected_metrics(self, price_df, strategies):
        result = compare_strategies(strategies, price_df)
        required = {"sharpe", "total_return", "max_drawdown", "calmar"}
        assert required.issubset(set(result.summary.columns))

    def test_best_sharpe_is_valid_name(self, price_df, strategies):
        result = compare_strategies(strategies, price_df)
        names = [s.name for s in strategies]
        assert result.best_sharpe in names

    def test_results_dict_keys_match_strategies(self, price_df, strategies):
        result = compare_strategies(strategies, price_df)
        for s in strategies:
            assert s.name in result.results

    def test_single_strategy_works(self, price_df):
        result = compare_strategies([SMACrossoverStrategy(fast=10, slow=40)], price_df)
        assert len(result.summary) == 1

    def test_summary_is_dataframe(self, price_df, strategies):
        result = compare_strategies(strategies, price_df)
        assert isinstance(result.summary, pd.DataFrame)
