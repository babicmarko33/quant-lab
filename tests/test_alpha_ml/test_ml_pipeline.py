"""Tests for MLSignalPipeline — end-to-end OHLCV → signals → backtest."""

import numpy as np
import pandas as pd
import pytest

from alpha_engine.backtest.types import BacktestResult
from alpha_ml.models.xgboost_model import XGBoostPredictor
from alpha_ml.pipeline import MLSignalPipeline


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """3 years of synthetic daily OHLCV."""
    rng = np.random.default_rng(99)
    n = 3 * 252
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    log_ret = rng.normal(0.0003, 0.012, n)
    close = 100 * np.exp(np.cumsum(log_ret))
    opens = np.roll(close, 1)
    opens[0] = 100.0
    return pd.DataFrame(
        {
            "open": opens,
            "high": close * (1 + rng.uniform(0, 0.01, n)),
            "low": close * (1 - rng.uniform(0, 0.01, n)),
            "close": close,
            "volume": rng.integers(500_000, 2_000_000, n).astype(float),
        },
        index=dates,
    )


class TestMLSignalPipeline:
    def test_run_returns_backtest_result(self, sample_ohlcv: pd.DataFrame) -> None:
        pipeline = MLSignalPipeline(
            model=XGBoostPredictor(n_estimators=10, random_state=42),
            train_ratio=0.6,
        )
        result = pipeline.run(sample_ohlcv)
        assert isinstance(result, BacktestResult)

    def test_signals_have_correct_length(self, sample_ohlcv: pd.DataFrame) -> None:
        pipeline = MLSignalPipeline(
            model=XGBoostPredictor(n_estimators=10),
            train_ratio=0.6,
        )
        signals = pipeline.generate_signals(sample_ohlcv)
        assert len(signals) == len(sample_ohlcv)

    def test_train_period_has_zero_signals(self, sample_ohlcv: pd.DataFrame) -> None:
        """No trading during the training period — only OOS signals traded."""
        train_ratio = 0.6
        pipeline = MLSignalPipeline(
            model=XGBoostPredictor(n_estimators=10),
            train_ratio=train_ratio,
        )
        signals = pipeline.generate_signals(sample_ohlcv)
        train_end = int(len(sample_ohlcv) * train_ratio)
        train_signals = signals.iloc[:train_end]
        assert (train_signals == 0).all(), "Train period should have zero signals"

    def test_oos_period_has_nonzero_signals(self, sample_ohlcv: pd.DataFrame) -> None:
        """OOS period should have at least some non-zero signals."""
        pipeline = MLSignalPipeline(
            model=XGBoostPredictor(n_estimators=10),
            train_ratio=0.6,
        )
        signals = pipeline.generate_signals(sample_ohlcv)
        train_end = int(len(sample_ohlcv) * 0.6)
        oos_signals = signals.iloc[train_end:]
        assert (oos_signals != 0).any(), "OOS period has no signals"

    def test_signals_in_valid_range(self, sample_ohlcv: pd.DataFrame) -> None:
        pipeline = MLSignalPipeline(
            model=XGBoostPredictor(n_estimators=10),
            train_ratio=0.6,
        )
        signals = pipeline.generate_signals(sample_ohlcv)
        assert signals.isin([-1.0, 0.0, 1.0]).all()

    def test_backtest_result_not_nan(self, sample_ohlcv: pd.DataFrame) -> None:
        pipeline = MLSignalPipeline(
            model=XGBoostPredictor(n_estimators=10),
            train_ratio=0.6,
        )
        result = pipeline.run(sample_ohlcv)
        assert not np.isnan(result.total_return)
        assert not np.isnan(result.max_drawdown)
