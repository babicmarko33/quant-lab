"""Tests for quant_dashboard Plotly chart components."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from quant_dashboard.components.charts import (
    correlation_heatmap_fig,
    equity_curve_fig,
    feature_importance_fig,
    returns_distribution_fig,
    weights_bar_fig,
)


@pytest.fixture
def sample_equity() -> pd.Series:
    idx = pd.date_range("2023-01-01", periods=252, freq="B")
    return pd.Series(
        np.cumprod(1 + np.random.default_rng(0).normal(0.0005, 0.01, 252)), index=idx
    )


@pytest.fixture
def sample_returns(sample_equity: pd.Series) -> pd.Series:
    return sample_equity.pct_change().dropna()


class TestEquityCurveFig:
    def test_returns_go_figure(self, sample_equity: pd.Series) -> None:
        fig = equity_curve_fig(sample_equity, title="Test")
        assert isinstance(fig, go.Figure)

    def test_has_equity_trace(self, sample_equity: pd.Series) -> None:
        fig = equity_curve_fig(sample_equity, title="Test")
        trace_names = [t.name for t in fig.data]
        assert any("equity" in (name or "").lower() for name in trace_names)

    def test_has_drawdown_trace(self, sample_equity: pd.Series) -> None:
        fig = equity_curve_fig(sample_equity, title="Test")
        trace_names = [t.name for t in fig.data]
        assert any("drawdown" in (name or "").lower() for name in trace_names)

    def test_title_set(self, sample_equity: pd.Series) -> None:
        fig = equity_curve_fig(sample_equity, title="My Strategy")
        assert "My Strategy" in fig.layout.title.text


class TestReturnsDistributionFig:
    def test_returns_go_figure(self, sample_returns: pd.Series) -> None:
        fig = returns_distribution_fig(sample_returns)
        assert isinstance(fig, go.Figure)

    def test_has_histogram_trace(self, sample_returns: pd.Series) -> None:
        fig = returns_distribution_fig(sample_returns)
        assert any(isinstance(t, go.Histogram) for t in fig.data)


class TestWeightsBarFig:
    def test_returns_go_figure(self) -> None:
        weights = pd.Series({"SPY": 0.5, "TLT": 0.3, "GLD": 0.2})
        fig = weights_bar_fig(weights, "Weights")
        assert isinstance(fig, go.Figure)

    def test_has_bar_trace(self) -> None:
        weights = pd.Series({"SPY": 0.5, "TLT": 0.3, "GLD": 0.2})
        fig = weights_bar_fig(weights, "Weights")
        assert len(fig.data) > 0
        assert isinstance(fig.data[0], go.Bar)


class TestCorrelationHeatmapFig:
    def test_returns_go_figure(self) -> None:
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.normal(0, 0.01, (100, 3)), columns=["SPY", "TLT", "GLD"])
        fig = correlation_heatmap_fig(df)
        assert isinstance(fig, go.Figure)

    def test_has_heatmap_trace(self) -> None:
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.normal(0, 0.01, (100, 3)), columns=["SPY", "TLT", "GLD"])
        fig = correlation_heatmap_fig(df)
        assert any(isinstance(t, go.Heatmap) for t in fig.data)


class TestFeatureImportanceFig:
    def test_returns_go_figure(self) -> None:
        importance = pd.Series({"rsi": 0.4, "macd": 0.35, "bb_pct": 0.25})
        fig = feature_importance_fig(importance)
        assert isinstance(fig, go.Figure)

    def test_bars_sorted_descending(self) -> None:
        importance = pd.Series({"rsi": 0.4, "macd": 0.35, "bb_pct": 0.25})
        fig = feature_importance_fig(importance)
        bar_trace = fig.data[0]
        # horizontal bar: x values should be ascending (lowest at top)
        assert bar_trace.x[0] <= bar_trace.x[-1]
