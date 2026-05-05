"""Pure Plotly chart factory functions for the quant-lab dashboard.

All functions are side-effect free — they take data and return a go.Figure.
This design makes them fully testable without a running Streamlit server.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def equity_curve_fig(equity: pd.Series, title: str = "Equity Curve") -> go.Figure:
    """Line chart of equity curve with drawdown subplot.

    Parameters
    ----------
    equity : pd.Series
        Cumulative equity values (not returns). DatetimeIndex preferred.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
        Two-row Plotly figure: equity (top) and drawdown (bottom).
    """
    rolling_max = equity.cummax()
    drawdown = (equity / rolling_max) - 1

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )
    fig.add_trace(
        go.Scatter(x=equity.index, y=equity.values, name="Equity", line=dict(color="#00B4D8")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=drawdown.index, y=drawdown.values,
            name="Drawdown",
            fill="tozeroy",
            fillcolor="rgba(220,50,50,0.2)",
            line=dict(color="rgba(220,50,50,0.8)"),
        ),
        row=2, col=1,
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=500,
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=50, b=20),
    )
    return fig


def returns_distribution_fig(returns: pd.Series, confidence: float = 0.95) -> go.Figure:
    """Histogram of daily returns with VaR vertical line.

    Parameters
    ----------
    returns : pd.Series
        Daily arithmetic returns.
    confidence : float
        Confidence level for VaR annotation. Default 0.95.

    Returns
    -------
    go.Figure
    """
    var = float(np.percentile(returns.dropna(), (1 - confidence) * 100))
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=returns.values, nbinsx=60, name="Returns", marker_color="#90E0EF", opacity=0.75)
    )
    fig.add_vline(
        x=var,
        line_dash="dash",
        line_color="red",
        annotation_text=f"VaR {confidence:.0%}: {var:.2%}",
    )
    fig.update_layout(
        title="Return Distribution",
        template="plotly_dark",
        xaxis_title="Daily Return",
        yaxis_title="Count",
        margin=dict(l=40, r=20, t=50, b=20),
    )
    return fig


def weights_bar_fig(weights: pd.Series, title: str = "Portfolio Weights") -> go.Figure:
    """Horizontal bar chart of portfolio weights.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by asset name.
    title : str
        Chart title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure(
        go.Bar(
            x=weights.values,
            y=weights.index.tolist(),
            orientation="h",
            marker_color="#00B4D8",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_tickformat=".0%",
        margin=dict(l=40, r=20, t=50, b=20),
    )
    return fig


def correlation_heatmap_fig(returns_df: pd.DataFrame) -> go.Figure:
    """Correlation matrix heatmap.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Daily returns with assets as columns.

    Returns
    -------
    go.Figure
    """
    corr = returns_df.corr()
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Return Correlation",
        template="plotly_dark",
        margin=dict(l=60, r=20, t=50, b=20),
    )
    return fig


def feature_importance_fig(importance: pd.Series) -> go.Figure:
    """Horizontal bar chart of feature importances, sorted ascending (highest at top).

    Parameters
    ----------
    importance : pd.Series
        Feature importance values (e.g. XGBoost gain), indexed by feature name.

    Returns
    -------
    go.Figure
    """
    sorted_imp = importance.sort_values(ascending=True)
    fig = go.Figure(
        go.Bar(
            x=sorted_imp.values,
            y=sorted_imp.index.tolist(),
            orientation="h",
            marker_color="#48CAE4",
        )
    )
    fig.update_layout(
        title="Feature Importance (XGBoost Gain)",
        template="plotly_dark",
        xaxis_tickformat=".0%",
        margin=dict(l=120, r=20, t=50, b=20),
    )
    return fig
