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


def payoff_fig(
    strikes: np.ndarray,
    payoffs: np.ndarray,
    spot: float,
    title: str = "Option Payoff at Expiry",
) -> go.Figure:
    """Line chart of option payoff profile at expiry.

    Parameters
    ----------
    strikes : np.ndarray
        Range of spot prices at expiry.
    payoffs : np.ndarray
        Corresponding net payoff values.
    spot : float
        Current spot price (shown as a vertical line).
    title : str
        Chart title.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=strikes, y=payoffs,
        name="Payoff",
        line=dict(color="#00B4D8", width=2),
        fill="tozeroy",
        fillcolor="rgba(0,180,216,0.15)",
    ))
    fig.add_vline(x=spot, line_dash="dash", line_color="#FFB703",
                  annotation_text=f"S={spot:.0f}", annotation_position="top right")
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title="Spot at Expiry",
        yaxis_title="Profit / Loss",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def greeks_sensitivity_fig(
    param_values: np.ndarray,
    greek_series: dict[str, np.ndarray],
    param_label: str = "Spot",
    title: str = "Greeks Sensitivity",
) -> go.Figure:
    """Multi-line chart of Greeks across a parameter range.

    Parameters
    ----------
    param_values : np.ndarray
        X-axis values (e.g. spot range).
    greek_series : dict[str, np.ndarray]
        Mapping from Greek name → values array.
    param_label : str
        X-axis label.
    title : str
        Chart title.
    """
    colours = ["#00B4D8", "#90E0EF", "#FFB703", "#FF6B6B", "#06D6A0"]
    fig = go.Figure()
    for (name, vals), colour in zip(greek_series.items(), colours, strict=False):
        fig.add_trace(go.Scatter(
            x=param_values, y=vals,
            name=name, line=dict(color=colour, width=2),
        ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title=param_label,
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def model_comparison_bar_fig(model_prices: dict[str, float], title: str = "Model Price Comparison") -> go.Figure:
    """Bar chart comparing prices from multiple models.

    Parameters
    ----------
    model_prices : dict[str, float]
        Model name → price.
    title : str
        Chart title.
    """
    names = list(model_prices.keys())
    prices = [model_prices[n] for n in names]
    fig = go.Figure(go.Bar(
        x=names, y=prices,
        marker_color=["#00B4D8", "#48CAE4", "#90E0EF", "#ADE8F4", "#CAF0F8"][:len(names)],
        text=[f"{p:.4f}" for p in prices],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        yaxis_title="Option Price",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def ohlcv_fig(df: pd.DataFrame, title: str = "Price History") -> go.Figure:
    """Candlestick chart with volume subplot.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with columns: open, high, low, close, volume.
        DatetimeIndex preferred.
    title : str
        Chart title.
    """
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.04,
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
        increasing_line_color="#06D6A0", decreasing_line_color="#FF6B6B",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"], name="Volume",
        marker_color="rgba(0,180,216,0.4)",
    ), row=2, col=1)
    fig.update_layout(
        title=title, template="plotly_dark", height=520,
        xaxis_rangeslider_visible=False,
        margin=dict(l=40, r=20, t=50, b=20),
    )
    return fig


def rolling_vol_fig(
    returns: pd.Series,
    windows: list[int] | None = None,
    title: str = "Rolling Annualised Volatility",
) -> go.Figure:
    """Line chart of rolling annualised volatility.

    Parameters
    ----------
    returns : pd.Series
        Daily arithmetic returns with DatetimeIndex.
    windows : list[int]
        Rolling window sizes in days. Default [20, 60, 120].
    title : str
        Chart title.
    """
    if windows is None:
        windows = [20, 60, 120]
    colours = ["#00B4D8", "#FFB703", "#FF6B6B", "#06D6A0"]
    fig = go.Figure()
    for w, colour in zip(windows, colours, strict=False):
        rv = returns.rolling(w).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=rv.index, y=rv.values,
            name=f"{w}d", line=dict(color=colour, width=1.5),
        ))
    fig.update_layout(
        title=title, template="plotly_dark",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=50, b=20),
    )
    return fig


def vol_surface_heatmap_fig(
    strikes: np.ndarray,
    expiries: np.ndarray,
    ivs: np.ndarray,
    title: str = "Implied Volatility Surface",
) -> go.Figure:
    """Heatmap of implied vols across strikes and expiries.

    Parameters
    ----------
    strikes : np.ndarray
        Strike values (columns), shape (n_strikes,).
    expiries : np.ndarray
        Expiry values in years (rows), shape (n_expiries,).
    ivs : np.ndarray
        Implied vol matrix, shape (n_expiries, n_strikes).
    title : str
        Chart title.
    """
    fig = go.Figure(go.Heatmap(
        z=ivs * 100,  # convert to %
        x=[f"{k:.0f}" for k in strikes],
        y=[f"{t:.2f}y" for t in expiries],
        colorscale="Viridis",
        colorbar=dict(title="IV (%)"),
        text=np.round(ivs * 100, 1),
        texttemplate="%{text}%",
    ))
    fig.update_layout(
        title=title, template="plotly_dark",
        xaxis_title="Strike", yaxis_title="Expiry",
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


def sabr_smile_fig(
    strikes: np.ndarray,
    market_vols: np.ndarray | None,
    model_vols: np.ndarray,
    title: str = "SABR Vol Smile",
) -> go.Figure:
    """Overlay of SABR model smile vs market quotes.

    Parameters
    ----------
    strikes : np.ndarray
        Strike values.
    market_vols : np.ndarray | None
        Observed market IV (can be None if no market data).
    model_vols : np.ndarray
        SABR-implied vols.
    title : str
        Chart title.
    """
    fig = go.Figure()
    if market_vols is not None:
        fig.add_trace(go.Scatter(
            x=strikes, y=market_vols * 100,
            mode="markers", name="Market",
            marker=dict(color="#FFB703", size=8, symbol="circle"),
        ))
    fig.add_trace(go.Scatter(
        x=strikes, y=model_vols * 100,
        name="SABR", line=dict(color="#00B4D8", width=2),
    ))
    fig.update_layout(
        title=title, template="plotly_dark",
        xaxis_title="Strike", yaxis_title="Implied Vol (%)",
        legend=dict(orientation="h"),
        margin=dict(l=40, r=20, t=50, b=40),
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
