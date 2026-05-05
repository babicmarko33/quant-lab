"""Position sizing functions for risk management.

Implements:
  1. Kelly Criterion — optimal f for maximizing log-wealth
  2. Fractional Kelly — fraction × Kelly for robustness
  3. Volatility Targeting — size inversely proportional to realized volatility

References:
  Kelly, J.L. (1956). A new interpretation of information rate.
  Thorp, E.O. (1969). Optimal gambling systems for favorable games.
  Hurst, B. et al. (2013). A century of evidence on trend-following investing.
"""

import numpy as np
import pandas as pd


def kelly_fraction(win_rate: float, win_loss_ratio: float) -> float:
    """Compute the full Kelly fraction.

    Kelly formula: f* = W - (1 - W) / R

    where:
      W = win_rate (probability of winning)
      R = win_loss_ratio (average win / average loss magnitude)

    Parameters
    ----------
    win_rate : float
        Probability of a winning trade [0, 1].
    win_loss_ratio : float
        Ratio of average gain to average loss (both positive magnitudes).

    Returns
    -------
    float
        Kelly fraction clipped to [0, 1]. Negative values → 0 (no bet).
    """
    f = win_rate - (1.0 - win_rate) / win_loss_ratio
    return float(max(0.0, min(1.0, f)))


def fractional_kelly(
    win_rate: float,
    win_loss_ratio: float,
    fraction: float = 0.5,
) -> float:
    """Compute fractional Kelly position size.

    Scales full Kelly by `fraction` for robustness to estimation error.
    Half-Kelly (fraction=0.5) is the most widely recommended value as it
    reduces variance by 75% while sacrificing only ~25% of expected growth.

    Parameters
    ----------
    win_rate : float
        Probability of a winning trade [0, 1].
    win_loss_ratio : float
        Ratio of average gain to average loss.
    fraction : float
        Scaling factor in [0, 1]. Default 0.5 (half-Kelly).

    Returns
    -------
    float
        Fractional Kelly position size in [0, 1].
    """
    full_kelly = kelly_fraction(win_rate, win_loss_ratio)
    return float(fraction * full_kelly)


def volatility_target_size(
    returns: pd.Series,
    vol_target: float = 0.15,
    lookback: int = 60,
    annualization_factor: int = 252,
    min_observations: int = 20,
) -> float:
    """Compute position size to hit a target annual volatility.

    size = min(vol_target / realized_annualized_vol, 1.0)

    This ensures that position sizing scales DOWN in high-vol regimes
    and is capped at 1.0 (no levered long positions).

    Parameters
    ----------
    returns : pd.Series
        Recent daily return series.
    vol_target : float
        Target annualized volatility (e.g. 0.15 = 15%). Default 0.15.
    lookback : int
        Rolling lookback window for volatility estimation. Default 60.
    annualization_factor : int
        Bars per year. Default 252 (daily data).
    min_observations : int
        Minimum number of return observations required. If fewer, returns 0.

    Returns
    -------
    float
        Position size in [0, 1].
    """
    recent = returns.iloc[-lookback:] if len(returns) >= lookback else returns

    if len(recent) < min_observations:
        return 0.0

    realized_vol_daily = float(recent.std(ddof=1))
    if realized_vol_daily < 1e-12:
        return 1.0  # Near-zero vol → fully invest (flat series)

    realized_vol_annual = realized_vol_daily * np.sqrt(annualization_factor)
    raw_size = vol_target / realized_vol_annual

    return float(min(raw_size, 1.0))
