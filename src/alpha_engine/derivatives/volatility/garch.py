"""GARCH volatility forecasting via the arch library.

Input returns should be decimal (e.g. 0.01 for 1%). Internally scaled ×100
for arch (percent returns), then results are converted back to decimal space.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    from arch import arch_model  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install arch>=6.3: pip install 'arch>=6.3'") from exc

_MIN_OBS = 50
_TRADING_DAYS = 252


@dataclass
class GarchResult:
    """Fitted GARCH model parameters and back-reference for forecasting."""

    omega: float
    alpha: list[float]
    beta: list[float]
    persistence: float
    unconditional_vol: float  # annualised decimal vol
    _fit: Any = field(repr=False)


def fit_garch(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
) -> GarchResult:
    """Fit a GARCH(p,q) model to a decimal returns series.

    Parameters
    ----------
    returns:
        Daily returns in decimal form (0.01 = 1%).
    p:
        Number of ARCH (lagged squared residual) terms.
    q:
        Number of GARCH (lagged variance) terms.

    Returns
    -------
    GarchResult with fitted parameters and forecast handle.

    Raises
    ------
    ValueError
        If ``returns`` has fewer than 50 observations.
    """
    if len(returns) < _MIN_OBS:
        raise ValueError(
            f"returns must have at least 50 observations, got {len(returns)}"
        )

    returns_pct = returns * 100.0  # arch expects percent-scaled returns

    am = arch_model(returns_pct, vol="Garch", p=p, q=q, dist="normal", rescale=False)
    fit = am.fit(disp="off")

    params = fit.params
    omega = float(params["omega"])
    alpha_vals = [float(params[f"alpha[{i + 1}]"]) for i in range(p)]
    beta_vals = [float(params[f"beta[{i + 1}]"]) for i in range(q)]

    persistence = sum(alpha_vals) + sum(beta_vals)

    # Unconditional variance in percent-squared: omega / (1 - persistence)
    if persistence >= 1.0:
        unc_var_pct_sq = float("nan")
    else:
        unc_var_pct_sq = omega / (1.0 - persistence)

    # Annualised decimal vol: sqrt(var_pct_sq / 10000 * 252)
    unconditional_vol = math.sqrt(unc_var_pct_sq / 10_000 * _TRADING_DAYS)

    return GarchResult(
        omega=omega,
        alpha=alpha_vals,
        beta=beta_vals,
        persistence=persistence,
        unconditional_vol=unconditional_vol,
        _fit=fit,
    )


def forecast_volatility(result: GarchResult, horizon: int) -> np.ndarray:
    """Forecast daily decimal volatility for ``horizon`` steps ahead.

    Parameters
    ----------
    result:
        A :class:`GarchResult` from :func:`fit_garch`.
    horizon:
        Number of trading days ahead to forecast.

    Returns
    -------
    np.ndarray of shape ``(horizon,)`` with daily decimal volatilities.
    """
    fcast = result._fit.forecast(horizon=horizon, reindex=False)
    # variance shape: (1, horizon) in percent-squared
    var_pct_sq: np.ndarray = fcast.variance.iloc[-1].values
    return np.sqrt(var_pct_sq / 10_000)


def garch_vol_forecast(
    returns: pd.Series,
    horizon: int = 1,
    p: int = 1,
    q: int = 1,
) -> float:
    """Fit GARCH and return mean forecast daily vol over ``horizon`` steps.

    Convenience wrapper: fit → forecast → mean of forecast path.

    Returns
    -------
    Mean daily decimal volatility over the forecast horizon.
    """
    result = fit_garch(returns, p=p, q=q)
    vols = forecast_volatility(result, horizon=horizon)
    return float(vols.mean())
