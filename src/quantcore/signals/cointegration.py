"""Engle-Granger cointegration test and related utilities.

Uses statsmodels `coint` for the ADF test on residuals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from itertools import combinations

try:
    from statsmodels.tsa.stattools import coint  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install statsmodels>=0.14: pip install 'statsmodels>=0.14'") from exc

_MIN_OBS = 30


@dataclass
class CointegrationResult:
    """Result of an Engle-Granger cointegration test."""

    asset_1: str
    asset_2: str
    test_statistic: float
    p_value: float
    hedge_ratio: float
    is_cointegrated: bool


def hedge_ratio(x: pd.Series, y: pd.Series) -> float:
    """OLS hedge ratio: coefficient of x when regressing y on x.

    Parameters
    ----------
    x, y : pd.Series
        Price series of equal length.

    Returns
    -------
    float
        OLS slope coefficient β such that y ≈ β·x + ε.
    """
    x_arr = x.values.astype(float)
    y_arr = y.values.astype(float)
    # OLS: β = (x'x)^{-1} x'y with intercept absorbed via demeaning
    X = np.column_stack([np.ones(len(x_arr)), x_arr])
    coeffs, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
    return float(coeffs[1])


def engle_granger_test(
    x: pd.Series,
    y: pd.Series,
    max_p_value: float = 0.05,
) -> CointegrationResult:
    """Run the Engle-Granger two-step cointegration test.

    Parameters
    ----------
    x, y:
        Price series to test. Must be same length and ≥30 observations.
    max_p_value:
        Significance threshold for `is_cointegrated`.

    Raises
    ------
    ValueError
        If series lengths differ or are too short.
    """
    if len(x) != len(y):
        raise ValueError(
            f"x and y must have the same length, got {len(x)} and {len(y)}"
        )
    if len(x) < _MIN_OBS:
        raise ValueError(
            f"Series must have at least 30 observations, got {len(x)}"
        )

    x_name = x.name if x.name else "x"
    y_name = y.name if y.name else "y"

    stat, p_val, _ = coint(x.values, y.values)
    hr = hedge_ratio(x, y)

    return CointegrationResult(
        asset_1=str(x_name),
        asset_2=str(y_name),
        test_statistic=float(stat),
        p_value=float(p_val),
        hedge_ratio=hr,
        is_cointegrated=float(p_val) < max_p_value,
    )


def find_cointegrated_pairs(
    prices: pd.DataFrame,
    max_p_value: float = 0.05,
) -> pd.DataFrame:
    """Test all pairwise combinations in ``prices`` for cointegration.

    Parameters
    ----------
    prices:
        DataFrame where each column is a price series.
    max_p_value:
        Significance threshold for ``is_cointegrated``.

    Returns
    -------
    pd.DataFrame with columns: asset_1, asset_2, p_value, hedge_ratio, is_cointegrated.
    """
    rows = []
    for col1, col2 in combinations(prices.columns, 2):
        result = engle_granger_test(
            prices[col1], prices[col2], max_p_value=max_p_value
        )
        rows.append({
            "asset_1": result.asset_1,
            "asset_2": result.asset_2,
            "p_value": result.p_value,
            "hedge_ratio": result.hedge_ratio,
            "is_cointegrated": result.is_cointegrated,
        })
    return pd.DataFrame(rows)
