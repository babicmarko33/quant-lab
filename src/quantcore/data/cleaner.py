from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Validate OHLCV DataFrame structure and invariants.

    Raises
    ------
    ValueError
        If any structural or domain invariant is violated.
    """
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if (df["close"] < 0).any() or (df["open"] < 0).any():
        raise ValueError("Prices contain negative values — check data source.")

    if (df["high"] < df["low"]).any():
        raise ValueError(
            "high < low detected — OHLCV data integrity failure. "
            "Check for split/dividend adjustments."
        )

    if (df["volume"] <= 0).any():
        raise ValueError(
            "volume <= 0 detected — non-trading rows present. "
            "Use drop_zero_volume=True in clean_ohlcv() to remove them."
        )


def detect_outliers(
    series: pd.Series,
    z_threshold: float = 3.0,
) -> pd.Series:
    """Detect outliers using Median Absolute Deviation (MAD).

    MAD is robust to the outlier itself (unlike global z-score, where a single
    extreme value inflates the mean and standard deviation, masking itself).

    Modified z-score: 0.6745 * (xi - median) / MAD

    Parameters
    ----------
    series : pd.Series
        Numeric price or volume series.
    z_threshold : float
        MAD-based z-score threshold to flag as outlier (default 3.0).

    Returns
    -------
    pd.Series[bool]
        True where the value is an outlier.
    """
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        return pd.Series(False, index=series.index)
    modified_z = 0.6745 * (series - median).abs() / mad
    return modified_z > z_threshold


def clean_ohlcv(
    df: pd.DataFrame,
    remove_outliers: bool = False,
    z_threshold: float = 3.0,
    drop_zero_volume: bool = False,
) -> pd.DataFrame:
    """Clean and normalise an OHLCV DataFrame.

    Operations applied (in order):
    1. Sort index ascending (ensure monotonic dates).
    2. Drop rows where volume <= 0 (optional, controlled by ``drop_zero_volume``).
    3. Replace outlier close prices with NaN (optional, controlled by ``remove_outliers``).
    4. Forward-fill NaN values (then back-fill any leading NaN).

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data with columns [open, high, low, close, volume].
    remove_outliers : bool
        If True, outlier close prices are replaced with NaN before filling.
    z_threshold : float
        Z-score threshold for outlier detection (default 3.0).
    drop_zero_volume : bool
        If True, rows with volume <= 0 are dropped entirely.

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV DataFrame with the same schema.
    """
    result = df.sort_index()

    if drop_zero_volume:
        result = result[result["volume"] > 0].copy()

    if remove_outliers:
        outlier_mask = detect_outliers(result["close"], z_threshold=z_threshold)
        result.loc[outlier_mask, "close"] = np.nan

    return result.ffill().bfill()
