"""Feature engineering pipeline for quantitative strategies and ML models.

Builds feature matrices from raw OHLCV data using configurable indicator sets.
Enforces no look-ahead bias: feature at time t uses only data up to and including t.
"""

import pandas as pd

from quantcore.indicators.technical import (
    atr,
    bollinger_bands,
    ema,
    macd,
    returns,
    rsi,
    sma,
    volatility,
)


def build_features(
    df: pd.DataFrame,
    sma_windows: list[int] | None = None,
    ema_windows: list[int] | None = None,
    rsi_window: int = 14,
    macd_params: tuple[int, int, int] = (12, 26, 9),
    bollinger_window: int = 20,
    atr_window: int = 14,
    vol_window: int = 20,
    lag_periods: list[int] | None = None,
    include_returns: bool = True,
    include_volume_features: bool = True,
) -> pd.DataFrame:
    """Build a comprehensive feature DataFrame from OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data with columns: [open, high, low, close, volume, adj_close]
    sma_windows : list[int]
        SMA periods to compute (default: [5, 10, 20, 50, 200])
    ema_windows : list[int]
        EMA periods to compute (default: [9, 12, 26])
    rsi_window : int
        RSI period (default: 14)
    macd_params : tuple
        (fast, slow, signal) periods for MACD
    bollinger_window : int
        Bollinger Bands period
    atr_window : int
        ATR period
    vol_window : int
        Historical volatility window
    lag_periods : list[int]
        Lag periods for return features (default: [1, 2, 3, 5, 10, 21])
    include_returns : bool
        Whether to include return-based features
    include_volume_features : bool
        Whether to include volume-based features

    Returns
    -------
    pd.DataFrame
        Feature matrix. All features use only data available at time t.
        Contains NaN rows at the start (max lookback period).
    """
    if sma_windows is None:
        sma_windows = [5, 10, 20, 50, 200]
    if ema_windows is None:
        ema_windows = [9, 12, 26]
    if lag_periods is None:
        lag_periods = [1, 2, 3, 5, 10, 21]

    close = df["close"]
    features = pd.DataFrame(index=df.index)

    # --- Price-based features ---
    for w in sma_windows:
        features[f"sma_{w}"] = sma(close, w)
        features[f"close_to_sma_{w}"] = close / sma(close, w) - 1  # Relative position

    for w in ema_windows:
        features[f"ema_{w}"] = ema(close, w)

    # --- Momentum indicators ---
    features["rsi"] = rsi(close, rsi_window)

    macd_df = macd(close, *macd_params)
    features["macd"] = macd_df["macd"]
    features["macd_signal"] = macd_df["signal"]
    features["macd_histogram"] = macd_df["histogram"]

    # --- Volatility indicators ---
    bb_df = bollinger_bands(close, bollinger_window)
    features["bb_upper"] = bb_df["upper"]
    features["bb_lower"] = bb_df["lower"]
    features["bb_bandwidth"] = bb_df["bandwidth"]
    features["bb_pct_b"] = bb_df["pct_b"]

    features["atr"] = atr(df, atr_window)
    features["volatility"] = volatility(close, vol_window)

    # --- Return features ---
    if include_returns:
        features["return_1d"] = returns(close, "simple")
        features["log_return_1d"] = returns(close, "log")
        for lag in lag_periods:
            features[f"return_{lag}d"] = close.pct_change(lag)

    # --- Volume features ---
    if include_volume_features and "volume" in df.columns:
        vol = df["volume"]
        features["volume_sma_20"] = sma(vol, 20)
        features["volume_ratio"] = vol / sma(vol, 20)  # Volume relative to 20d average
        features["dollar_volume"] = vol * close

    return features


def add_target(
    features: pd.DataFrame,
    close: pd.Series,
    horizon: int = 1,
    target_type: str = "direction",
) -> pd.DataFrame:
    """Add prediction target column to features.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix
    close : pd.Series
        Close price series
    horizon : int
        Forward-looking period for target
    target_type : str
        'direction' — binary (1 if price goes up, 0 otherwise)
        'return' — continuous forward return

    Returns
    -------
    pd.DataFrame
        Features with 'target' column appended.
        NaN in last `horizon` rows (no forward data available).
    """
    df = features.copy()
    forward_return = close.shift(-horizon) / close - 1

    if target_type == "direction":
        df["target"] = (forward_return > 0).astype(float)
        df.loc[forward_return.isna(), "target"] = float("nan")
    elif target_type == "return":
        df["target"] = forward_return
    else:
        raise ValueError(f"Unknown target_type: {target_type}. Use 'direction' or 'return'.")

    return df
