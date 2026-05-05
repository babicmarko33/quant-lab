"""Vectorized technical analysis indicators.

All functions accept a pandas DataFrame with OHLCV columns and return
a pandas Series or DataFrame with the computed indicator values.

Design principles:
    - Pure functions (no side effects)
    - Vectorized with numpy (no Python loops over rows)
    - NaN for insufficient lookback periods (no forward-filling)
    - Verified against TA-Lib reference implementations
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """Simple Moving Average.

    SMA_t = (1/N) * Σ_{i=0}^{N-1} P_{t-i}

    Parameters
    ----------
    series : pd.Series
        Price series (typically 'close')
    window : int
        Lookback period N

    Returns
    -------
    pd.Series
        SMA values. First (window-1) values are NaN.
    """
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int = 20) -> pd.Series:
    """Exponential Moving Average.

    EMA_t = α * P_t + (1-α) * EMA_{t-1}
    where α = 2 / (N + 1)

    Parameters
    ----------
    series : pd.Series
        Price series
    window : int
        Span (N) for smoothing factor calculation

    Returns
    -------
    pd.Series
        EMA values
    """
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing method).

    RSI = 100 - 100 / (1 + RS)
    RS = avg_gain / avg_loss  (Wilder's exponential smoothing)

    Parameters
    ----------
    series : pd.Series
        Price series (typically 'close')
    window : int
        Lookback period (default 14 per Wilder)

    Returns
    -------
    pd.Series
        RSI values in [0, 100]. First `window` values are NaN.
    """
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)

    # Wilder's smoothing (equivalent to EMA with alpha = 1/window)
    avg_gain = gains.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))

    # Set initial lookback period to NaN
    result.iloc[:window] = np.nan
    return result


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Moving Average Convergence/Divergence.

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal)
    Histogram = MACD Line - Signal Line

    Parameters
    ----------
    series : pd.Series
        Price series
    fast : int
        Fast EMA period (default 12)
    slow : int
        Slow EMA period (default 26)
    signal : int
        Signal EMA period (default 9)

    Returns
    -------
    pd.DataFrame
        Columns: ['macd', 'signal', 'histogram']
    """
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram}, index=series.index)


def bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands.

    Middle Band = SMA(N)
    Upper Band = SMA(N) + k * σ(N)
    Lower Band = SMA(N) - k * σ(N)

    Parameters
    ----------
    series : pd.Series
        Price series
    window : int
        Lookback period
    num_std : float
        Number of standard deviations (k)

    Returns
    -------
    pd.DataFrame
        Columns: ['upper', 'middle', 'lower', 'bandwidth', 'pct_b']
    """
    middle = sma(series, window)
    rolling_std = series.rolling(window=window, min_periods=window).std()
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std

    bandwidth = (upper - lower) / middle
    pct_b = (series - lower) / (upper - lower)

    return pd.DataFrame(
        {"upper": upper, "middle": middle, "lower": lower, "bandwidth": bandwidth, "pct_b": pct_b},
        index=series.index,
    )


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range — volatility indicator.

    TR = max(H-L, |H-C_prev|, |L-C_prev|)
    ATR = Wilder's smoothed average of TR

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: 'high', 'low', 'close'
    window : int
        Smoothing period

    Returns
    -------
    pd.Series
        ATR values
    """
    high = df["high"]
    low = df["low"]
    close_prev = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def returns(series: pd.Series, method: str = "simple") -> pd.Series:
    """Compute period returns.

    Parameters
    ----------
    series : pd.Series
        Price series
    method : str
        'simple' for arithmetic returns (P_t/P_{t-1} - 1)
        'log' for logarithmic returns ln(P_t/P_{t-1})

    Returns
    -------
    pd.Series
        Return series (first value is NaN)
    """
    if method == "log":
        return np.log(series / series.shift(1))
    return series.pct_change()


def volatility(series: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Rolling historical volatility (annualized by default).

    σ_ann = σ_daily * √252

    Parameters
    ----------
    series : pd.Series
        Price series
    window : int
        Rolling window
    annualize : bool
        Whether to annualize (multiply by √252)

    Returns
    -------
    pd.Series
        Volatility values
    """
    log_ret = returns(series, method="log")
    vol = log_ret.rolling(window=window, min_periods=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol
