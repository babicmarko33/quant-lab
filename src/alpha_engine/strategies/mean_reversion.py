"""Bollinger Band Mean-Reversion Strategy.

Signal construction:
  - Compute 20-day SMA and ±2σ Bollinger Bands
  - Signal = +1 (long) when price crosses below lower band → expect reversion up
  - Signal = -1 (short) when price crosses above upper band → expect reversion down
  - Signal = 0 when price is within the bands (neutral zone)
  - Exit signal: revert to 0 when price crosses back to SMA

Reference: Bollinger, J. (2001). Bollinger on Bollinger Bands.
"""

import pandas as pd

from alpha_engine.strategies.base import Strategy


class BollingerMeanReversionStrategy(Strategy):
    """Bollinger Band mean-reversion strategy.

    Parameters
    ----------
    window : int
        Rolling window for SMA and standard deviation (default 20).
    num_std : float
        Number of standard deviations for upper/lower bands (default 2.0).
    """

    def __init__(self, window: int = 20, num_std: float = 2.0) -> None:
        self.window = window
        self.num_std = num_std

    @property
    def name(self) -> str:
        return f"BollingerMeanReversion({self.window}d, {self.num_std}std)"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Compute Bollinger Band mean-reversion signals.

        Below lower band → +1 (expect bounce up)
        Above upper band → -1 (expect reversion down)
        Within bands    →  0 (no position)
        """
        close = df["close"]

        sma = close.rolling(self.window).mean()
        std = close.rolling(self.window).std(ddof=1)

        upper_band = sma + self.num_std * std
        lower_band = sma - self.num_std * std

        signals = pd.Series(0.0, index=df.index)
        signals[close < lower_band] = 1.0   # Oversold → long
        signals[close > upper_band] = -1.0  # Overbought → short

        # Warmup: insufficient rolling window → stay neutral
        signals.iloc[: self.window] = 0.0

        return signals
