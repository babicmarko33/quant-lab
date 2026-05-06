"""RSI Mean-Reversion Strategy (Wilder, 1978).

Signal construction:
  - RSI(14) < oversold (default 30) → long (+1)  -- oversold, expect bounce
  - RSI(14) > overbought (default 70) → short (-1) -- overbought, expect pullback
  - Otherwise → flat (0)

RSI is computed using Wilder's exponential smoothing:
  RS = avg_gain / avg_loss over the window period.
  RSI = 100 - 100 / (1 + RS)
"""

import pandas as pd

from alpha_engine.strategies.base import Strategy
from quantcore.indicators.technical import rsi as compute_rsi


class RSIMeanReversionStrategy(Strategy):
    """RSI mean-reversion contrarian strategy.

    Parameters
    ----------
    window : int
        RSI lookback window (default 14 — Wilder's original).
    oversold : float
        RSI level below which to go long (default 30).
    overbought : float
        RSI level above which to go short (default 70).
    """

    def __init__(
        self,
        window: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        if oversold >= overbought:
            raise ValueError(
                f"oversold threshold ({oversold}) must be less than "
                f"overbought threshold ({overbought})."
            )
        self.window = window
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return "rsi_mean_reversion"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate RSI mean-reversion signals.

        Signal at day T:
          +1 if RSI[T] < oversold  (buy the dip)
          -1 if RSI[T] > overbought (sell the rip)
           0 otherwise (neutral — within normal RSI band)
        """
        rsi_values = compute_rsi(df["close"], window=self.window)

        signals = pd.Series(0.0, index=df.index)
        signals[rsi_values < self.oversold] = 1.0
        signals[rsi_values > self.overbought] = -1.0
        return signals
