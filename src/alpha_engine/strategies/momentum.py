"""12-1 Cross-sectional Momentum Strategy (Jegadeesh & Titman, 1993).

Signal construction:
  - Lookback window: 12 months (252 trading days by default)
  - Skip last 1 month (21 days) to avoid short-term reversal
  - Signal = +1 if 12-1 return > 0 (or cross-sectional rank > median)
  - Signal = -1 if 12-1 return < 0
  - Signal = 0 during warmup period
"""

import pandas as pd

from alpha_engine.strategies.base import Strategy


class MomentumStrategy(Strategy):
    """Pure time-series 12-1 momentum strategy.

    Parameters
    ----------
    lookback : int
        Momentum lookback in trading days (default 252 ≈ 12 months).
    skip : int
        Skip most recent days to avoid short-term reversal (default 21 ≈ 1 month).
    """

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        self.lookback = lookback
        self.skip = skip

    @property
    def name(self) -> str:
        return f"Momentum({self.lookback}d-skip{self.skip}d)"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Compute 12-1 momentum signal.

        Signal at day T = sign(close[T-skip] / close[T-lookback] - 1)
        Returns 0 during warmup period, NaN-safe.
        """
        close = df["close"]
        # Price lookback days ago (skip most recent `skip` days)
        past_price = close.shift(self.skip)
        older_price = close.shift(self.lookback)

        momentum_return = past_price / older_price - 1.0

        signals = pd.Series(0.0, index=df.index)
        signals[momentum_return > 0] = 1.0
        signals[momentum_return < 0] = -1.0
        # Warmup: not enough history → stay 0
        signals.iloc[: self.lookback + self.skip] = 0.0

        return signals
