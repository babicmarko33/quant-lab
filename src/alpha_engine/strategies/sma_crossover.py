"""SMA Crossover Strategy (Golden Cross / Death Cross).

Signal construction:
  - Fast SMA crosses above slow SMA → long (+1)
  - Fast SMA crosses below slow SMA → short (-1)
  - Otherwise → flat (0)

Classic variant: 50-day / 200-day SMA ("Golden Cross").
"""

import pandas as pd

from alpha_engine.strategies.base import Strategy


class SMACrossoverStrategy(Strategy):
    """SMA crossover trend-following strategy.

    Parameters
    ----------
    fast : int
        Fast SMA window in trading days (default 50).
    slow : int
        Slow SMA window in trading days (default 200).
    """

    def __init__(self, fast: int = 50, slow: int = 200) -> None:
        if fast >= slow:
            raise ValueError(
                f"fast window ({fast}) must be less than slow window ({slow})."
            )
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return "sma_crossover"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate crossover signals.

        Signal at day T:
          +1 if SMA_fast[T] > SMA_slow[T] and SMA_fast[T-1] <= SMA_slow[T-1]
          -1 if SMA_fast[T] < SMA_slow[T] and SMA_fast[T-1] >= SMA_slow[T-1]
           0 otherwise (hold / no signal change)

        Position is held until the opposite crossover occurs.
        """
        close = df["close"]
        sma_fast = close.rolling(self.fast, min_periods=self.fast).mean()
        sma_slow = close.rolling(self.slow, min_periods=self.slow).mean()

        above = (sma_fast > sma_slow).fillna(False)

        # Carry position: +1 when fast > slow, -1 when fast < slow, 0 during warmup
        position = pd.Series(0.0, index=df.index)
        valid = sma_fast.notna() & sma_slow.notna()
        position[valid & above] = 1.0
        position[valid & ~above] = -1.0

        return position
