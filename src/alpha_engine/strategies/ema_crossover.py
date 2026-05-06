"""EMA Crossover Strategy (MACD-style signal line).

Signal construction:
  - Fast EMA crosses above slow EMA → long (+1)
  - Fast EMA crosses below slow EMA → short (-1)
  - Otherwise → flat (0)

Classic variant: 12-day / 26-day EMA (MACD components).
EMA reacts faster than SMA to recent price changes.
"""

import pandas as pd

from alpha_engine.strategies.base import Strategy


class EMACrossoverStrategy(Strategy):
    """EMA crossover trend-following strategy.

    Parameters
    ----------
    fast : int
        Fast EMA span in trading days (default 12).
    slow : int
        Slow EMA span in trading days (default 26).
    """

    def __init__(self, fast: int = 12, slow: int = 26) -> None:
        if fast >= slow:
            raise ValueError(
                f"fast window ({fast}) must be less than slow window ({slow})."
            )
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return "ema_crossover"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate EMA crossover signals.

        Uses pandas ewm() with span= for standard EMA:
          alpha = 2 / (span + 1)

        Position:
          +1 when EMA_fast > EMA_slow (uptrend)
          -1 when EMA_fast < EMA_slow (downtrend)
           0 during the slow EMA warmup period (first `slow` bars)
        """
        close = df["close"]
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()

        # Warmup mask: require at least `slow` bars of data
        warmup_mask = pd.Series(False, index=df.index)
        warmup_mask.iloc[: self.slow - 1] = True

        signals = pd.Series(0.0, index=df.index)
        signals[~warmup_mask & (ema_fast > ema_slow)] = 1.0
        signals[~warmup_mask & (ema_fast < ema_slow)] = -1.0
        return signals
