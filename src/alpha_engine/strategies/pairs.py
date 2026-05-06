"""Pairs trading strategy using Kalman filter dynamic hedge ratio.

Generates long/short signals for the spread between two cointegrated assets.
Signal +1 = spread too low (buy Y, sell X); -1 = spread too high (sell Y, buy X).
"""
from __future__ import annotations

import pandas as pd

from quantcore.signals.kalman_filter import KalmanPairFilter


class PairsStrategy:
    """Statistical arbitrage strategy on a cointegrated pair.

    Uses a Kalman filter to track the dynamic hedge ratio and generates
    entry/exit signals based on the normalised spread (z-score).

    Parameters
    ----------
    entry_z:
        Z-score threshold to enter a trade. Default 2.0.
    exit_z:
        Z-score threshold to exit (mean-revert back to). Default 0.5.
    """

    def __init__(self, entry_z: float = 2.0, exit_z: float = 0.5) -> None:
        if entry_z <= exit_z:
            raise ValueError(f"entry_z ({entry_z}) must be greater than exit_z ({exit_z})")
        self.entry_z = entry_z
        self.exit_z = exit_z

    @property
    def name(self) -> str:
        return "pairs_trading"

    def generate_signals(
        self,
        df_x: pd.DataFrame,
        df_y: pd.DataFrame,
    ) -> pd.Series:
        """Generate +1/-1/0 signals for the Y leg of the pair.

        Signal convention (Y leg):
        - +1: spread is unusually low → buy Y (expect spread to widen)
        - -1: spread is unusually high → sell Y (expect spread to compress)
        - 0: no position

        Parameters
        ----------
        df_x, df_y:
            OHLCV DataFrames with a ``close`` column, same length and index.

        Returns
        -------
        pd.Series of {-1, 0, 1} indexed by ``df_x.index``.
        """
        if len(df_x) != len(df_y):
            raise ValueError(
                f"df_x and df_y must have the same length, got {len(df_x)} and {len(df_y)}"
            )

        kf = KalmanPairFilter()
        kf.fit(df_x["close"], df_y["close"])
        z = kf.zscore_

        signals = pd.Series(0, index=df_x.index, dtype=int)

        position = 0
        for i in range(len(z)):
            if position == 0:
                if z[i] < -self.entry_z:
                    position = 1   # spread low → long Y
                elif z[i] > self.entry_z:
                    position = -1  # spread high → short Y
            elif position == 1 and z[i] >= -self.exit_z:
                position = 0
            elif position == -1 and z[i] <= self.exit_z:
                position = 0
            signals.iloc[i] = position

        return signals
