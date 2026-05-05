"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod

import pandas as pd

from alpha_engine.backtest.engine import run_backtest
from alpha_engine.backtest.types import BacktestResult


class Strategy(ABC):
    """Abstract base for all alpha strategies.

    Subclasses must implement:
      - ``name`` (str property)
      - ``generate_signals(df) -> pd.Series``

    The base class provides a concrete ``run()`` method that wires
    signals through the vectorized backtest engine.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Compute position signals from OHLCV data.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with at minimum 'close', 'open' columns.

        Returns
        -------
        pd.Series
            Float series in [-1.0, 0.0, +1.0] indexed like df.
            Values in (-1, 0) and (0, 1) are allowed for fractional sizing.
        """

    def run(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100_000.0,
        commission_bps: int = 10,
        slippage_bps: int = 5,
    ) -> BacktestResult:
        """Generate signals then run a full vectorized backtest.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data.
        initial_capital : float
            Starting portfolio value.
        commission_bps : int
            One-way commission in basis points.
        slippage_bps : int
            One-way slippage in basis points.

        Returns
        -------
        BacktestResult
        """
        signals = self.generate_signals(df)
        return run_backtest(signals, df, initial_capital, commission_bps, slippage_bps)
