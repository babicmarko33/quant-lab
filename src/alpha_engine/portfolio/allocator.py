"""Abstract base class for all portfolio allocators."""

from abc import ABC, abstractmethod

import pandas as pd


class Allocator(ABC):
    """Abstract interface for portfolio weight allocation.

    All allocators implement a single ``fit(returns)`` method that
    takes a DataFrame of asset returns and returns portfolio weights.

    Weights contract:
      - ``pd.Series`` indexed by asset names (same as ``returns.columns``)
      - ``weights.sum() == 1.0`` (fully invested)
      - ``weights >= 0`` (long-only, unless subclass explicitly allows shorts)
    """

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> pd.Series:
        """Compute portfolio weights from historical returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Daily returns with assets as columns. No NaN values.

        Returns
        -------
        pd.Series
            Portfolio weights indexed by asset name, summing to 1.0.
        """
