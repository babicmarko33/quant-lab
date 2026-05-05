"""Purged K-Fold cross-validator for financial time series.

Standard k-fold cross-validation leaks information in financial data because
adjacent observations have overlapping label periods (e.g., a 5-day forward
return computed at day T uses data from T+1 through T+5, which may overlap
with the test fold).

This implementation follows López de Prado (2018), Chapter 7:
  1. **Purging**: Remove training samples whose label period overlaps with
     the test fold's sample period.
  2. **Embargo**: Remove training samples immediately following the test fold
     to prevent autocorrelation leakage.

Reference:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    John Wiley & Sons. Chapter 7: "Cross-Validation in Finance."
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PurgedKFold:
    """Purged k-fold cross-validator.

    Parameters
    ----------
    n_splits : int
        Number of folds. Default 5.
    purge_window : int
        Number of bars to purge from the TRAIN set adjacent to the test fold
        START. Prevents label overlap leakage at the fold boundary.
        Set to prediction horizon (e.g., 5 for 5-day ahead prediction).
    embargo : int
        Number of bars to exclude from the TRAIN set immediately AFTER the
        test fold END. Prevents autocorrelation leakage.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_window: int = 0,
        embargo: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = n_splits
        self.purge_window = purge_window
        self.embargo = embargo

    def split(
        self,
        X: pd.DatetimeIndex | pd.DataFrame | pd.Series,
        y: pd.Series | None = None,
        groups: None = None,
    ):
        """Generate train/test index arrays for each fold.

        Parameters
        ----------
        X : array-like or DatetimeIndex
            Input data. Only len(X) is used.
        y, groups : ignored

        Yields
        ------
        train_idx : np.ndarray of int
            Integer indices of training samples.
        test_idx : np.ndarray of int
            Integer indices of test samples.
        """
        n = len(X)
        indices = np.arange(n)
        fold_size = n // self.n_splits

        for k in range(self.n_splits):
            test_start = k * fold_size
            # Last fold absorbs remainder
            test_end = (test_start + fold_size) if k < self.n_splits - 1 else n

            test_idx = indices[test_start:test_end]

            # Purge: exclude train samples within purge_window of test_start
            purge_start = max(0, test_start - self.purge_window)

            # Embargo: exclude train samples within embargo bars after test_end
            embargo_end = min(n, test_end + self.embargo)

            # Train = everything before purge boundary + everything after embargo
            train_idx = np.concatenate([
                indices[:purge_start],
                indices[embargo_end:],
            ])

            yield train_idx, test_idx
