"""Equal-weight portfolio allocator.

The simplest possible allocator: 1/N weight per asset.
Serves as a benchmark — difficult to consistently beat with sophisticated methods.

Reference:
    DeMiguel, V., Garlappi, L., Uppal, R. (2009).
    Optimal Versus Naive Diversification: How Inefficient is the 1/N Portfolio Strategy?
    Review of Financial Studies, 22(5), 1915-1953.
"""

import pandas as pd

from alpha_engine.portfolio.allocator import Allocator


class EqualWeightAllocator(Allocator):
    """Equal-weight (1/N) portfolio allocator."""

    def fit(self, returns: pd.DataFrame) -> pd.Series:
        """Return 1/N weights for all assets."""
        n = len(returns.columns)
        return pd.Series(1.0 / n, index=returns.columns, name="weight")
