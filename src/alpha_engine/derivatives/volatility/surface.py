from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

REQUIRED_COLS = {"expiry", "strike", "iv"}


class VolatilitySurface:
    """Cubic spline volatility surface.

    Fits a bivariate spline on an (expiry, strike) grid of implied volatilities.
    Supports pointwise interpolation and bulk surface_grid queries.

    Usage
    -----
    surf = VolatilitySurface()
    surf.fit(data)   # data: DataFrame with columns [expiry, strike, iv]
    iv = surf.interpolate(strike=100, expiry=0.5)
    grid = surf.surface_grid(strikes=[90,100,110], expiries=[0.25,0.5,1.0])
    """

    def __init__(self) -> None:
        self._spline: RectBivariateSpline | None = None

    def fit(self, data: pd.DataFrame) -> None:
        """Fit the spline surface to observed implied volatility data.

        Parameters
        ----------
        data : pd.DataFrame
            Must contain columns 'expiry', 'strike', 'iv'.
            Each (expiry, strike) pair must be unique.
        """
        missing = REQUIRED_COLS - set(data.columns)
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}. "
                f"Got: {list(data.columns)}"
            )
        pivoted = (
            data.pivot(index="expiry", columns="strike", values="iv")
            .sort_index()
        )
        expiries = pivoted.index.values.astype(float)
        strikes = pivoted.columns.values.astype(float)
        z = pivoted.values.astype(float)

        kx = min(3, len(expiries) - 1)
        ky = min(3, len(strikes) - 1)
        self._spline = RectBivariateSpline(expiries, strikes, z, kx=kx, ky=ky)

    def interpolate(self, strike: float, expiry: float) -> float:
        """Interpolate implied volatility at a single (strike, expiry) point."""
        if self._spline is None:
            raise RuntimeError(
                "VolatilitySurface is not fitted. Call fit() before interpolate()."
            )
        return float(self._spline(expiry, strike)[0, 0])

    def surface_grid(
        self, strikes: list[float], expiries: list[float]
    ) -> pd.DataFrame:
        """Evaluate the surface on a grid.

        Returns
        -------
        pd.DataFrame
            Shape (len(expiries), len(strikes)), indexed by expiry, columned by strike.
        """
        if self._spline is None:
            raise RuntimeError(
                "VolatilitySurface is not fitted. Call fit() before surface_grid()."
            )
        grid = self._spline(np.array(expiries), np.array(strikes))
        return pd.DataFrame(grid, index=expiries, columns=strikes)
