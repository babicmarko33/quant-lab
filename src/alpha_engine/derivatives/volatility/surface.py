from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

REQUIRED_COLS = {"expiry", "strike", "iv"}
_TRADING_DAYS = 252


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

    @classmethod
    def from_garch_forecasts(
        cls,
        returns: pd.Series,
        expiries: list[float],
        strikes: list[float],
        spot: float,
        p: int = 1,
        q: int = 1,
    ) -> VolatilitySurface:
        """Build a flat-smile volatility surface from GARCH term-structure.

        Fits a GARCH(p,q) model to ``returns``, forecasts daily vol for each
        expiry horizon, and constructs a surface where IV is constant across
        strikes for a given expiry (flat smile) but varies by term.

        Parameters
        ----------
        returns:
            Daily decimal returns (0.01 = 1%).
        expiries:
            List of option expiries in years (e.g. [0.083, 0.25, 0.5, 1.0]).
            Must have at least 2 elements.
        strikes:
            List of strike prices. Must have at least 2 elements.
        spot:
            Current underlying price (unused in flat-smile construction but
            reserved for future moneyness-adjusted term structure).
        p:
            GARCH lag order for squared residuals.
        q:
            GARCH lag order for variance.

        Returns
        -------
        VolatilitySurface fitted on the resulting (expiry, strike, iv) grid.
        """
        from alpha_engine.derivatives.volatility.garch import fit_garch, forecast_volatility  # noqa: PLC0415

        if len(expiries) < 2:
            raise ValueError("expiries must have at least 2 elements")
        if len(strikes) < 2:
            raise ValueError("strikes must have at least 2 elements")

        result = fit_garch(returns, p=p, q=q)

        # Determine max horizon in trading days
        max_horizon = max(int(math.ceil(T * _TRADING_DAYS)) for T in expiries)
        daily_vols = forecast_volatility(result, horizon=max_horizon)

        # For each expiry T, the IV is the mean daily vol over [0, T*252] annualised
        rows: list[dict] = []
        for T in expiries:
            horizon_days = max(1, int(round(T * _TRADING_DAYS)))
            mean_daily_vol = float(daily_vols[: min(horizon_days, len(daily_vols))].mean())
            annual_iv = mean_daily_vol * math.sqrt(_TRADING_DAYS)
            for K in strikes:
                rows.append({"expiry": T, "strike": float(K), "iv": annual_iv})

        surface = cls()
        surface.fit(pd.DataFrame(rows))
        return surface

    @classmethod
    def from_options_chain(cls, chain: pd.DataFrame) -> VolatilitySurface:
        """Build a volatility surface from a live options chain DataFrame.

        The DataFrame must contain columns ``mid_iv`` (implied volatility as
        decimal, e.g. 0.25 = 25%) and ``expiry_years`` (time to expiry in
        years).  Rows with ``mid_iv == 0`` or NaN are silently dropped before
        fitting.

        Parameters
        ----------
        chain:
            DataFrame with at minimum columns ``strike``, ``expiry_years``,
            ``mid_iv``.  Compatible with the output of
            ``TradierClient.get_option_chain(...).to_dataframe()`` after
            adding an ``expiry_years`` column.

        Returns
        -------
        VolatilitySurface fitted on the (expiry, strike, iv) grid derived
        from the live chain.

        Raises
        ------
        ValueError
            If required columns ``mid_iv`` or ``expiry_years`` are absent.
        """
        required = {"mid_iv", "expiry_years", "strike"}
        missing = required - set(chain.columns)
        if missing:
            raise ValueError(
                f"chain DataFrame missing required columns: {missing}. "
                f"Got: {list(chain.columns)}"
            )

        # Drop invalid rows
        valid = chain[(chain["mid_iv"] > 0) & chain["mid_iv"].notna()].copy()

        data = pd.DataFrame({
            "expiry": valid["expiry_years"].astype(float),
            "strike": valid["strike"].astype(float),
            "iv": valid["mid_iv"].astype(float),
        })

        surface = cls()
        surface.fit(data)
        return surface
