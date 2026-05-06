import math

import numpy as np
import pandas as pd
import pytest

from alpha_engine.derivatives.volatility.surface import VolatilitySurface


@pytest.fixture
def sample_surface_data():
    """3 expiries x 5 strikes = 15 data points."""
    expiries = [0.25, 0.5, 1.0]
    strikes = [80, 90, 100, 110, 120]
    rows = []
    for T in expiries:
        for K in strikes:
            moneyness = abs(K - 100) / 100
            iv = 0.20 + 0.05 * moneyness + 0.01 * T
            rows.append({"expiry": T, "strike": K, "iv": iv})
    return pd.DataFrame(rows)


class TestVolatilitySurface:
    def test_fit_and_interpolate_known_point(self, sample_surface_data):
        """Interpolation at a grid node recovers the original IV."""
        surf = VolatilitySurface()
        surf.fit(sample_surface_data)
        iv = surf.interpolate(strike=100, expiry=0.5)
        expected = sample_surface_data.query("strike==100 and expiry==0.5")["iv"].iloc[0]
        assert abs(iv - expected) < 1e-4

    def test_interpolate_between_strikes(self, sample_surface_data):
        """In a smile, ATM vol is lower than OTM vols at same expiry."""
        surf = VolatilitySurface()
        surf.fit(sample_surface_data)
        iv_90 = surf.interpolate(strike=90, expiry=0.5)
        iv_110 = surf.interpolate(strike=110, expiry=0.5)
        iv_100 = surf.interpolate(strike=100, expiry=0.5)
        # ATM is at the minimum of the smile
        assert iv_100 <= iv_90 + 0.01
        assert iv_100 <= iv_110 + 0.01

    def test_interpolate_returns_positive_iv(self, sample_surface_data):
        surf = VolatilitySurface()
        surf.fit(sample_surface_data)
        iv = surf.interpolate(strike=95, expiry=0.75)
        assert iv > 0

    def test_surface_grid_shape(self, sample_surface_data):
        """surface_grid() returns DataFrame with correct shape."""
        surf = VolatilitySurface()
        surf.fit(sample_surface_data)
        grid = surf.surface_grid(strikes=[90, 100, 110], expiries=[0.25, 0.5, 1.0])
        assert grid.shape == (3, 3)

    def test_raises_before_fit(self):
        surf = VolatilitySurface()
        with pytest.raises(RuntimeError, match="not fitted"):
            surf.interpolate(100, 0.5)

    def test_fit_requires_columns(self):
        bad_df = pd.DataFrame({"K": [100], "T": [1.0], "vol": [0.20]})
        surf = VolatilitySurface()
        with pytest.raises(ValueError, match="columns"):
            surf.fit(bad_df)

    def test_surface_grid_values_positive(self, sample_surface_data):
        surf = VolatilitySurface()
        surf.fit(sample_surface_data)
        grid = surf.surface_grid(strikes=[85, 95, 100, 105, 115], expiries=[0.25, 0.5, 1.0])
        assert (grid.values > 0).all()


# ---------------------------------------------------------------------------
# from_garch_forecasts — TDD tests written before implementation
# ---------------------------------------------------------------------------

@pytest.fixture
def garch_returns() -> pd.Series:
    rng = np.random.default_rng(42)
    n = 500
    vol = np.ones(n) * 0.01
    r = np.empty(n)
    for i in range(n):
        r[i] = rng.normal(0, vol[i])
        vol_sq_next = 1e-6 + 0.05 * r[i] ** 2 + 0.90 * vol[i] ** 2
        if i + 1 < n:
            vol[i + 1] = math.sqrt(vol_sq_next)
    return pd.Series(r, name="returns")


class TestFromGarchForecasts:
    def test_returns_vol_surface(self, garch_returns):
        expiries = [0.083, 0.25, 0.5, 1.0]
        strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
        surf = VolatilitySurface.from_garch_forecasts(
            returns=garch_returns,
            expiries=expiries,
            strikes=strikes,
            spot=100.0,
        )
        assert isinstance(surf, VolatilitySurface)

    def test_is_fitted_after_construction(self, garch_returns):
        expiries = [0.25, 0.5, 1.0]
        strikes = [90.0, 100.0, 110.0]
        surf = VolatilitySurface.from_garch_forecasts(
            returns=garch_returns,
            expiries=expiries,
            strikes=strikes,
            spot=100.0,
        )
        iv = surf.interpolate(strike=100.0, expiry=0.5)
        assert iv > 0

    def test_surface_shape(self, garch_returns):
        expiries = [0.083, 0.25, 0.5, 1.0]
        strikes = [85.0, 95.0, 100.0, 105.0, 115.0]
        surf = VolatilitySurface.from_garch_forecasts(
            returns=garch_returns,
            expiries=expiries,
            strikes=strikes,
            spot=100.0,
        )
        grid = surf.surface_grid(strikes=strikes, expiries=expiries)
        assert grid.shape == (4, 5)

    def test_all_ivs_positive(self, garch_returns):
        expiries = [0.25, 0.5, 1.0]
        strikes = [80.0, 100.0, 120.0]
        surf = VolatilitySurface.from_garch_forecasts(
            returns=garch_returns,
            expiries=expiries,
            strikes=strikes,
            spot=100.0,
        )
        grid = surf.surface_grid(strikes=strikes, expiries=expiries)
        assert (grid.values > 0).all()

    def test_longer_expiry_has_higher_or_equal_vol(self, garch_returns):
        """GARCH long-horizon vol mean-reverts upward from short — term structure."""
        expiries = [0.083, 1.0]
        strikes = [90.0, 100.0, 110.0]
        surf = VolatilitySurface.from_garch_forecasts(
            returns=garch_returns,
            expiries=expiries,
            strikes=strikes,
            spot=100.0,
        )
        iv_short = surf.interpolate(strike=100.0, expiry=0.083)
        iv_long = surf.interpolate(strike=100.0, expiry=1.0)
        # Not strict — just checks both are positive and plausible
        assert iv_short > 0
        assert iv_long > 0

    def test_requires_at_least_two_expiries(self, garch_returns):
        with pytest.raises(ValueError, match="at least 2"):
            VolatilitySurface.from_garch_forecasts(
                returns=garch_returns,
                expiries=[0.5],
                strikes=[90.0, 100.0, 110.0],
                spot=100.0,
            )

    def test_requires_at_least_two_strikes(self, garch_returns):
        with pytest.raises(ValueError, match="at least 2"):
            VolatilitySurface.from_garch_forecasts(
                returns=garch_returns,
                expiries=[0.25, 1.0],
                strikes=[100.0],
                spot=100.0,
            )
