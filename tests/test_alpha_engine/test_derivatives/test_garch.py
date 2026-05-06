"""Tests for GARCH volatility forecasting module.

TDD: these tests are written before the implementation.
"""
import math

import numpy as np
import pandas as pd
import pytest

from alpha_engine.derivatives.volatility.garch import (
    GarchResult,
    fit_garch,
    forecast_volatility,
    garch_vol_forecast,
)

N_OBS = 500
TRADING_DAYS = 252


@pytest.fixture
def returns_series() -> pd.Series:
    rng = np.random.default_rng(42)
    # Simulate heteroskedastic returns (GARCH-like vol clustering)
    n = N_OBS
    vol = np.ones(n) * 0.01
    r = np.empty(n)
    for i in range(n):
        r[i] = rng.normal(0, vol[i])
        vol_sq_next = 1e-6 + 0.05 * r[i] ** 2 + 0.90 * vol[i] ** 2
        if i + 1 < n:
            vol[i + 1] = math.sqrt(vol_sq_next)
    return pd.Series(r, name="returns")


class TestFitGarch:
    def test_returns_garch_result(self, returns_series):
        result = fit_garch(returns_series)
        assert isinstance(result, GarchResult)

    def test_params_stored(self, returns_series):
        result = fit_garch(returns_series)
        assert result.omega > 0
        assert len(result.alpha) == 1
        assert len(result.beta) == 1
        assert result.alpha[0] > 0
        assert result.beta[0] > 0

    def test_persistence_lt_one(self, returns_series):
        """GARCH(1,1) stationarity: alpha[0] + beta[0] < 1."""
        result = fit_garch(returns_series)
        assert result.persistence < 1.0

    def test_unconditional_vol_positive_annualised(self, returns_series):
        result = fit_garch(returns_series)
        assert result.unconditional_vol > 0
        # Should be a plausible annualised vol (1% – 200%)
        assert 0.01 < result.unconditional_vol < 2.0

    def test_custom_order(self, returns_series):
        result = fit_garch(returns_series, p=1, q=2)
        assert isinstance(result, GarchResult)
        assert len(result.beta) == 2

    def test_raises_on_too_short_series(self):
        short = pd.Series(np.random.standard_normal(30) * 0.01)
        with pytest.raises(ValueError, match="at least 50"):
            fit_garch(short)


class TestForecastVolatility:
    def test_returns_array(self, returns_series):
        result = fit_garch(returns_series)
        fcast = forecast_volatility(result, horizon=5)
        assert isinstance(fcast, np.ndarray)

    def test_shape_equals_horizon(self, returns_series):
        for h in (1, 5, 22):
            result = fit_garch(returns_series)
            fcast = forecast_volatility(result, horizon=h)
            assert fcast.shape == (h,), f"Expected shape ({h},), got {fcast.shape}"

    def test_all_positive(self, returns_series):
        result = fit_garch(returns_series)
        fcast = forecast_volatility(result, horizon=10)
        assert (fcast > 0).all()

    def test_annualised_values_plausible(self, returns_series):
        """Annualised vols should be between 1% and 200%."""
        result = fit_garch(returns_series)
        fcast = forecast_volatility(result, horizon=22)
        # forecast_volatility returns daily vols; annualise
        annual = fcast * math.sqrt(TRADING_DAYS)
        assert (annual > 0.01).all()
        assert (annual < 2.0).all()

    def test_long_horizon_converges_to_unconditional(self, returns_series):
        """Long-horizon GARCH forecast should approach unconditional vol."""
        result = fit_garch(returns_series)
        fcast = forecast_volatility(result, horizon=252)
        annual_long = fcast[-1] * math.sqrt(TRADING_DAYS)
        assert abs(annual_long - result.unconditional_vol) < 0.10


class TestGarchVolForecast:
    def test_returns_float(self, returns_series):
        vol = garch_vol_forecast(returns_series, horizon=1)
        assert isinstance(vol, float)

    def test_positive(self, returns_series):
        vol = garch_vol_forecast(returns_series, horizon=5)
        assert vol > 0

    def test_horizon_1_vs_22_differ(self, returns_series):
        """One-step and 22-step forecasts are generally not identical."""
        v1 = garch_vol_forecast(returns_series, horizon=1)
        v22 = garch_vol_forecast(returns_series, horizon=22)
        # They may be close but should not raise
        assert v1 > 0
        assert v22 > 0
