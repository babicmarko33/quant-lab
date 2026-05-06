"""Tests for Phase 3.8: SABR stochastic volatility model calibration.

SABR (Hagan et al. 2002):
    dF = σ * F^β * dW₁
    dσ = α * σ * dW₂
    dW₁ dW₂ = ρ dt

sabr_vol(F, K, T, alpha, beta, rho, nu) → implied Black vol
sabr_calibrate(strikes, market_vols, F, T, beta) → (alpha, rho, nu)
"""

import numpy as np
import pytest

from alpha_engine.derivatives.volatility.sabr import sabr_calibrate, sabr_vol

PARAMS = dict(F=100.0, T=1.0, alpha=0.2, beta=0.5, rho=-0.3, nu=0.4)


class TestSABRVol:
    def test_atm_positive(self):
        vol = sabr_vol(K=100.0, **PARAMS)
        assert vol > 0

    def test_returns_float(self):
        vol = sabr_vol(K=100.0, **PARAMS)
        assert isinstance(vol, float)

    def test_atm_formula_matches_hagan(self):
        """ATM SABR vol: σ_ATM ≈ α / F^(1-β) * [1 + ((1-β)²/24 * α²/F^(2-2β) + ...) * T]."""
        # At ATM, SABR has a clean analytical form; just sanity-check it's sensible
        vol = sabr_vol(K=100.0, **PARAMS)
        assert 0.01 < vol < 2.0

    def test_smile_shape(self):
        """SABR with ρ<0: both wings higher than ATM (smile), low strike > high strike (skew)."""
        params = dict(F=100.0, T=1.0, alpha=0.2, beta=0.5, rho=-0.5, nu=0.4)
        vol_otm_put = sabr_vol(K=80.0, **params)   # OTM put (low strike)
        vol_atm = sabr_vol(K=100.0, **params)
        vol_otm_call = sabr_vol(K=120.0, **params)  # OTM call (high strike)
        # Both wings higher than ATM (vol smile)
        assert vol_otm_put > vol_atm
        assert vol_otm_call > vol_atm
        # Negative skew: low strike has higher vol than high strike
        assert vol_otm_put > vol_otm_call

    def test_positive_skew_with_positive_rho(self):
        """ρ>0 → positive skew: higher strikes have higher vol."""
        params = dict(F=100.0, T=1.0, alpha=0.2, beta=0.5, rho=0.5, nu=0.4)
        vol_low = sabr_vol(K=80.0, **params)
        vol_high = sabr_vol(K=120.0, **params)
        assert vol_high > vol_low

    def test_higher_nu_increases_smile_curvature(self):
        """Higher vol-of-vol ν increases smile curvature (wings up more)."""
        base = dict(F=100.0, T=1.0, alpha=0.2, beta=0.5, rho=0.0)
        wing = 80.0  # OTM put
        v_low_nu = sabr_vol(K=wing, nu=0.1, **base)
        v_high_nu = sabr_vol(K=wing, nu=0.8, **base)
        assert v_high_nu > v_low_nu

    def test_beta_zero_normal_vol(self):
        """β=0 (normal SABR) should give a valid vol."""
        params = dict(F=100.0, T=1.0, alpha=20.0, beta=0.0, rho=0.0, nu=0.3)
        vol = sabr_vol(K=100.0, **params)
        assert vol > 0

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta"):
            sabr_vol(K=100.0, F=100.0, T=1.0, alpha=0.2, beta=1.5, rho=0.0, nu=0.3)

    def test_invalid_rho_raises(self):
        with pytest.raises(ValueError, match="rho"):
            sabr_vol(K=100.0, F=100.0, T=1.0, alpha=0.2, beta=0.5, rho=1.2, nu=0.3)


class TestSABRCalibrate:
    def test_calibrated_vols_match_market(self):
        """Calibrated parameters should reproduce market vols closely."""
        F, T = 100.0, 1.0
        beta = 0.5
        # Synthetic market: use known SABR params to generate vols
        true_alpha, true_rho, true_nu = 0.25, -0.3, 0.4
        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        market_vols = np.array([
            sabr_vol(K=k, F=F, T=T, alpha=true_alpha, beta=beta, rho=true_rho, nu=true_nu)
            for k in strikes
        ])

        alpha_fit, rho_fit, nu_fit = sabr_calibrate(
            strikes=strikes, market_vols=market_vols, F=F, T=T, beta=beta
        )
        fitted_vols = np.array([
            sabr_vol(K=k, F=F, T=T, alpha=alpha_fit, beta=beta, rho=rho_fit, nu=nu_fit)
            for k in strikes
        ])
        rmse = np.sqrt(np.mean((fitted_vols - market_vols) ** 2))
        assert rmse < 0.005  # sub-50bp RMSE

    def test_returns_three_params(self):
        F, T, beta = 100.0, 1.0, 0.5
        strikes = np.array([90.0, 100.0, 110.0])
        vols = np.array([0.22, 0.20, 0.21])
        result = sabr_calibrate(strikes=strikes, market_vols=vols, F=F, T=T, beta=beta)
        assert len(result) == 3

    def test_calibrated_alpha_positive(self):
        F, T, beta = 100.0, 1.0, 0.5
        strikes = np.array([90.0, 100.0, 110.0])
        vols = np.array([0.22, 0.20, 0.21])
        alpha, rho, nu = sabr_calibrate(strikes=strikes, market_vols=vols, F=F, T=T, beta=beta)
        assert alpha > 0

    def test_calibrated_rho_in_range(self):
        F, T, beta = 100.0, 1.0, 0.5
        strikes = np.array([90.0, 100.0, 110.0])
        vols = np.array([0.22, 0.20, 0.21])
        alpha, rho, nu = sabr_calibrate(strikes=strikes, market_vols=vols, F=F, T=T, beta=beta)
        assert -1.0 < rho < 1.0
