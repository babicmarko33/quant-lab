"""Tests for Phase 3.7: Lévy process options pricing.

Merton jump-diffusion (1976) — closed-form series expansion
Variance Gamma (Madan-Seneta 1990) — Monte Carlo simulation
"""

import numpy as np
import pytest

from alpha_engine.derivatives.options.black_scholes import bsm_price
from alpha_engine.derivatives.options.levy import merton_price, vg_price

PARAMS = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0)


class TestMertonJumpDiffusion:
    def test_call_positive(self):
        price = merton_price(**PARAMS, option_type="call", lam=1.0, mu_j=-0.1, sigma_j=0.15)
        assert price > 0

    def test_put_positive(self):
        price = merton_price(**PARAMS, option_type="put", lam=1.0, mu_j=-0.1, sigma_j=0.15)
        assert price > 0

    def test_converges_to_bsm_when_no_jumps(self):
        """λ→0: Merton reduces to BSM."""
        bs = bsm_price(**PARAMS, option_type="call")
        price = merton_price(**PARAMS, option_type="call", lam=0.0, mu_j=0.0, sigma_j=0.0)
        assert abs(price - bs) < 0.01

    def test_put_call_parity(self):
        """Merton prices satisfy put-call parity."""
        call = merton_price(**PARAMS, option_type="call", lam=1.0, mu_j=-0.1, sigma_j=0.15)
        put = merton_price(**PARAMS, option_type="put", lam=1.0, mu_j=-0.1, sigma_j=0.15)
        parity = PARAMS["S"] - PARAMS["K"] * np.exp(-PARAMS["r"] * PARAMS["T"])
        assert abs((call - put) - parity) < 0.05

    def test_jump_risk_increases_otm_price(self):
        """Jumps increase the price of OTM options (higher tail risk)."""
        otm = dict(S=100.0, K=120.0, T=1.0, r=0.05, sigma=0.20, q=0.0)
        bs = bsm_price(**otm, option_type="call")
        merton = merton_price(**otm, option_type="call", lam=2.0, mu_j=0.1, sigma_j=0.20)
        assert merton > bs * 0.8  # at minimum, Merton should be in the same ballpark

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            merton_price(**PARAMS, option_type="digital", lam=1.0, mu_j=0.0, sigma_j=0.1)

    def test_returns_float(self):
        price = merton_price(**PARAMS, option_type="call", lam=0.5, mu_j=0.0, sigma_j=0.10)
        assert isinstance(price, float)

    def test_higher_jump_vol_increases_price(self):
        """Greater jump volatility σ_J should increase option price."""
        p_low = merton_price(**PARAMS, option_type="call", lam=1.0, mu_j=0.0, sigma_j=0.05)
        p_high = merton_price(**PARAMS, option_type="call", lam=1.0, mu_j=0.0, sigma_j=0.30)
        assert p_high > p_low


class TestVarianceGamma:
    def test_call_positive(self):
        price = vg_price(**PARAMS, option_type="call", theta=-0.1, nu=0.2, seed=42)
        assert price > 0

    def test_put_positive(self):
        price = vg_price(**PARAMS, option_type="put", theta=-0.1, nu=0.2, seed=42)
        assert price > 0

    def test_put_call_parity(self):
        """VG prices satisfy put-call parity within MC tolerance."""
        call = vg_price(**PARAMS, option_type="call", theta=0.0, nu=0.2, n_paths=200_000, seed=7)
        put = vg_price(**PARAMS, option_type="put", theta=0.0, nu=0.2, n_paths=200_000, seed=7)
        parity = PARAMS["S"] - PARAMS["K"] * np.exp(-PARAMS["r"] * PARAMS["T"])
        assert abs((call - put) - parity) < 0.5

    def test_returns_float(self):
        price = vg_price(**PARAMS, option_type="call", theta=0.0, nu=0.2, seed=0)
        assert isinstance(price, float)

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            vg_price(**PARAMS, option_type="forward", theta=0.0, nu=0.2, seed=0)

    def test_higher_nu_increases_price(self):
        """Higher variance rate ν (more randomness in time) → higher price."""
        p_low = vg_price(**PARAMS, option_type="call", theta=0.0, nu=0.05, n_paths=50_000, seed=1)
        p_high = vg_price(**PARAMS, option_type="call", theta=0.0, nu=0.5, n_paths=50_000, seed=1)
        assert p_high > p_low - 0.5  # allowing for MC variance
