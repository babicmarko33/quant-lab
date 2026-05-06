"""Tests for Phase 3.5-3.6: PDE finite difference methods for options pricing.

Phase 3.5: European Crank-Nicolson (θ-scheme, θ=0.5)
Phase 3.6: American PSOR (Projected Successive Over-Relaxation)
"""

import numpy as np
import pytest

from alpha_engine.derivatives.options.black_scholes import bsm_price
from alpha_engine.derivatives.options.pde import pde_american, pde_european

PARAMS = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0)


class TestPDEEuropean:
    def test_call_converges_to_bsm(self):
        """Crank-Nicolson European call should converge to BSM."""
        bs = bsm_price(**PARAMS, option_type="call")
        price = pde_european(**PARAMS, option_type="call", n_s=200, n_t=200)
        assert abs(price - bs) < 0.05

    def test_put_converges_to_bsm(self):
        bs = bsm_price(**PARAMS, option_type="put")
        price = pde_european(**PARAMS, option_type="put", n_s=200, n_t=200)
        assert abs(price - bs) < 0.05

    def test_call_put_parity(self):
        """European PDE prices satisfy put-call parity."""
        call = pde_european(**PARAMS, option_type="call", n_s=200, n_t=200)
        put = pde_european(**PARAMS, option_type="put", n_s=200, n_t=200)
        parity = PARAMS["S"] - PARAMS["K"] * np.exp(-PARAMS["r"] * PARAMS["T"])
        assert abs((call - put) - parity) < 0.1

    def test_itm_call_has_positive_intrinsic(self):
        """Deep ITM call price > intrinsic value (time value > 0)."""
        params = dict(S=120.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0)
        price = pde_european(**params, option_type="call", n_s=200, n_t=200)
        assert price > 20.0

    def test_otm_call_approaches_zero(self):
        """Far OTM call should be close to zero."""
        params = dict(S=50.0, K=100.0, T=0.25, r=0.05, sigma=0.10, q=0.0)
        price = pde_european(**params, option_type="call", n_s=200, n_t=200)
        assert price < 0.01

    def test_returns_positive_float(self):
        price = pde_european(**PARAMS, option_type="call", n_s=100, n_t=100)
        assert isinstance(price, float)
        assert price > 0

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            pde_european(**PARAMS, option_type="binary", n_s=50, n_t=50)

    def test_call_with_dividends(self):
        """European call with dividend yield: PDE should match BSM."""
        params = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.03)
        bs = bsm_price(**params, option_type="call")
        price = pde_european(**params, option_type="call", n_s=200, n_t=200)
        assert abs(price - bs) < 0.1


class TestPDEAmerican:
    def test_american_put_ge_european_put(self):
        """American put >= European put (early exercise premium)."""
        eur = pde_european(**PARAMS, option_type="put", n_s=200, n_t=200)
        amer = pde_american(**PARAMS, option_type="put", n_s=200, n_t=200)
        assert amer >= eur - 0.01

    def test_american_put_ge_intrinsic(self):
        """American put price >= max(K - S, 0) at all times."""
        price = pde_american(**PARAMS, option_type="put", n_s=200, n_t=200)
        intrinsic = max(PARAMS["K"] - PARAMS["S"], 0.0)
        assert price >= intrinsic - 0.01

    def test_american_call_no_dividends_equals_european(self):
        """American call with q=0 has no early exercise premium."""
        eur = pde_european(**PARAMS, option_type="call", n_s=200, n_t=200)
        amer = pde_american(**PARAMS, option_type="call", n_s=200, n_t=200)
        assert abs(amer - eur) < 0.05

    def test_american_deep_itm_put_early_exercise(self):
        """Deep ITM put: American > European due to early exercise."""
        deep = dict(S=60.0, K=100.0, T=1.0, r=0.10, sigma=0.20, q=0.0)
        eur = pde_european(**deep, option_type="put", n_s=200, n_t=200)
        amer = pde_american(**deep, option_type="put", n_s=200, n_t=200)
        assert amer > eur

    def test_american_put_greater_than_zero(self):
        price = pde_american(**PARAMS, option_type="put", n_s=100, n_t=100)
        assert price > 0

    def test_returns_float(self):
        price = pde_american(**PARAMS, option_type="put", n_s=100, n_t=100)
        assert isinstance(price, float)

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            pde_american(**PARAMS, option_type="forward", n_s=50, n_t=50)
