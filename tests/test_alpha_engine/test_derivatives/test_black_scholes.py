import numpy as np
import pytest

from alpha_engine.derivatives.options.black_scholes import bsm_greeks, bsm_price


class TestBSMPrice:
    def test_call_put_parity(self):
        """C - P = S*exp(-q*T) - K*exp(-r*T)"""
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.0
        call = bsm_price(S, K, T, r, sigma, option_type="call", q=q)
        put = bsm_price(S, K, T, r, sigma, option_type="put", q=q)
        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 1e-10

    def test_deep_itm_call_approaches_intrinsic(self):
        """Deep ITM call price approaches S - K*exp(-r*T)."""
        call = bsm_price(200, 100, 1.0, 0.05, 0.01, option_type="call")
        intrinsic = 200 - 100 * np.exp(-0.05)
        assert abs(call - intrinsic) < 0.5

    def test_deep_otm_call_near_zero(self):
        call = bsm_price(50, 200, 1.0, 0.05, 0.20, option_type="call")
        assert call < 0.01

    def test_call_price_positive(self):
        call = bsm_price(100, 100, 1.0, 0.05, 0.20, option_type="call")
        assert call > 0

    def test_put_price_positive(self):
        put = bsm_price(100, 100, 1.0, 0.05, 0.20, option_type="put")
        assert put > 0

    def test_higher_vol_higher_price(self):
        lo = bsm_price(100, 100, 1.0, 0.05, 0.10, option_type="call")
        hi = bsm_price(100, 100, 1.0, 0.05, 0.40, option_type="call")
        assert hi > lo

    def test_invalid_option_type(self):
        with pytest.raises(ValueError, match="option_type"):
            bsm_price(100, 100, 1.0, 0.05, 0.20, option_type="forward")

    def test_known_value_atm_call(self):
        """ATM call with T=1, r=5%, sigma=20% ~ 10.45."""
        call = bsm_price(100, 100, 1.0, 0.05, 0.20, option_type="call")
        assert abs(call - 10.45) < 0.5


class TestBSMGreeks:
    def test_call_delta_between_0_and_1(self):
        g = bsm_greeks(100, 100, 1.0, 0.05, 0.20, option_type="call")
        assert 0 < g["delta"] < 1

    def test_put_delta_between_minus1_and_0(self):
        g = bsm_greeks(100, 100, 1.0, 0.05, 0.20, option_type="put")
        assert -1 < g["delta"] < 0

    def test_atm_call_delta_near_half(self):
        """ATM call delta ~ 0.5 for short T."""
        g = bsm_greeks(100, 100, 0.01, 0.0, 0.20, option_type="call")
        assert abs(g["delta"] - 0.5) < 0.05

    def test_put_call_delta_parity(self):
        """Delta_call - Delta_put = exp(-q*T)."""
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.02
        gc = bsm_greeks(S, K, T, r, sigma, option_type="call", q=q)
        gp = bsm_greeks(S, K, T, r, sigma, option_type="put", q=q)
        assert abs((gc["delta"] - gp["delta"]) - np.exp(-q * T)) < 1e-10

    def test_gamma_positive(self):
        g = bsm_greeks(100, 100, 1.0, 0.05, 0.20, option_type="call")
        assert g["gamma"] > 0

    def test_vega_positive(self):
        g = bsm_greeks(100, 100, 1.0, 0.05, 0.20, option_type="call")
        assert g["vega"] > 0

    def test_call_theta_negative(self):
        """Long call loses time value (theta < 0)."""
        g = bsm_greeks(100, 100, 1.0, 0.05, 0.20, option_type="call")
        assert g["theta"] < 0

    def test_call_rho_positive(self):
        """Higher rates -> higher call price."""
        g = bsm_greeks(100, 100, 1.0, 0.05, 0.20, option_type="call")
        assert g["rho"] > 0

    def test_put_rho_negative(self):
        g = bsm_greeks(100, 100, 1.0, 0.05, 0.20, option_type="put")
        assert g["rho"] < 0

    def test_greeks_dict_keys(self):
        g = bsm_greeks(100, 100, 1.0, 0.05, 0.20, option_type="call")
        assert set(g.keys()) == {"delta", "gamma", "theta", "vega", "rho"}

    def test_deep_itm_call_delta_near_one(self):
        g = bsm_greeks(300, 100, 1.0, 0.05, 0.20, option_type="call")
        assert g["delta"] > 0.99

    def test_deep_otm_call_delta_near_zero(self):
        g = bsm_greeks(50, 200, 1.0, 0.05, 0.20, option_type="call")
        assert g["delta"] < 0.01
