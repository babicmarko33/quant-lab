import pytest

from alpha_engine.derivatives.options.black_scholes import bsm_price
from alpha_engine.derivatives.options.implied_vol import implied_vol


class TestImpliedVol:
    def test_roundtrip_call(self):
        """IV(BSM(sigma)) == sigma for calls."""
        params = dict(S=100, K=100, T=1.0, r=0.05, q=0.0)
        for true_sigma in [0.10, 0.20, 0.30, 0.50, 0.80]:
            price = bsm_price(**params, sigma=true_sigma, option_type="call")
            iv = implied_vol(price, **params, option_type="call")
            assert abs(iv - true_sigma) < 1e-6, f"sigma={true_sigma}, got iv={iv}"

    def test_roundtrip_put(self):
        """IV(BSM(sigma)) == sigma for puts."""
        params = dict(S=100, K=100, T=1.0, r=0.05, q=0.0)
        for true_sigma in [0.10, 0.25, 0.40]:
            price = bsm_price(**params, sigma=true_sigma, option_type="put")
            iv = implied_vol(price, **params, option_type="put")
            assert abs(iv - true_sigma) < 1e-6

    def test_otm_call_roundtrip(self):
        """OTM call: K > S."""
        price = bsm_price(100, 110, 0.5, 0.03, 0.25, option_type="call")
        iv = implied_vol(price, 100, 110, 0.5, 0.03, option_type="call")
        assert abs(iv - 0.25) < 1e-5

    def test_itm_put_roundtrip(self):
        price = bsm_price(100, 110, 1.0, 0.05, 0.30, option_type="put")
        iv = implied_vol(price, 100, 110, 1.0, 0.05, option_type="put")
        assert abs(iv - 0.30) < 1e-5

    def test_price_below_intrinsic_raises(self):
        """Price below intrinsic has no valid IV."""
        with pytest.raises(ValueError, match="below intrinsic"):
            implied_vol(0.001, 100, 100, 1.0, 0.05, option_type="call")

    def test_returns_float(self):
        price = bsm_price(100, 100, 1.0, 0.05, 0.20, option_type="call")
        iv = implied_vol(price, 100, 100, 1.0, 0.05, option_type="call")
        assert isinstance(iv, float)

    def test_high_vol_roundtrip(self):
        """High vol (100%) case."""
        price = bsm_price(100, 100, 1.0, 0.05, 1.0, option_type="call")
        iv = implied_vol(price, 100, 100, 1.0, 0.05, option_type="call")
        assert abs(iv - 1.0) < 1e-4

    def test_short_expiry_roundtrip(self):
        """Short expiry (30 days)."""
        price = bsm_price(100, 100, 30 / 252, 0.05, 0.20, option_type="call")
        iv = implied_vol(price, 100, 100, 30 / 252, 0.05, option_type="call")
        assert abs(iv - 0.20) < 1e-5
