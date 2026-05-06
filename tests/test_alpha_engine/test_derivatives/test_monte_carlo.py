import numpy as np
import pytest

from alpha_engine.derivatives.options.black_scholes import bsm_price
from alpha_engine.derivatives.options.monte_carlo import (
    mc_asian,
    mc_barrier,
    mc_european,
)

PARAMS = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0)


class TestMCEuropean:
    def test_call_converges_to_bsm(self):
        """MC European call should be within 0.5 of analytical BSM."""
        mc_price, _ = mc_european(**PARAMS, option_type="call", n_paths=50_000, seed=42)
        bs_price = bsm_price(**PARAMS, option_type="call")
        assert abs(mc_price - bs_price) < 0.5

    def test_put_converges_to_bsm(self):
        mc_price, _ = mc_european(**PARAMS, option_type="put", n_paths=50_000, seed=42)
        bs_price = bsm_price(**PARAMS, option_type="put")
        assert abs(mc_price - bs_price) < 0.5

    def test_returns_price_and_stderr(self):
        result = mc_european(**PARAMS, option_type="call", n_paths=10_000, seed=0)
        assert len(result) == 2
        price, stderr = result
        assert price > 0
        assert stderr > 0

    def test_antithetic_reduces_variance(self):
        """Antithetic variates should reduce standard error vs plain MC."""
        _, se_plain = mc_european(**PARAMS, option_type="call", n_paths=10_000, seed=1, antithetic=False)
        _, se_anti = mc_european(**PARAMS, option_type="call", n_paths=10_000, seed=1, antithetic=True)
        assert se_anti < se_plain

    def test_call_put_parity_holds(self):
        """MC prices should approximately satisfy put-call parity."""
        call, _ = mc_european(**PARAMS, option_type="call", n_paths=100_000, seed=7)
        put, _ = mc_european(**PARAMS, option_type="put", n_paths=100_000, seed=7)
        # C - P = S*exp(-q*T) - K*exp(-r*T)
        parity = PARAMS["S"] * np.exp(-PARAMS["q"] * PARAMS["T"]) - PARAMS["K"] * np.exp(-PARAMS["r"] * PARAMS["T"])
        assert abs((call - put) - parity) < 0.5

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            mc_european(**PARAMS, option_type="binary", n_paths=100, seed=0)

    def test_higher_vol_higher_price(self):
        lo, _ = mc_european(S=100, K=100, T=1.0, r=0.05, sigma=0.10, option_type="call", n_paths=50_000, seed=0)
        hi, _ = mc_european(S=100, K=100, T=1.0, r=0.05, sigma=0.40, option_type="call", n_paths=50_000, seed=0)
        assert hi > lo


class TestMCAsian:
    def test_arithmetic_call_less_than_european(self):
        """Arithmetic Asian call <= European call (averaging reduces variance)."""
        asian, _ = mc_asian(**PARAMS, option_type="call", averaging="arithmetic", n_paths=50_000, seed=42)
        european, _ = mc_european(**PARAMS, option_type="call", n_paths=50_000, seed=42)
        assert asian <= european + 0.1  # small tolerance for MC noise

    def test_geometric_call_less_than_arithmetic(self):
        """Geometric mean < arithmetic mean → geometric Asian call < arithmetic."""
        arith, _ = mc_asian(**PARAMS, option_type="call", averaging="arithmetic", n_paths=50_000, seed=42)
        geom, _ = mc_asian(**PARAMS, option_type="call", averaging="geometric", n_paths=50_000, seed=42)
        assert geom <= arith + 0.1

    def test_returns_price_and_stderr(self):
        price, stderr = mc_asian(**PARAMS, option_type="call", averaging="arithmetic", n_paths=10_000, seed=0)
        assert price > 0
        assert stderr > 0

    def test_invalid_averaging_raises(self):
        with pytest.raises(ValueError, match="averaging"):
            mc_asian(**PARAMS, option_type="call", averaging="harmonic", n_paths=100, seed=0)


class TestMCBarrier:
    def test_up_and_out_call_less_than_vanilla(self):
        """Up-and-out call: if spot hits barrier, option is knocked out → cheaper than vanilla."""
        vanilla, _ = mc_european(**PARAMS, option_type="call", n_paths=50_000, seed=42)
        barrier, _ = mc_barrier(
            **PARAMS, option_type="call", barrier=120.0,
            barrier_type="up-and-out", n_paths=50_000, seed=42,
        )
        assert barrier <= vanilla + 0.1

    def test_down_and_out_put_less_than_vanilla(self):
        vanilla, _ = mc_european(**PARAMS, option_type="put", n_paths=50_000, seed=42)
        barrier, _ = mc_barrier(
            **PARAMS, option_type="put", barrier=80.0,
            barrier_type="down-and-out", n_paths=50_000, seed=42,
        )
        assert barrier <= vanilla + 0.1

    def test_very_high_barrier_approaches_vanilla(self):
        """With barrier far OTM, almost no paths knock out → price ≈ vanilla."""
        vanilla, _ = mc_european(**PARAMS, option_type="call", n_paths=50_000, seed=42)
        barrier, _ = mc_barrier(
            **PARAMS, option_type="call", barrier=500.0,
            barrier_type="up-and-out", n_paths=50_000, seed=42,
        )
        assert abs(barrier - vanilla) < 1.0

    def test_returns_price_and_stderr(self):
        price, stderr = mc_barrier(
            **PARAMS, option_type="call", barrier=130.0,
            barrier_type="up-and-out", n_paths=10_000, seed=0,
        )
        assert isinstance(price, float)
        assert stderr >= 0

    def test_invalid_barrier_type_raises(self):
        with pytest.raises(ValueError, match="barrier_type"):
            mc_barrier(**PARAMS, option_type="call", barrier=120.0, barrier_type="knock-in-out", n_paths=100, seed=0)
