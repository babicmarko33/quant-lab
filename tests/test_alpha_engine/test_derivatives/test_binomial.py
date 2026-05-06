import numpy as np
import pytest

from alpha_engine.derivatives.options.binomial import binomial_price
from alpha_engine.derivatives.options.black_scholes import bsm_price

PARAMS = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, q=0.0)


class TestBinomialEuropean:
    def test_call_converges_to_bsm(self):
        """CRR binomial call should converge to BSM as steps increase."""
        bs = bsm_price(**PARAMS, option_type="call")
        price = binomial_price(**PARAMS, option_type="call", n_steps=500, american=False)
        assert abs(price - bs) < 0.1

    def test_put_converges_to_bsm(self):
        bs = bsm_price(**PARAMS, option_type="put")
        price = binomial_price(**PARAMS, option_type="put", n_steps=500, american=False)
        assert abs(price - bs) < 0.1

    def test_call_put_parity(self):
        """European binomial prices satisfy put-call parity."""
        call = binomial_price(**PARAMS, option_type="call", n_steps=200, american=False)
        put = binomial_price(**PARAMS, option_type="put", n_steps=200, american=False)
        parity = PARAMS["S"] * np.exp(-PARAMS["q"] * PARAMS["T"]) - PARAMS["K"] * np.exp(-PARAMS["r"] * PARAMS["T"])
        assert abs((call - put) - parity) < 0.1

    def test_returns_positive_price(self):
        price = binomial_price(**PARAMS, option_type="call", n_steps=100, american=False)
        assert price > 0

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError, match="option_type"):
            binomial_price(**PARAMS, option_type="binary", n_steps=50, american=False)

    def test_convergence_improves_with_steps(self):
        """Price with 500 steps is closer to BSM than with 50 steps."""
        bs = bsm_price(**PARAMS, option_type="call")
        p50 = binomial_price(**PARAMS, option_type="call", n_steps=50, american=False)
        p500 = binomial_price(**PARAMS, option_type="call", n_steps=500, american=False)
        assert abs(p500 - bs) < abs(p50 - bs)


class TestBinomialAmerican:
    def test_american_put_ge_european_put(self):
        """American put >= European put (early exercise has value)."""
        eur = binomial_price(**PARAMS, option_type="put", n_steps=200, american=False)
        amer = binomial_price(**PARAMS, option_type="put", n_steps=200, american=True)
        assert amer >= eur - 0.001  # tiny tolerance for floating point

    def test_american_call_equals_european_call_no_dividends(self):
        """With q=0, early exercise of call is never optimal → same price."""
        eur = binomial_price(**PARAMS, option_type="call", n_steps=200, american=False)
        amer = binomial_price(**PARAMS, option_type="call", n_steps=200, american=True)
        assert abs(amer - eur) < 0.01

    def test_american_put_deep_itm_has_early_exercise_premium(self):
        """Deep ITM put: intrinsic value > discounted payoff → early exercise valuable."""
        deep_itm = dict(S=60.0, K=100.0, T=1.0, r=0.10, sigma=0.20, q=0.0)
        eur = binomial_price(**deep_itm, option_type="put", n_steps=200, american=False)
        amer = binomial_price(**deep_itm, option_type="put", n_steps=200, american=True)
        assert amer > eur

    def test_american_call_with_dividends_may_exceed_european(self):
        """High dividend call: early exercise can be optimal with q > 0."""
        high_div = dict(S=100.0, K=90.0, T=1.0, r=0.05, sigma=0.20, q=0.10)
        eur = binomial_price(**high_div, option_type="call", n_steps=200, american=False)
        amer = binomial_price(**high_div, option_type="call", n_steps=200, american=True)
        assert amer >= eur - 0.001
