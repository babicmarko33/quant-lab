# Phase 5: Derivatives Lab Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an institutional-grade derivatives pricing and analysis library covering Black-Scholes, all first- and second-order Greeks, implied volatility solving, multi-leg strategy P&L diagrams, and volatility surface construction.

**Architecture:** Three sub-modules under `src/alpha_engine/derivatives/`:
1. `options/black_scholes.py` — BSM pricing + analytic Greeks (Delta, Gamma, Theta, Vega, Rho)
2. `options/implied_vol.py` — IV solver (Newton-Raphson with bisection fallback)
3. `options/strategies.py` — Multi-leg P&L at expiry (covered call, protective put, straddle, strangle, iron condor)
4. `volatility/surface.py` — Strike/expiry grid construction + cubic spline interpolation

All functions are pure math (no I/O). No new dependencies required — uses `scipy.stats`, `scipy.optimize`, `numpy` already installed.

**Tech Stack:** scipy.stats.norm (BSM), scipy.optimize (IV bisection), numpy (vectorized Greeks), pandas (vol surface DataFrame).

---

## Task 1: Black-Scholes pricing + Greeks

**Files:**
- Create: `src/alpha_engine/derivatives/__init__.py`
- Create: `src/alpha_engine/derivatives/options/__init__.py`
- Create: `src/alpha_engine/derivatives/options/black_scholes.py`
- Create: `tests/test_alpha_engine/test_derivatives/__init__.py`
- Create: `tests/test_alpha_engine/test_derivatives/test_black_scholes.py`

**Mathematical background:**

Black-Scholes formula (continuous dividend yield q):
```
d1 = (ln(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)
call = S*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)
put  = K*exp(-r*T)*N(-d2) - S*exp(-q*T)*N(-d1)
```

Greeks:
- Delta_call = exp(-q*T) * N(d1)
- Delta_put  = exp(-q*T) * (N(d1) - 1)
- Gamma      = exp(-q*T) * n(d1) / (S * sigma * sqrt(T))
- Theta_call = (-S*n(d1)*sigma*exp(-q*T) / (2*sqrt(T))
               - r*K*exp(-r*T)*N(d2) + q*S*exp(-q*T)*N(d1)) / 252
- Vega       = S * exp(-q*T) * n(d1) * sqrt(T) / 100  (per 1% vol move)
- Rho_call   = K * T * exp(-r*T) * N(d2) / 100  (per 1% rate move)

**Step 1: Write failing tests**

```python
# tests/test_alpha_engine/test_derivatives/test_black_scholes.py
import pytest
import numpy as np
from alpha_engine.derivatives.options.black_scholes import bsm_price, bsm_greeks

class TestBSMPrice:
    def test_call_put_parity(self):
        """C - P = S*exp(-q*T) - K*exp(-r*T)"""
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.0
        call = bsm_price(S, K, T, r, sigma, option_type="call", q=q)
        put  = bsm_price(S, K, T, r, sigma, option_type="put",  q=q)
        parity = S * np.exp(-q*T) - K * np.exp(-r*T)
        assert abs((call - put) - parity) < 1e-10

    def test_deep_itm_call_approaches_intrinsic(self):
        """Deep ITM call price approaches S - K*exp(-r*T) (no time value)."""
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
        with pytest.raises(ValueError):
            bsm_price(100, 100, 1.0, 0.05, 0.20, option_type="forward")

    def test_known_value_atm_call(self):
        """ATM call with T=1, r=5%, sigma=20% ~ 10.45 (Merton 1973)."""
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

    def test_put_call_delta_sum(self):
        """Delta_call - Delta_put = exp(-q*T) (put-call delta parity)."""
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.20, 0.02
        gc = bsm_greeks(S, K, T, r, sigma, option_type="call", q=q)
        gp = bsm_greeks(S, K, T, r, sigma, option_type="put",  q=q)
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
        """Higher rates -> higher call price (positive rho)."""
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
```

**Step 2: Run — confirm RED**

```powershell
pytest tests/test_alpha_engine/test_derivatives/test_black_scholes.py -q
```

**Step 3: Implement `black_scholes.py`**

```python
from __future__ import annotations
import numpy as np
from scipy.stats import norm

VALID_TYPES = {"call", "put"}

def bsm_price(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "call", q: float = 0.0) -> float:
    """Black-Scholes-Merton option price with continuous dividend yield."""
    if option_type not in VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1))

def bsm_greeks(S: float, K: float, T: float, r: float, sigma: float,
               option_type: str = "call", q: float = 0.0) -> dict[str, float]:
    """First-order Greeks + Gamma for a BSM option."""
    if option_type not in VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    n_d1 = norm.pdf(d1)
    exp_qT = np.exp(-q * T)
    exp_rT = np.exp(-r * T)

    gamma = float(exp_qT * n_d1 / (S * sigma * sqrt_T))
    vega  = float(S * exp_qT * n_d1 * sqrt_T / 100)  # per 1% vol

    if option_type == "call":
        delta = float(exp_qT * norm.cdf(d1))
        theta = float((-S * n_d1 * sigma * exp_qT / (2 * sqrt_T)
                       - r * K * exp_rT * norm.cdf(d2)
                       + q * S * exp_qT * norm.cdf(d1)) / 252)
        rho   = float(K * T * exp_rT * norm.cdf(d2) / 100)
    else:
        delta = float(exp_qT * (norm.cdf(d1) - 1))
        theta = float((-S * n_d1 * sigma * exp_qT / (2 * sqrt_T)
                       + r * K * exp_rT * norm.cdf(-d2)
                       - q * S * exp_qT * norm.cdf(-d1)) / 252)
        rho   = float(-K * T * exp_rT * norm.cdf(-d2) / 100)

    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}
```

**Step 4: Run — confirm GREEN (18 tests)**

**Step 5: ruff + commit**

```
feat(derivatives): Black-Scholes pricing and Greeks
```

---

## Task 2: Implied Volatility Solver

**Files:**
- Create: `src/alpha_engine/derivatives/options/implied_vol.py`
- Create: `tests/test_alpha_engine/test_derivatives/test_implied_vol.py`

**Algorithm:**
1. Newton-Raphson: `sigma_{n+1} = sigma_n - (BSM(sigma_n) - market_price) / vega(sigma_n)`
2. Fallback to Brent's method (scipy.optimize.brentq) if NR diverges

**Step 1: Write failing tests**

```python
# tests/test_alpha_engine/test_derivatives/test_implied_vol.py
import pytest
import numpy as np
from alpha_engine.derivatives.options.implied_vol import implied_vol
from alpha_engine.derivatives.options.black_scholes import bsm_price

class TestImpliedVol:
    def test_roundtrip_call(self):
        """IV(BSM(sigma)) == sigma."""
        params = dict(S=100, K=100, T=1.0, r=0.05, q=0.0)
        for true_sigma in [0.10, 0.20, 0.30, 0.50, 0.80]:
            price = bsm_price(**params, sigma=true_sigma, option_type="call")
            iv = implied_vol(price, **params, option_type="call")
            assert abs(iv - true_sigma) < 1e-6, f"sigma={true_sigma}, got iv={iv}"

    def test_roundtrip_put(self):
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
        """Price below intrinsic value has no valid IV."""
        with pytest.raises(ValueError, match="below intrinsic"):
            implied_vol(0.001, 100, 100, 1.0, 0.05, option_type="call")

    def test_returns_float(self):
        price = bsm_price(100, 100, 1.0, 0.05, 0.20, option_type="call")
        iv = implied_vol(price, 100, 100, 1.0, 0.05, option_type="call")
        assert isinstance(iv, float)
```

**Step 2: Run — confirm RED**

**Step 3: Implement `implied_vol.py`**

```python
from __future__ import annotations
import numpy as np
from scipy.optimize import brentq
from alpha_engine.derivatives.options.black_scholes import bsm_price, bsm_greeks

def implied_vol(
    market_price: float,
    S: float, K: float, T: float, r: float,
    option_type: str = "call",
    q: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """Solve for implied volatility using Newton-Raphson with Brent fallback."""
    intrinsic = max(0.0, (S - K) * np.exp(-r * T) if option_type == "call"
                    else (K - S) * np.exp(-r * T))
    if market_price <= intrinsic + 1e-10:
        raise ValueError(f"market_price {market_price:.4f} is below intrinsic {intrinsic:.4f}")

    sigma = 0.20  # initial guess
    for _ in range(max_iter):
        price = bsm_price(S, K, T, r, sigma, option_type=option_type, q=q)
        vega = bsm_greeks(S, K, T, r, sigma, option_type=option_type, q=q)["vega"] * 100
        if abs(vega) < 1e-12:
            break
        sigma -= (price - market_price) / vega
        sigma = max(1e-8, min(sigma, 10.0))  # keep in valid range
        if abs(price - market_price) < tol:
            return float(sigma)

    # Brent's method fallback
    def objective(s: float) -> float:
        return bsm_price(S, K, T, r, s, option_type=option_type, q=q) - market_price

    lo, hi = 1e-6, 10.0
    if objective(lo) * objective(hi) > 0:
        raise ValueError("Could not bracket IV — market price may be invalid")
    return float(brentq(objective, lo, hi, xtol=tol))
```

**Step 4: Run — confirm GREEN (8 tests)**

**Step 5: ruff + commit**

```
feat(derivatives): implied volatility solver (Newton-Raphson + Brent fallback)
```

---

## Task 3: Multi-leg Options Strategies

**Files:**
- Create: `src/alpha_engine/derivatives/options/strategies.py`
- Create: `tests/test_alpha_engine/test_derivatives/test_strategies.py`

**Strategy P&L at expiry:**
- `covered_call(S_range, S0, K, premium)` — Long stock + short call
- `protective_put(S_range, S0, K, premium)` — Long stock + long put
- `straddle(S_range, K, call_premium, put_premium)` — Long call + long put
- `strangle(S_range, K_put, K_call, put_premium, call_premium)`
- `iron_condor(S_range, K1, K2, K3, K4, net_credit)`

All return `pd.Series` indexed by `S_range` (underlying price at expiry).

**Step 1: Write failing tests**

```python
# tests/test_alpha_engine/test_derivatives/test_strategies.py
import numpy as np, pandas as pd, pytest
from alpha_engine.derivatives.options.strategies import (
    covered_call, protective_put, straddle, strangle, iron_condor
)

S_range = np.linspace(50, 150, 200)

class TestCoveredCall:
    def test_returns_pd_series(self):
        pnl = covered_call(S_range, S0=100, K=110, premium=5.0)
        assert isinstance(pnl, pd.Series)

    def test_capped_upside(self):
        """Profit capped at K - S0 + premium above strike."""
        pnl = covered_call(S_range, S0=100, K=110, premium=5.0)
        assert pnl[pnl.index > 110].max() <= (110 - 100 + 5.0 + 0.01)

    def test_downside_loss(self):
        """Below S0 - premium the strategy loses money."""
        pnl = covered_call(S_range, S0=100, K=110, premium=5.0)
        assert pnl[pnl.index < 90].min() < 0

class TestProtectivePut:
    def test_returns_pd_series(self):
        pnl = protective_put(S_range, S0=100, K=90, premium=3.0)
        assert isinstance(pnl, pd.Series)

    def test_downside_floored(self):
        """Max loss is K - S0 - premium."""
        pnl = protective_put(S_range, S0=100, K=90, premium=3.0)
        max_loss = -(100 - 90 + 3.0)
        assert pnl.min() >= max_loss - 0.01

class TestStraddle:
    def test_returns_pd_series(self):
        pnl = straddle(S_range, K=100, call_premium=5.0, put_premium=4.0)
        assert isinstance(pnl, pd.Series)

    def test_profit_far_from_strike(self):
        """Far OTM either side → positive P&L."""
        pnl = straddle(S_range, K=100, call_premium=5.0, put_premium=4.0)
        assert pnl[pnl.index > 130].mean() > 0
        assert pnl[pnl.index < 70].mean() > 0

    def test_max_loss_at_strike(self):
        """Max loss is total premium paid."""
        pnl = straddle(S_range, K=100, call_premium=5.0, put_premium=4.0)
        max_loss = -(5.0 + 4.0)
        assert abs(pnl[pnl.index == 100].iloc[0] - max_loss) < 0.1

class TestIronCondor:
    def test_returns_pd_series(self):
        pnl = iron_condor(S_range, K1=85, K2=90, K3=110, K4=115, net_credit=2.0)
        assert isinstance(pnl, pd.Series)

    def test_max_profit_is_net_credit(self):
        """P&L inside K2..K3 equals net credit."""
        pnl = iron_condor(S_range, K1=85, K2=90, K3=110, K4=115, net_credit=2.0)
        middle = pnl[(pnl.index >= 91) & (pnl.index <= 109)]
        assert (middle - 2.0).abs().max() < 0.1

    def test_bounded_loss(self):
        """Max loss bounded by wing width minus credit."""
        pnl = iron_condor(S_range, K1=85, K2=90, K3=110, K4=115, net_credit=2.0)
        max_loss = -(5.0 - 2.0)  # wing width = 5
        assert pnl.min() >= max_loss - 0.01
```

**Step 2: Run — confirm RED**

**Step 3: Implement `strategies.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd

def _call_payoff(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(S - K, 0.0)

def _put_payoff(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - S, 0.0)

def covered_call(S_range: np.ndarray, S0: float, K: float, premium: float) -> pd.Series:
    """Long stock + short call. P&L at expiry."""
    stock_pnl = S_range - S0
    short_call = premium - _call_payoff(S_range, K)
    return pd.Series(stock_pnl + short_call, index=S_range, name="covered_call")

def protective_put(S_range: np.ndarray, S0: float, K: float, premium: float) -> pd.Series:
    """Long stock + long put. P&L at expiry."""
    stock_pnl = S_range - S0
    long_put = _put_payoff(S_range, K) - premium
    return pd.Series(stock_pnl + long_put, index=S_range, name="protective_put")

def straddle(S_range: np.ndarray, K: float, call_premium: float, put_premium: float) -> pd.Series:
    """Long call + long put. P&L at expiry."""
    pnl = (_call_payoff(S_range, K) - call_premium
           + _put_payoff(S_range, K) - put_premium)
    return pd.Series(pnl, index=S_range, name="straddle")

def strangle(S_range: np.ndarray, K_put: float, K_call: float,
             put_premium: float, call_premium: float) -> pd.Series:
    """Long OTM put + long OTM call. P&L at expiry."""
    pnl = (_call_payoff(S_range, K_call) - call_premium
           + _put_payoff(S_range, K_put) - put_premium)
    return pd.Series(pnl, index=S_range, name="strangle")

def iron_condor(S_range: np.ndarray, K1: float, K2: float, K3: float,
                K4: float, net_credit: float) -> pd.Series:
    """Iron condor: short K2/K3 strangle + long K1/K4 wings.
    K1 < K2 < K3 < K4. Net credit received upfront."""
    short_put_spread  = _put_payoff(S_range, K2)  - _put_payoff(S_range, K1)
    short_call_spread = _call_payoff(S_range, K3) - _call_payoff(S_range, K4)
    pnl = net_credit - short_put_spread - short_call_spread
    return pd.Series(pnl, index=S_range, name="iron_condor")
```

**Step 4: Run — confirm GREEN (12 tests)**

**Step 5: ruff + commit**

```
feat(derivatives): multi-leg options strategies (covered call, straddle, iron condor)
```

---

## Task 4: Volatility Surface

**Files:**
- Create: `src/alpha_engine/derivatives/volatility/__init__.py`
- Create: `src/alpha_engine/derivatives/volatility/surface.py`
- Create: `tests/test_alpha_engine/test_derivatives/test_vol_surface.py`

**Domain:** Build a volatility surface from a sparse grid of (strike, expiry, implied_vol) observations. Interpolate across both dimensions using cubic splines.

**Step 1: Write failing tests**

```python
# tests/test_alpha_engine/test_derivatives/test_vol_surface.py
import numpy as np, pandas as pd, pytest
from alpha_engine.derivatives.volatility.surface import VolatilitySurface

@pytest.fixture
def sample_surface_data():
    """3 expiries x 5 strikes = 15 data points."""
    expiries = [0.25, 0.5, 1.0]
    strikes  = [80, 90, 100, 110, 120]
    rows = []
    for T in expiries:
        for K in strikes:
            # Smile: vol higher away from ATM
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
        """Interpolated value between 90 and 110 is between their IVs."""
        surf = VolatilitySurface()
        surf.fit(sample_surface_data)
        iv_90  = surf.interpolate(strike=90,  expiry=0.5)
        iv_110 = surf.interpolate(strike=110, expiry=0.5)
        iv_100 = surf.interpolate(strike=100, expiry=0.5)
        assert min(iv_90, iv_110) < iv_100 < max(iv_90, iv_110) + 0.01

    def test_interpolate_returns_positive_iv(self, sample_surface_data):
        surf = VolatilitySurface()
        surf.fit(sample_surface_data)
        iv = surf.interpolate(strike=95, expiry=0.75)
        assert iv > 0

    def test_surface_dataframe_shape(self, sample_surface_data):
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
```

**Step 2: Run — confirm RED**

**Step 3: Implement `surface.py`**

```python
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

REQUIRED_COLS = {"expiry", "strike", "iv"}

class VolatilitySurface:
    """Cubic spline volatility surface.

    Fits a RectBivariateSpline on a (strike, expiry) grid of implied vols.
    Interpolates to any (strike, expiry) within the fitted range.
    """

    def __init__(self) -> None:
        self._spline: RectBivariateSpline | None = None
        self._strikes: np.ndarray | None = None
        self._expiries: np.ndarray | None = None

    def fit(self, data: pd.DataFrame) -> None:
        missing = REQUIRED_COLS - set(data.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        pivoted = data.pivot(index="expiry", columns="strike", values="iv").sort_index()
        self._expiries = pivoted.index.values.astype(float)
        self._strikes  = pivoted.columns.values.astype(float)
        z = pivoted.values
        self._spline = RectBivariateSpline(self._expiries, self._strikes, z, kx=min(3, len(self._expiries)-1), ky=min(3, len(self._strikes)-1))

    def interpolate(self, strike: float, expiry: float) -> float:
        if self._spline is None:
            raise RuntimeError("VolatilitySurface is not fitted. Call fit() first.")
        return float(self._spline(expiry, strike))

    def surface_grid(self, strikes: list[float], expiries: list[float]) -> pd.DataFrame:
        if self._spline is None:
            raise RuntimeError("VolatilitySurface is not fitted. Call fit() first.")
        grid = self._spline(expiries, strikes)
        return pd.DataFrame(grid, index=expiries, columns=strikes)
```

**Step 4: Run — confirm GREEN (7 tests)**

**Step 5: ruff + commit**

```
feat(derivatives): volatility surface with cubic spline interpolation
```

---

## Task 5: Full verification + push

**Step 1: Full suite**

```powershell
pytest -m "not integration" -q --tb=short
```

Expected: >= 245 tests passing.

**Step 2: ruff clean**

```powershell
ruff check src/ tests/ scripts/
```

**Step 3: Run smoke test**

```powershell
python scripts/run_backtest.py
```

Expected: all strategies produce output (no exceptions).

**Step 4: Update README badge** (test count: 190 -> ~245)

**Step 5: Commit + push**

```
feat(phase5): Derivatives Lab -- BSM, Greeks, IV solver, strategies, vol surface (~55 tests)
git push origin main
npx gitnexus analyze
```

---

## Test File Locations Summary

| File | Tests |
|------|-------|
| `tests/test_alpha_engine/test_derivatives/test_black_scholes.py` | Put-call parity, Greeks sign/bounds, known values |
| `tests/test_alpha_engine/test_derivatives/test_implied_vol.py` | IV roundtrip, edge cases |
| `tests/test_alpha_engine/test_derivatives/test_strategies.py` | P&L profiles, payoff caps/floors |
| `tests/test_alpha_engine/test_derivatives/test_vol_surface.py` | Interpolation accuracy, shape |

---

## Key Mathematical Properties (for reviewers)

| Property | Formula | Test |
|----------|---------|------|
| Put-call parity | C - P = S·exp(-qT) - K·exp(-rT) | `test_call_put_parity` |
| Delta parity | Δ_call - Δ_put = exp(-qT) | `test_put_call_delta_sum` |
| Gamma always ≥ 0 | Second derivative of price w.r.t. S | `test_gamma_positive` |
| Vega always ≥ 0 | Higher vol → higher option value | `test_vega_positive` |
| IV roundtrip | IV(BSM(sigma)) == sigma | `test_roundtrip_call/put` |

---

## No New Dependencies

Phase 5 uses only already-installed packages:
- `scipy.stats.norm` — BSM N(d1), N(d2)
- `scipy.optimize.brentq` — IV solver fallback
- `scipy.interpolate.RectBivariateSpline` — vol surface
- `numpy`, `pandas` — everywhere
