from __future__ import annotations

import numpy as np
import pandas as pd


def _call_payoff(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(S - K, 0.0)


def _put_payoff(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - S, 0.0)


def covered_call(
    S_range: np.ndarray,
    S0: float,
    K: float,
    premium: float,
) -> pd.Series:
    """Long stock (entered at S0) + short call. P&L at expiry."""
    stock_pnl = S_range - S0
    short_call = premium - _call_payoff(S_range, K)
    return pd.Series(stock_pnl + short_call, index=S_range, name="covered_call")


def protective_put(
    S_range: np.ndarray,
    S0: float,
    K: float,
    premium: float,
) -> pd.Series:
    """Long stock (entered at S0) + long put. P&L at expiry."""
    stock_pnl = S_range - S0
    long_put = _put_payoff(S_range, K) - premium
    return pd.Series(stock_pnl + long_put, index=S_range, name="protective_put")


def straddle(
    S_range: np.ndarray,
    K: float,
    call_premium: float,
    put_premium: float,
) -> pd.Series:
    """Long call + long put at same strike. P&L at expiry."""
    pnl = (
        _call_payoff(S_range, K) - call_premium
        + _put_payoff(S_range, K) - put_premium
    )
    return pd.Series(pnl, index=S_range, name="straddle")


def strangle(
    S_range: np.ndarray,
    K_put: float,
    K_call: float,
    put_premium: float,
    call_premium: float,
) -> pd.Series:
    """Long OTM put (K_put) + long OTM call (K_call). P&L at expiry."""
    pnl = (
        _call_payoff(S_range, K_call) - call_premium
        + _put_payoff(S_range, K_put) - put_premium
    )
    return pd.Series(pnl, index=S_range, name="strangle")


def iron_condor(
    S_range: np.ndarray,
    K1: float,
    K2: float,
    K3: float,
    K4: float,
    net_credit: float,
) -> pd.Series:
    """Iron condor: sell K2/K3 strangle, buy K1/K4 wing protection.
    K1 < K2 < K3 < K4. net_credit is the premium received upfront."""
    short_put_spread = _put_payoff(S_range, K2) - _put_payoff(S_range, K1)
    short_call_spread = _call_payoff(S_range, K3) - _call_payoff(S_range, K4)
    pnl = net_credit - short_put_spread - short_call_spread
    return pd.Series(pnl, index=S_range, name="iron_condor")
