import numpy as np
import pandas as pd

from alpha_engine.derivatives.options.strategies import (
    covered_call,
    iron_condor,
    protective_put,
    straddle,
    strangle,
)

S_range = np.linspace(50, 150, 200)


class TestCoveredCall:
    def test_returns_pd_series(self):
        pnl = covered_call(S_range, S0=100, K=110, premium=5.0)
        assert isinstance(pnl, pd.Series)

    def test_capped_upside(self):
        """Profit is capped above the strike."""
        pnl = covered_call(S_range, S0=100, K=110, premium=5.0)
        max_profit = 110 - 100 + 5.0
        assert pnl.max() <= max_profit + 0.01

    def test_downside_loss(self):
        """Below S0 - premium the strategy loses money."""
        pnl = covered_call(S_range, S0=100, K=110, premium=5.0)
        assert pnl[pnl.index < 90].min() < 0

    def test_breakeven_approximate(self):
        """Breakeven at S0 - premium = 95."""
        pnl = covered_call(S_range, S0=100, K=110, premium=5.0)
        near_breakeven = pnl[(pnl.index >= 94.5) & (pnl.index <= 95.5)]
        assert near_breakeven.abs().min() < 1.0


class TestProtectivePut:
    def test_returns_pd_series(self):
        pnl = protective_put(S_range, S0=100, K=90, premium=3.0)
        assert isinstance(pnl, pd.Series)

    def test_downside_floored(self):
        """Max loss is K - S0 - premium."""
        pnl = protective_put(S_range, S0=100, K=90, premium=3.0)
        max_loss = -(100 - 90 + 3.0)
        assert pnl.min() >= max_loss - 0.01

    def test_upside_unlimited(self):
        """Unlimited upside above S0 + premium."""
        pnl = protective_put(S_range, S0=100, K=90, premium=3.0)
        assert pnl[pnl.index > 130].mean() > 0


class TestStraddle:
    def test_returns_pd_series(self):
        pnl = straddle(S_range, K=100, call_premium=5.0, put_premium=4.0)
        assert isinstance(pnl, pd.Series)

    def test_profit_far_from_strike(self):
        """Far from strike either side -> positive P&L."""
        pnl = straddle(S_range, K=100, call_premium=5.0, put_premium=4.0)
        assert pnl[pnl.index > 130].mean() > 0
        assert pnl[pnl.index < 70].mean() > 0

    def test_max_loss_at_strike(self):
        """Max loss is total premium paid."""
        pnl = straddle(S_range, K=100, call_premium=5.0, put_premium=4.0)
        max_loss = -(5.0 + 4.0)
        distances = np.abs(pnl.index.values - 100)
        closest_idx = int(distances.argmin())
        assert abs(pnl.iloc[closest_idx] - max_loss) < 0.5


class TestStrangle:
    def test_returns_pd_series(self):
        pnl = strangle(S_range, K_put=90, K_call=110, put_premium=2.0, call_premium=2.0)
        assert isinstance(pnl, pd.Series)

    def test_max_loss_between_strikes(self):
        """Between K_put and K_call, both options expire worthless."""
        pnl = strangle(S_range, K_put=90, K_call=110, put_premium=2.0, call_premium=2.0)
        middle = pnl[(pnl.index >= 91) & (pnl.index <= 109)]
        assert (middle + 4.0).abs().max() < 0.1

    def test_profit_far_otm(self):
        pnl = strangle(S_range, K_put=90, K_call=110, put_premium=2.0, call_premium=2.0)
        assert pnl[pnl.index > 130].mean() > 0
        assert pnl[pnl.index < 70].mean() > 0


class TestIronCondor:
    def test_returns_pd_series(self):
        pnl = iron_condor(S_range, K1=85, K2=90, K3=110, K4=115, net_credit=2.0)
        assert isinstance(pnl, pd.Series)

    def test_max_profit_is_net_credit(self):
        """Inside K2..K3 the profit equals net credit."""
        pnl = iron_condor(S_range, K1=85, K2=90, K3=110, K4=115, net_credit=2.0)
        middle = pnl[(pnl.index >= 91) & (pnl.index <= 109)]
        assert (middle - 2.0).abs().max() < 0.1

    def test_bounded_loss(self):
        """Max loss bounded by wing width minus credit."""
        pnl = iron_condor(S_range, K1=85, K2=90, K3=110, K4=115, net_credit=2.0)
        max_loss = -(5.0 - 2.0)  # wing width = 5
        assert pnl.min() >= max_loss - 0.01
