"""Core result types for the backtesting engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    """Immutable container for a single backtest run's performance.

    Parameters
    ----------
    returns : pd.Series
        Daily arithmetic returns (not log). Index: DatetimeIndex.
    n_trades : int
        Total number of round-trip trades executed.
    """

    returns: pd.Series
    n_trades: int = 0
    risk_free_rate: float = 0.0  # annual, e.g. 0.05 for 5%

    # Computed on first access — stored here to avoid re-computation
    _equity_curve: pd.Series = field(default=None, init=False, repr=False)
    _total_return: float = field(default=None, init=False, repr=False)
    _sharpe: float = field(default=None, init=False, repr=False)
    _max_drawdown: float = field(default=None, init=False, repr=False)
    _calmar: float = field(default=None, init=False, repr=False)

    # ---------- computed properties ----------

    @property
    def equity_curve(self) -> pd.Series:
        """Cumulative equity curve starting at 1.0."""
        if self._equity_curve is None:
            object.__setattr__(self, "_equity_curve", (1 + self.returns).cumprod())
            # Prepend 1.0 at the start so curve visually starts at initial capital
            start = pd.Series([1.0], index=[self.returns.index[0] - pd.Timedelta(days=1)])
            ec = pd.concat([start, self._equity_curve])
            object.__setattr__(self, "_equity_curve", ec)
        return self._equity_curve

    @property
    def total_return(self) -> float:
        """Total return over the period as a fraction (e.g. 0.25 = 25%)."""
        if self._total_return is None:
            val = float((1 + self.returns).prod() - 1.0)
            object.__setattr__(self, "_total_return", val)
        return self._total_return

    @property
    def sharpe(self) -> float:
        """Annualized Sharpe Ratio (using 252 trading days)."""
        if self._sharpe is None:
            rf_daily = (1 + self.risk_free_rate) ** (1 / 252) - 1
            excess = self.returns - rf_daily
            std = excess.std(ddof=1)
            if std == 0 or math.isnan(std):
                object.__setattr__(self, "_sharpe", 0.0)
            else:
                object.__setattr__(self, "_sharpe", float(excess.mean() / std * math.sqrt(252)))
        return self._sharpe

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown as a negative fraction (e.g. -0.35 = -35%)."""
        if self._max_drawdown is None:
            ec = (1 + self.returns).cumprod()
            rolling_peak = ec.cummax()
            drawdown = ec / rolling_peak - 1
            object.__setattr__(self, "_max_drawdown", float(drawdown.min()))
        return self._max_drawdown

    @property
    def calmar(self) -> float:
        """Calmar Ratio: annualized return / |max drawdown|."""
        if self._calmar is None:
            n = len(self.returns)
            ann_return = (1 + self.total_return) ** (252 / max(n, 1)) - 1
            mdd = abs(self.max_drawdown)
            if mdd < 1e-10:
                object.__setattr__(self, "_calmar", float("inf") if ann_return > 0 else 0.0)
            else:
                object.__setattr__(self, "_calmar", float(ann_return / mdd))
        return self._calmar

    def __repr__(self) -> str:
        return (
            f"BacktestResult("
            f"total_return={self.total_return:.2%}, "
            f"sharpe={self.sharpe:.2f}, "
            f"max_drawdown={self.max_drawdown:.2%}, "
            f"calmar={self.calmar:.2f}, "
            f"n_trades={self.n_trades})"
        )

    def to_dict(self) -> dict:
        """Serializable summary dict for reporting and logging."""
        return {
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "n_trades": self.n_trades,
            "n_days": len(self.returns),
        }
