"""Regime-filtered strategy decorator.

Wraps any Strategy and zeros out signals in non-active regimes,
classified by a GaussianHMM fitted on the close returns.
"""
from __future__ import annotations

import pandas as pd

from alpha_engine.regime.hmm_classifier import RegimeClassifier
from alpha_engine.strategies.base import Strategy


class RegimeFilteredStrategy(Strategy):
    """Passes through signals from ``inner`` only when the HMM predicts ``active_regime``.

    In all other regimes, signal is set to 0 (flat).

    Parameters
    ----------
    inner:
        Any Strategy whose signals to filter.
    active_regime:
        Integer label of the regime in which trading is allowed.
    n_regimes:
        Number of HMM hidden states.
    """

    def __init__(
        self,
        inner: Strategy,
        active_regime: int = 0,
        n_regimes: int = 2,
    ) -> None:
        self._inner = inner
        self._active_regime = active_regime
        self._n_regimes = n_regimes

    @property
    def name(self) -> str:
        return f"regime_filtered_{self._inner.name}"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generate regime-filtered signals.

        1. Compute inner strategy signals on ``df``.
        2. Fit HMM on log-returns of ``df['close']``.
        3. Zero out signals where regime != ``active_regime``.

        Returns
        -------
        pd.Series of {-1, 0, 1} aligned to ``df.index``.
        """
        inner_signals = self._inner.generate_signals(df)

        returns = df["close"].pct_change().fillna(0.0)
        clf = RegimeClassifier(n_regimes=self._n_regimes)
        clf.fit(returns)
        regimes = clf.predict(returns)

        mask = pd.Series(regimes, index=df.index) == self._active_regime
        filtered = inner_signals.where(mask, other=0)
        return filtered.astype(int)
