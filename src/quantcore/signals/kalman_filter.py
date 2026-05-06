"""Kalman filter for dynamic hedge ratio estimation in pairs trading.

Implements a state-space model where the hedge ratio β_t evolves as a
random walk, estimated via the standard Kalman filter equations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_DELTA = 1e-4          # state transition variance (controls β drift speed)
_VE = 0.001            # observation noise variance initial estimate


class KalmanPairFilter:
    """Kalman filter for dynamic hedge ratio and spread estimation.

    Models y_t = β_t · x_t + α_t + ε_t where [β_t, α_t] follow a random walk.

    After fitting, access:
    - ``spread_`` — Kalman-filtered residual series (y - β̂x - α̂)
    - ``hedge_ratio_`` — time-varying β̂_t series
    - ``zscore_`` — spread normalised by Kalman-estimated std deviation
    """

    def __init__(self, delta: float = _DELTA) -> None:
        self._delta = delta
        self._fitted = False

    def fit(self, x: pd.Series, y: pd.Series) -> KalmanPairFilter:
        """Run the Kalman filter on the (x, y) pair.

        Parameters
        ----------
        x, y:
            Price series of equal length.

        Returns
        -------
        self
        """
        if len(x) != len(y):
            raise ValueError(
                f"x and y must have the same length, got {len(x)} and {len(y)}"
            )

        n = len(x)
        x_arr = x.values.astype(float)
        y_arr = y.values.astype(float)

        # State vector: [β, α]; observation matrix F_t = [x_t, 1]
        # State covariance P, process noise Q, observation noise var R

        Q = self._delta / (1.0 - self._delta) * np.eye(2)
        R = _VE

        beta = np.zeros(2)          # initial state [β, α]
        P = np.zeros((2, 2))        # initial state covariance

        hedge_ratios = np.empty(n)
        spread = np.empty(n)
        var_spread = np.empty(n)

        for t in range(n):
            F = np.array([x_arr[t], 1.0])  # observation vector

            # Predict
            P_pred = P + Q

            # Innovation
            y_hat = F @ beta
            e = y_arr[t] - y_hat
            S = float(F @ P_pred @ F) + R

            # Update
            K = P_pred @ F / S
            beta = beta + K * e
            P = (np.eye(2) - np.outer(K, F)) @ P_pred

            hedge_ratios[t] = beta[0]
            spread[t] = e
            var_spread[t] = S

        self._hedge_ratio = hedge_ratios
        self._spread = spread
        self._var_spread = var_spread
        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "KalmanPairFilter is not fitted. Call fit() first."
            )

    @property
    def spread_(self) -> np.ndarray:
        """Raw Kalman innovation (residual) series."""
        self._check_fitted()
        return self._spread

    @property
    def hedge_ratio_(self) -> np.ndarray:
        """Time-varying hedge ratio β̂_t series."""
        self._check_fitted()
        return self._hedge_ratio

    @property
    def zscore_(self) -> np.ndarray:
        """Spread normalised by Kalman-estimated standard deviation."""
        self._check_fitted()
        std = np.sqrt(np.maximum(self._var_spread, 1e-12))
        return self._spread / std
