"""Gymnasium portfolio environment for RL portfolio optimisation."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """Single-period portfolio environment.

    Observation: flattened (lookback × n_assets) matrix of log-returns,
                 shape ``(lookback * n_assets,)``, clipped to [-5, 5].
    Action:      continuous weights per asset in [0, 1]; normalised to sum=1
                 inside ``step()``.
    Reward:      log portfolio return for the current step.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.DataFrame,
        initial_capital: float = 100_000.0,
        lookback: int = 20,
    ) -> None:
        super().__init__()
        if len(prices) <= lookback:
            raise ValueError("prices must have more rows than lookback")

        self._prices = prices.values.astype(np.float64)
        self._n_assets = self._prices.shape[1]
        self._lookback = lookback
        self._initial_capital = initial_capital

        obs_dim = lookback * self._n_assets
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self._n_assets,), dtype=np.float32
        )

        self._step_idx: int = 0

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        window = self._prices[self._step_idx : self._step_idx + self._lookback + 1]
        log_rets = np.log(window[1:] / window[:-1])  # (lookback, n_assets)
        obs = log_rets.flatten().astype(np.float32)
        return np.clip(obs, -5.0, 5.0)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._step_idx = 0
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        # Normalise weights to sum=1; fall back to equal weight if all zero.
        total = action.sum()
        weights = action / total if total > 1e-8 else np.ones(self._n_assets) / self._n_assets

        # Next price row index
        cur_idx = self._step_idx + self._lookback
        nxt_idx = cur_idx + 1

        if nxt_idx >= len(self._prices):
            return self._get_obs(), 0.0, True, False, {}

        cur_prices = self._prices[cur_idx]
        nxt_prices = self._prices[nxt_idx]
        asset_returns = nxt_prices / cur_prices  # gross return per asset
        portfolio_return = float(np.dot(weights, asset_returns))
        reward = float(np.log(max(portfolio_return, 1e-8)))

        self._step_idx += 1

        terminated = (self._step_idx + self._lookback + 1) >= len(self._prices)
        return self._get_obs(), reward, terminated, False, {}
