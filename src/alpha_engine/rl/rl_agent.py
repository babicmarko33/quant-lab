"""Lightweight PPO wrapper for portfolio weight prediction."""
from __future__ import annotations

import numpy as np
from stable_baselines3 import PPO

from alpha_engine.rl.portfolio_env import PortfolioEnv


class RLPortfolioAgent:
    """Train a PPO agent on a :class:`PortfolioEnv` and predict weights.

    Parameters
    ----------
    env:
        A ``PortfolioEnv`` instance (or any compatible gymnasium env).
    total_timesteps:
        Number of environment steps to train for.
    """

    def __init__(
        self,
        env: PortfolioEnv,
        total_timesteps: int = 10_000,
    ) -> None:
        self._env = env
        self._total_timesteps = total_timesteps
        self._model: PPO | None = None
        self._n_assets: int = env.action_space.shape[0]

    # ------------------------------------------------------------------
    def train(self) -> RLPortfolioAgent:
        """Train the PPO model. Returns self for chaining."""
        self._model = PPO("MlpPolicy", self._env, verbose=0)
        self._model.learn(total_timesteps=self._total_timesteps)
        return self

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return normalised portfolio weights for the given observation."""
        if self._model is None:
            raise RuntimeError("Call train() before predict()")
        action, _ = self._model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float64)
        total = action.sum()
        if total > 1e-8:
            return action / total
        return np.ones(self._n_assets) / self._n_assets

    def save(self, path: str) -> None:
        """Save the trained model to *path*."""
        if self._model is None:
            raise RuntimeError("Call train() before save()")
        self._model.save(path)

    def load(self, path: str) -> RLPortfolioAgent:
        """Load a previously saved model from *path*."""
        self._model = PPO.load(path, env=self._env)
        return self
