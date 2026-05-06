"""RLStrategy — wraps RLPortfolioAgent as a Strategy-compatible class.

Trains a PPO agent on the provided OHLCV data and converts the predicted
portfolio weights into discrete {-1, 0, +1} position signals compatible
with the vectorized BacktestEngine.

The single-asset case maps the weight of the first (and only) asset:
  weight > threshold  → +1  (long)
  weight < (1-threshold) → -1 (short, if allow_short=True)
  otherwise           →  0  (flat)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_engine.rl.portfolio_env import PortfolioEnv
from alpha_engine.rl.rl_agent import RLPortfolioAgent
from alpha_engine.strategies.base import Strategy


class RLStrategy(Strategy):
    """Reinforcement-learning portfolio strategy backed by a PPO agent.

    Parameters
    ----------
    lookback:
        Rolling window length fed to the PortfolioEnv observation.
    total_timesteps:
        PPO training steps. 10 000 is sufficient for smoke-test; use
        200 000+ for production.
    long_threshold:
        Weight threshold above which the signal is +1.
    allow_short:
        If True, weight below ``1 - long_threshold`` emits -1 signal.
    """

    def __init__(
        self,
        lookback: int = 20,
        total_timesteps: int = 10_000,
        long_threshold: float = 0.6,
        allow_short: bool = False,
    ) -> None:
        self._lookback = lookback
        self._total_timesteps = total_timesteps
        self._long_threshold = long_threshold
        self._allow_short = allow_short

    @property
    def name(self) -> str:
        return "rl_portfolio"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Train PPO on ``df`` then replay to generate {-1, 0, +1} signals.

        The environment is re-trained fresh each call (no persistence).
        For production use, train once then cache the agent.
        """
        # Build a single-column prices frame for the environment
        prices = df[["close"]].copy()
        prices.columns = ["asset"]

        env = PortfolioEnv(prices=prices, lookback=self._lookback)
        agent = RLPortfolioAgent(env=env, total_timesteps=self._total_timesteps)
        agent.train()

        # Replay to collect signals
        signals = pd.Series(0.0, index=df.index)
        obs, _ = env.reset()
        done = False
        step = self._lookback  # first valid bar index

        while not done and step < len(df):
            weights = agent.predict(obs)
            w = float(weights[0])  # single asset weight

            if w >= self._long_threshold:
                sig = 1.0
            elif self._allow_short and w < (1.0 - self._long_threshold):
                sig = -1.0
            else:
                sig = 0.0

            signals.iloc[step] = sig
            obs, _, terminated, truncated, _ = env.step(np.array(weights, dtype=np.float32))
            done = terminated or truncated
            step += 1

        return signals
