"""Tests for gymnasium PortfolioEnv and RLPortfolioAgent."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_engine.rl.portfolio_env import PortfolioEnv
from alpha_engine.rl.rl_agent import RLPortfolioAgent

pytestmark = pytest.mark.slow

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_prices():
    """Synthetic 3-asset price DataFrame (120 daily bars)."""
    rng = np.random.default_rng(42)
    n = 120
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "SPY": 400.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)),
        "QQQ": 300.0 * np.cumprod(1 + rng.normal(0.0006, 0.012, n)),
        "TLT": 100.0 * np.cumprod(1 + rng.normal(0.0001, 0.005, n)),
    }, index=idx)


# ---------------------------------------------------------------------------
# PortfolioEnv
# ---------------------------------------------------------------------------

def test_env_instantiation(sample_prices):
    """PortfolioEnv can be instantiated with price data."""
    env = PortfolioEnv(prices=sample_prices)
    assert env is not None


def test_env_observation_space(sample_prices):
    """observation_space is a Box with correct shape."""
    import gymnasium as gym
    env = PortfolioEnv(prices=sample_prices)
    assert hasattr(env, "observation_space")
    assert isinstance(env.observation_space, gym.spaces.Box)


def test_env_action_space(sample_prices):
    """action_space is a Box with n_assets dimensions."""
    import gymnasium as gym
    env = PortfolioEnv(prices=sample_prices)
    n_assets = sample_prices.shape[1]
    assert isinstance(env.action_space, gym.spaces.Box)
    assert env.action_space.shape == (n_assets,)


def test_env_reset_returns_obs(sample_prices):
    """reset() returns (obs, info) tuple; obs matches observation_space."""
    env = PortfolioEnv(prices=sample_prices)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_env_step_returns_tuple(sample_prices):
    """step() returns (obs, reward, terminated, truncated, info)."""
    env = PortfolioEnv(prices=sample_prices)
    env.reset()
    action = env.action_space.sample()
    result = env.step(action)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_env_action_normalized(sample_prices):
    """Weights sum to 1 after action normalization inside env."""
    env = PortfolioEnv(prices=sample_prices)
    env.reset()
    # Use equal weight action — reward should be finite
    n_assets = sample_prices.shape[1]
    action = np.ones(n_assets) / n_assets
    _, reward, _, _, _ = env.step(action)
    assert np.isfinite(reward)


def test_env_episode_terminates(sample_prices):
    """Episode terminates after exhausting all price data."""
    env = PortfolioEnv(prices=sample_prices)
    env.reset()
    n_assets = sample_prices.shape[1]
    action = np.ones(n_assets) / n_assets
    terminated = False
    steps = 0
    while not terminated and steps < 500:
        _, _, terminated, truncated, _ = env.step(action)
        terminated = terminated or truncated
        steps += 1
    assert terminated


# ---------------------------------------------------------------------------
# RLPortfolioAgent
# ---------------------------------------------------------------------------

def test_rl_agent_instantiation(sample_prices):
    """RLPortfolioAgent can be constructed with an env."""
    env = PortfolioEnv(prices=sample_prices)
    agent = RLPortfolioAgent(env=env)
    assert agent is not None


def test_rl_agent_train_runs(sample_prices):
    """train() completes without error for a small step count."""
    env = PortfolioEnv(prices=sample_prices)
    agent = RLPortfolioAgent(env=env, total_timesteps=200)
    agent.train()  # should not raise


def test_rl_agent_predict_returns_array(sample_prices):
    """predict() returns a weights array after training."""
    env = PortfolioEnv(prices=sample_prices)
    agent = RLPortfolioAgent(env=env, total_timesteps=200)
    agent.train()
    obs, _ = env.reset()
    weights = agent.predict(obs)
    assert isinstance(weights, np.ndarray)
    assert weights.shape == (sample_prices.shape[1],)


def test_rl_agent_weights_sum_to_one(sample_prices):
    """Predicted weights sum to approximately 1.0."""
    env = PortfolioEnv(prices=sample_prices)
    agent = RLPortfolioAgent(env=env, total_timesteps=200)
    agent.train()
    obs, _ = env.reset()
    weights = agent.predict(obs)
    assert abs(weights.sum() - 1.0) < 0.01
