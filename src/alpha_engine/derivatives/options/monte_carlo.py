"""Monte Carlo Options Pricing.

Implements:
- European options via Geometric Brownian Motion (GBM) path simulation
- Asian options (arithmetic and geometric average)
- Barrier options (up-and-out, down-and-out, up-and-in, down-and-in)

Variance reduction:
- Antithetic variates for European pricing (halves simulation variance at no cost)

Path model (GBM with continuous dividend yield q):
    S(t+dt) = S(t) * exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0,1)
"""

from __future__ import annotations

import numpy as np

VALID_TYPES = {"call", "put"}
VALID_AVERAGING = {"arithmetic", "geometric"}
VALID_BARRIER_TYPES = {"up-and-out", "down-and-out", "up-and-in", "down-and-in"}

_N_STEPS_DEFAULT = 252  # daily steps


def _gbm_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
    antithetic: bool = False,
) -> np.ndarray:
    """Simulate GBM paths.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_steps + 1). paths[:, 0] = S.
    """
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        Z = rng.standard_normal((half, n_steps))
        Z = np.vstack([Z, -Z])
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    log_returns = drift + diffusion * Z
    log_paths = np.cumsum(log_returns, axis=1)
    return S * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))


def mc_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    n_paths: int = 100_000,
    n_steps: int = _N_STEPS_DEFAULT,
    antithetic: bool = True,
    seed: int | None = None,
) -> tuple[float, float]:
    """Price a European option via Monte Carlo simulation.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
        Standard BSM parameters.
    option_type : str
        'call' or 'put'.
    n_paths : int
        Number of simulated paths (default 100_000).
    n_steps : int
        Time steps per path (default 252 = daily).
    antithetic : bool
        Use antithetic variates for variance reduction (default True).
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    (price, stderr) : tuple[float, float]
        MC price and standard error of the mean.
    """
    if option_type not in VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    rng = np.random.default_rng(seed)
    paths = _gbm_paths(S, T, r, sigma, q, n_paths, n_steps, rng, antithetic=antithetic)
    terminal = paths[:, -1]

    if option_type == "call":
        payoffs = np.maximum(terminal - K, 0.0)
    else:
        payoffs = np.maximum(K - terminal, 0.0)

    discount = np.exp(-r * T)
    discounted = discount * payoffs
    price = float(discounted.mean())
    stderr = float(discounted.std() / np.sqrt(n_paths))
    return price, stderr


def mc_asian(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    averaging: str = "arithmetic",
    n_paths: int = 100_000,
    n_steps: int = _N_STEPS_DEFAULT,
    seed: int | None = None,
) -> tuple[float, float]:
    """Price an Asian option (fixed strike) via Monte Carlo.

    Parameters
    ----------
    averaging : str
        'arithmetic' or 'geometric' average of the path.

    Returns
    -------
    (price, stderr) : tuple[float, float]
    """
    if option_type not in VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
    if averaging not in VALID_AVERAGING:
        raise ValueError(f"averaging must be 'arithmetic' or 'geometric', got '{averaging}'")

    rng = np.random.default_rng(seed)
    paths = _gbm_paths(S, T, r, sigma, q, n_paths, n_steps, rng, antithetic=False)
    # Exclude time-0 (paths[:, 1:] are the daily prices)
    price_paths = paths[:, 1:]

    if averaging == "arithmetic":
        avg = price_paths.mean(axis=1)
    else:
        avg = np.exp(np.log(price_paths).mean(axis=1))

    if option_type == "call":
        payoffs = np.maximum(avg - K, 0.0)
    else:
        payoffs = np.maximum(K - avg, 0.0)

    discount = np.exp(-r * T)
    discounted = discount * payoffs
    price = float(discounted.mean())
    stderr = float(discounted.std() / np.sqrt(n_paths))
    return price, stderr


def mc_barrier(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    barrier_type: str = "up-and-out",
    option_type: str = "call",
    q: float = 0.0,
    n_paths: int = 100_000,
    n_steps: int = _N_STEPS_DEFAULT,
    seed: int | None = None,
) -> tuple[float, float]:
    """Price a barrier option via Monte Carlo.

    Parameters
    ----------
    barrier : float
        Barrier level.
    barrier_type : str
        One of: 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'.

    Returns
    -------
    (price, stderr) : tuple[float, float]
    """
    if option_type not in VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
    if barrier_type not in VALID_BARRIER_TYPES:
        raise ValueError(
            f"barrier_type must be one of {VALID_BARRIER_TYPES}, got '{barrier_type}'"
        )

    rng = np.random.default_rng(seed)
    paths = _gbm_paths(S, T, r, sigma, q, n_paths, n_steps, rng, antithetic=False)
    terminal = paths[:, -1]

    # Determine which paths crossed the barrier
    if barrier_type == "up-and-out":
        crossed = paths.max(axis=1) >= barrier
        alive = ~crossed
    elif barrier_type == "down-and-out":
        crossed = paths.min(axis=1) <= barrier
        alive = ~crossed
    elif barrier_type == "up-and-in":
        crossed = paths.max(axis=1) >= barrier
        alive = crossed
    else:  # down-and-in
        crossed = paths.min(axis=1) <= barrier
        alive = crossed

    if option_type == "call":
        payoffs = np.where(alive, np.maximum(terminal - K, 0.0), 0.0)
    else:
        payoffs = np.where(alive, np.maximum(K - terminal, 0.0), 0.0)

    discount = np.exp(-r * T)
    discounted = discount * payoffs
    price = float(discounted.mean())
    stderr = float(discounted.std() / np.sqrt(n_paths))
    return price, stderr
