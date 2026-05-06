"""Cox-Ross-Rubinstein (CRR) Binomial Tree Options Pricing.

Prices European and American options by backward induction on a recombining
binomial tree. Supports continuous dividend yield q.

Tree parameters (CRR, 1979):
    u = exp(sigma * sqrt(dt))        — up factor
    d = 1 / u                        — down factor (recombining)
    p = (exp((r-q)*dt) - d) / (u-d) — risk-neutral up probability

Backward induction:
    - At expiry: V[i] = max(0, phi * (S_T[i] - K))   (phi = +1 call, -1 put)
    - At each earlier node: V = exp(-r*dt) * (p*V_up + (1-p)*V_down)
    - American: V = max(V, intrinsic)   (early exercise check)
"""

from __future__ import annotations

import numpy as np

VALID_TYPES = {"call", "put"}


def binomial_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    n_steps: int = 200,
    american: bool = False,
) -> float:
    """Price a European or American option using the CRR binomial tree.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
        Standard option parameters. q = continuous dividend yield.
    option_type : str
        'call' or 'put'.
    n_steps : int
        Number of time steps. Higher = more accurate (default 200).
    american : bool
        If True, add early exercise check at each node (American option).

    Returns
    -------
    float
        Option price.
    """
    if option_type not in VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    phi = 1.0 if option_type == "call" else -1.0
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    discount = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    q_prob = 1.0 - p

    # Terminal asset prices: S * u^j * d^(n-j) for j = 0..n
    j = np.arange(n_steps + 1)
    ST = S * (u ** j) * (d ** (n_steps - j))

    # Terminal option values
    V = np.maximum(phi * (ST - K), 0.0)

    # Backward induction
    for _ in range(n_steps):
        V = discount * (p * V[1:] + q_prob * V[:-1])
        if american:
            # Asset prices at this node level
            level = n_steps - _ - 1
            j_curr = np.arange(level + 1)
            S_node = S * (u ** j_curr) * (d ** (level - j_curr))
            intrinsic = np.maximum(phi * (S_node - K), 0.0)
            V = np.maximum(V, intrinsic)

    return float(V[0])
