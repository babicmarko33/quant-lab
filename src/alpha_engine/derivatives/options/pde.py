"""Finite Difference PDE solvers for options pricing.

Phase 3.5 — European option: Crank-Nicolson (θ=0.5) scheme
Phase 3.6 — American option: PSOR (Projected Successive Over-Relaxation)

The Black-Scholes PDE in log-space (x = ln(S)):
    ∂V/∂τ = 0.5*σ²*∂²V/∂x² + (r-q-0.5*σ²)*∂V/∂x - r*V
where τ = T - t (time to expiry, backwards in time).

Grid:
    - x ∈ [x_min, x_max] with x_min = ln(S) - 4*σ*√T, x_max = ln(S) + 4*σ*√T
    - Uniform spacing: dx = (x_max - x_min) / (n_s - 1)
    - Time steps: dτ = T / n_t, marching forward in τ (backward in calendar time)

Boundary conditions (call):
    - At x_min: V = 0  (S → 0, call worthless)
    - At x_max: V = exp(x_max) * exp(-q*τ) - K * exp(-r*τ)  (deep ITM: delta ≈ 1)

Boundary conditions (put):
    - At x_min: V = K * exp(-r*τ) - exp(x_min) * exp(-q*τ)  (deep ITM: delta ≈ -1)
    - At x_max: V = 0  (S → ∞, put worthless)

PSOR for American:
    At each time step after solving the CN system, project each node:
        V[i] = max(V[i], intrinsic[i])
    This enforces the early exercise constraint.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded

VALID_TYPES = {"call", "put"}


def _build_cn_matrices(
    alpha_coeff: np.ndarray,
    beta_coeff: np.ndarray,
    n_inner: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Crank-Nicolson tri-diagonal LHS (A) and RHS (B) matrices.

    Returns (A_banded, B) where A_banded is in scipy banded format (3×n_inner).
    The θ=0.5 scheme: A * V^{n+1} = B * V^n + boundary terms.

    Parameters
    ----------
    alpha_coeff, beta_coeff : np.ndarray
        Per-node coefficients, shape (n_inner,).
    """
    # Diagonal entries
    main = 1.0 + alpha_coeff
    lower = -0.5 * beta_coeff[:-1]  # sub-diagonal
    upper = -0.5 * beta_coeff[1:]   # super-diagonal

    # Pack into scipy banded format: row0=upper-diag, row1=main-diag, row2=lower-diag
    A_banded = np.zeros((3, n_inner))
    A_banded[0, 1:] = upper       # upper diag (offset +1): starts at col 1
    A_banded[1, :] = main
    A_banded[2, :-1] = lower      # lower diag (offset -1): ends at col n-2

    # RHS matrix (dense, but we only need B*V which we compute directly)
    return A_banded, None  # RHS computed inline in the solver


def pde_european(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    n_s: int = 200,
    n_t: int = 200,
) -> float:
    """Price a European option via Crank-Nicolson finite differences.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
        Standard BSM parameters.
    option_type : str
        'call' or 'put'.
    n_s : int
        Number of spatial grid points (default 200).
    n_t : int
        Number of time steps (default 200).

    Returns
    -------
    float
        Option price at (S, 0).
    """
    if option_type not in VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return _pde_solve(S, K, T, r, sigma, q, option_type, n_s, n_t, american=False)


def pde_american(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    q: float = 0.0,
    n_s: int = 200,
    n_t: int = 200,
) -> float:
    """Price an American option via Crank-Nicolson with PSOR early exercise.

    At each time step the solution is projected onto the early-exercise
    constraint: V >= max(phi*(S-K), 0).

    Parameters
    ----------
    See pde_european for parameter descriptions.

    Returns
    -------
    float
        American option price at (S, 0).
    """
    if option_type not in VALID_TYPES:
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    return _pde_solve(S, K, T, r, sigma, q, option_type, n_s, n_t, american=True)


def _pde_solve(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    option_type: str,
    n_s: int,
    n_t: int,
    american: bool,
) -> float:
    """Core finite difference solver (European CN or American CN+projection)."""
    phi = 1.0 if option_type == "call" else -1.0

    # ---- Log-space grid ------------------------------------------------
    x0 = np.log(S)
    x_min = x0 - 4.0 * sigma * np.sqrt(T)
    x_max = x0 + 4.0 * sigma * np.sqrt(T)
    dx = (x_max - x_min) / (n_s - 1)
    x = np.linspace(x_min, x_max, n_s)
    dt = T / n_t

    # ---- Terminal condition (τ=0: V = payoff) ---------------------------
    S_grid = np.exp(x)
    V = np.maximum(phi * (S_grid - K), 0.0)

    # ---- PDE coefficients (constant across nodes for log-space) ---------
    nu = r - q - 0.5 * sigma**2          # drift in log-space
    sigma2 = sigma**2

    # Crank-Nicolson theta=0.5 coefficients for interior nodes
    # Second-order: σ²/(2*dx²); first-order: ν/(2*dx)
    a = 0.5 * dt * (sigma2 / dx**2 - nu / dx)   # lower coeff (implicit part ×2)
    b = 0.5 * dt * (sigma2 / dx**2 + r)          # diagonal coeff
    c = 0.5 * dt * (sigma2 / dx**2 + nu / dx)    # upper coeff

    # Interior node count
    n_inner = n_s - 2  # indices 1..n_s-2

    # LHS tri-diagonal (implicit part): A = I + θ·L
    # a_i * V_{i-1} is on sub-diag, b_i * V_i on main, c_i * V_{i+1} on super
    main_diag = np.full(n_inner, 1.0 + b)
    sub_diag = np.full(n_inner - 1, -0.5 * a)  # ×0.5 because θ=0.5 already in a,b,c
    sup_diag = np.full(n_inner - 1, -0.5 * c)

    # Banded format for solve_banded (ab[0]=upper, ab[1]=main, ab[2]=lower)
    ab = np.zeros((3, n_inner))
    ab[0, 1:] = sup_diag
    ab[1, :] = main_diag
    ab[2, :-1] = sub_diag

    # ---- Time march (τ: 0 → T) -----------------------------------------
    for n in range(n_t):
        tau = (n + 1) * dt  # current τ after this step

        # Boundary values
        if option_type == "call":
            V_lo = 0.0
            V_hi = np.exp(x_max - q * tau) - K * np.exp(-r * tau)
        else:
            V_lo = K * np.exp(-r * tau) - np.exp(x_min - q * tau)
            V_hi = 0.0
        V_lo = max(V_lo, 0.0)
        V_hi = max(V_hi, 0.0)

        V_inner = V[1:-1]

        # RHS: explicit part (1 - θ·L)*V^n + boundary contributions
        rhs = np.empty(n_inner)
        rhs[0] = (1.0 - b) * V_inner[0] + 0.5 * a * V_inner[1] + a * V_lo
        rhs[-1] = 0.5 * c * V_inner[-2] + (1.0 - b) * V_inner[-1] + c * V_hi
        if n_inner > 2:
            rhs[1:-1] = (
                0.5 * a * V_inner[:-2]
                + (1.0 - b) * V_inner[1:-1]
                + 0.5 * c * V_inner[2:]
            )

        # Boundary corrections to RHS for implicit boundary terms
        rhs[0] += 0.5 * a * V_lo
        rhs[-1] += 0.5 * c * V_hi

        # Solve tri-diagonal system
        V_new_inner = solve_banded((1, 1), ab, rhs)

        # American: project onto early-exercise constraint
        if american:
            intrinsic_inner = np.maximum(phi * (S_grid[1:-1] - K), 0.0)
            V_new_inner = np.maximum(V_new_inner, intrinsic_inner)

        V[0] = V_lo
        V[1:-1] = V_new_inner
        V[-1] = V_hi

    # ---- Interpolate at S ----------------------------------------------
    return float(np.interp(x0, x, V))
