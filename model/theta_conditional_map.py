"""Conditional MAP estimate for theta given data, z_map, and consensus rho_c.

These helpers compute the mode of

    p(theta_c | data, z_map, rho_c)  ~  Mallows-likelihood × Gamma(a_theta, b_theta) prior

for a single cluster, given a fixed consensus weak order (block partition).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Consensus helpers
# ---------------------------------------------------------------------------

def block_of(blocks: Sequence[Sequence[int]], n_items: int) -> np.ndarray:
    """Map each item to its consensus block index (0 = top block)."""
    b = np.full(n_items, -1, dtype=np.intp)
    for idx, blk in enumerate(blocks):
        for item in blk:
            b[item] = idx
    if np.any(b < 0):
        missing = np.where(b < 0)[0].tolist()
        raise ValueError(f"items {missing} are not assigned to any block")
    return b


def block_sizes(blocks: Sequence[Sequence[int]]) -> np.ndarray:
    return np.array([len(blk) for blk in blocks], dtype=np.intp)


# ---------------------------------------------------------------------------
# Observed total distance  D_c
# ---------------------------------------------------------------------------

def total_between_block_distance(rank: np.ndarray, b_of: np.ndarray) -> float:
    """Sum of between-block inversions over a set of (strict) assessors.

    Parameters
    ----------
    rank : (N_c, n_items) int array
        rank[r, item] = position of `item` for assessor r (0 = top).
    b_of : (n_items,) int array
        Consensus block index per item (0 = top block).

    Returns
    -------
    float
        D_c = sum_{r in cluster} d(r, rho_c), the between-block inversion count.
    """
    N_c, n = rank.shape
    D = 0.0
    for i in range(n):
        bi = b_of[i]
        ri = rank[:, i]
        for j in range(i + 1, n):
            bj = b_of[j]
            if bi == bj:
                continue  # within consensus block: no contribution
            rj = rank[:, j]
            if bi < bj:
                # consensus: i above j  ->  disagreement when j is ranked above i
                D += np.count_nonzero(rj < ri)
            else:
                # consensus: j above i  ->  disagreement when i is ranked above j
                D += np.count_nonzero(ri < rj)
    return float(D)


def ranks_from_orderings(R_cluster: np.ndarray, n_items: int) -> np.ndarray:
    """Convert position->item orderings to item->position ranks.

    R_cluster[r, pos] = item  ==>  rank[r, item] = pos.
    """
    R = np.asarray(R_cluster, dtype=np.intp)
    N_c = R.shape[0]
    rank = np.empty((N_c, n_items), dtype=np.intp)
    rows = np.arange(N_c)[:, None]
    rank[rows, R] = np.arange(n_items)[None, :]
    return rank


# ---------------------------------------------------------------------------
# Model moments  E_theta[d],  Var_theta[d]
# ---------------------------------------------------------------------------

def moments(theta: float, sizes: Sequence[int], n_items: int) -> Tuple[float, float]:
    """Return (E_theta[d], Var_theta[d]) for the given consensus block sizes.

    Cancellation-aware multiplicity form::

        E   = sum_{j=1}^n (m_j - 1) * t_j
        Var = sum_{j=1}^n (m_j - 1) * v_j

    with m_j = #{blocks of size >= j}, so the (m_j - 1) coefficients sum to
    zero and the 1/theta divergences of t_j, v_j as theta -> 0 cancel.
    """
    s = np.asarray(sizes, dtype=np.intp)
    n = int(n_items)
    j = np.arange(1, n + 1, dtype=np.float64)
    m = (s[:, None] >= j[None, :]).sum(axis=0).astype(np.float64)  # (n,)
    coeff = m - 1.0
    phij = np.exp(-theta * j)            # phi^j = exp(-theta * j)
    denom = 1.0 - phij                   # safe for theta > ~1e-15
    t = j * phij / denom
    v = (j * j) * phij / (denom * denom)
    return float(np.sum(coeff * t)), float(np.sum(coeff * v))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_theta_conditional_map(
    D_c: float,
    N_c: int,
    sizes: Sequence[int],
    n_items: int,
    a_theta: float,
    b_theta: float,
    theta_init: float = 1.0,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """Mode of p(theta_c | data, z_map, rho_c) under a Gamma(a_theta, b_theta) prior.

    Assumes a_theta >= 1 (log-concave conditional, unique mode). For a_theta < 1
    a root is still returned via the bracket, but uniqueness is not guaranteed.
    """
    s = np.asarray(sizes, dtype=np.intp)
    n = int(n_items)

    # All-tied consensus: E[d] == 0 and D_c == 0; posterior reduces to the prior.
    if len(s) <= 1:
        if b_theta > 0:
            return max((a_theta - 1.0) / b_theta, 1e-6)
        return float(theta_init)  # flat/improper prior: mode undefined

    def F(theta: float) -> float:        # score  d/dtheta log p
        E, _ = moments(theta, s, n)
        return -D_c + N_c * E + (a_theta - 1.0) / theta - b_theta

    def Fp(theta: float) -> float:       # d/dtheta of the score  (<= 0 for a>=1)
        _, Var = moments(theta, s, n)
        return -N_c * Var - (a_theta - 1.0) / (theta * theta)

    # --- bracket [lo, hi] with F(lo) > 0 > F(hi) ---
    lo = 1e-6
    hi = max(float(theta_init), 1.0)
    f_hi = F(hi)
    expand = 0
    while f_hi > 0 and expand < 80:
        hi *= 2.0
        f_hi = F(hi)
        expand += 1
    if F(lo) <= 0:
        # score already non-positive at the boundary -> prior-dominated, tiny theta
        return lo

    # --- safeguarded Newton ---
    theta = min(max(float(theta_init), lo), hi)
    for _ in range(max_iter):
        f = F(theta)
        if abs(f) < tol:
            break
        if f > 0:
            lo = theta
        else:
            hi = theta
        fp = Fp(theta)
        took_newton = False
        if fp < 0:
            cand = theta - f / fp
            if lo < cand < hi:
                theta = cand
                took_newton = True
        if not took_newton:
            theta = 0.5 * (lo + hi)
        if hi - lo < tol:
            break
    return float(theta)


def theta_conditional_map_for_cluster(
    R_cluster: np.ndarray,
    blocks: Sequence[Sequence[int]],
    n_items: int,
    a_theta: float,
    b_theta: float,
    theta_init: float = 1.0,
) -> Tuple[float, float]:
    """Conditional MAP of theta for one cluster.

    Parameters
    ----------
    R_cluster : (N_c, n_items) array
        Position-to-item orderings for this cluster's assessors.
    blocks : list of lists
        Consensus weak order (top block first).
    n_items : int
    a_theta, b_theta : float
        Gamma prior parameters.
    theta_init : float
        Starting point for the Newton solver.

    Returns
    -------
    (theta_hat, D_c) : float, float
        Conditional MAP estimate and total between-block distance.
    """
    n = int(n_items)
    rank = ranks_from_orderings(R_cluster, n)
    b_of = block_of(blocks, n)
    D_c = total_between_block_distance(rank, b_of)
    sizes = block_sizes(blocks)
    N_c = int(R_cluster.shape[0])
    theta_hat = solve_theta_conditional_map(
        D_c, N_c, sizes, n, a_theta=a_theta, b_theta=b_theta, theta_init=theta_init
    )
    return theta_hat, D_c
