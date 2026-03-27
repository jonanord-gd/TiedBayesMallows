"""Block helpers and fast Gibbs item-reassignment move.

Fast Gibbs complexity (per call to ``fast_gibbs_reassign_one_item``)
--------------------------------------------------------------------
* U_all precomputation (amortised, done once): O(N n²)
* H from U_all (once per cluster): O(N_c · n_pairs) via vectorised sum
* Per call: O(n) H_gt groups + O(K) prefix-sum; then O(1) per candidate.

vs old ``gibbs_reassign_one_item`` (in legacy/old_moves.py): O(2K · N n log K).
"""

import math
import random
from math import comb, lgamma, log
from typing import List, Optional, Tuple

import numpy as np

from .blocks import blocks_to_block_index
from .priors import build_log_qfactorials
from .utils import sample_categorical_from_logweights


# ── basic block helpers ───────────────────────────────────────

def remove_item_from_blocks(blocks: List[List[int]], x: int) -> Tuple[List[List[int]], int]:
    new_blocks = [b[:] for b in blocks]
    old_block = None
    for k, b in enumerate(new_blocks):
        if x in b:
            b.remove(x)
            old_block = k
            break
    if old_block is None:
        raise ValueError("Item not found in blocks.")
    new_blocks = [b for b in new_blocks if b]
    return new_blocks, old_block


def apply_move_existing_block(blocks_minus: List[List[int]], x: int, k: int) -> List[List[int]]:
    nb = [b[:] for b in blocks_minus]
    nb[k].append(x)
    return nb


def apply_move_new_block(blocks_minus: List[List[int]], x: int, pos: int) -> List[List[int]]:
    K = len(blocks_minus)
    if not (0 <= pos <= K):
        raise ValueError("Invalid insertion position.")
    nb = [b[:] for b in blocks_minus]
    nb.insert(pos, [x])
    return nb


# ── pairwise-preference precomputation ───────────────────────

def _pair_index(a: int, b: int, n: int) -> int:
    """Flat index into upper-triangular pair vector for (a, b) with a < b."""
    return a * (2 * n - a - 1) // 2 + (b - a - 1)


def compute_U_all(
    rankings: List[List[int]], n: int
) -> np.ndarray:
    """Build per-assessor pairwise preference matrix.  Computed **once**.

    Rankings are in position→item format (strict_r[pos] = item).
    We invert each ranking to get item→position, then compare positions.

    Returns U_all of shape ``(N, comb(n, 2))`` where
    ``U_all[i, pair_index(a, b)] = 1`` iff item *a* is ranked **after**
    item *b* by assessor *i* (i.e. position_of(a) > position_of(b)).
    """
    N = len(rankings)
    n_pairs = comb(n, 2)
    U = np.zeros((N, n_pairs), dtype=np.float64)
    for i in range(N):
        r = rankings[i]
        inv_r = [0] * n
        for pos in range(n):
            inv_r[r[pos]] = pos
        idx = 0
        for a in range(n):
            for b in range(a + 1, n):
                if inv_r[a] > inv_r[b]:
                    U[i, idx] = 1.0
                idx += 1
    return U


def build_cluster_pair_masks(
    block_idx_list: List[List[int]], n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Build pair-mask matrix M and discordant-count offsets for all clusters.

    Returns
    -------
    M : ndarray, shape (C, n_pairs)
        ``+1`` concordant, ``-1`` discordant, ``0`` same-block.
    offsets : ndarray, shape (C,)
    """
    C = len(block_idx_list)
    n_pairs = comb(n, 2)
    a_idx, b_idx = np.triu_indices(n, k=1)
    M = np.zeros((C, n_pairs), dtype=np.float64)
    offsets = np.zeros(C, dtype=np.float64)
    for c in range(C):
        bidx = np.asarray(block_idx_list[c], dtype=np.intp)
        diff = bidx[a_idx] - bidx[b_idx]
        M[c] = np.sign(-diff)
        offsets[c] = np.count_nonzero(diff > 0)
    return M, offsets


def update_pair_mask_for_item(
    M_row: np.ndarray, offset: float,
    block_idx: List[int], item: int, old_block: int, new_block: int, n: int,
) -> Tuple[np.ndarray, float]:
    """Incrementally update one row of M when *item* moves blocks.  O(n)."""
    for x in range(n):
        if x == item:
            continue
        a, b = min(item, x), max(item, x)
        pidx = _pair_index(a, b, n)
        bx = block_idx[x]
        if item < x:
            old_sign = -1.0 if old_block > bx else (1.0 if old_block < bx else 0.0)
            new_sign = -1.0 if new_block > bx else (1.0 if new_block < bx else 0.0)
        else:
            old_sign = -1.0 if bx > old_block else (1.0 if bx < old_block else 0.0)
            new_sign = -1.0 if bx > new_block else (1.0 if bx < new_block else 0.0)
        if old_sign != new_sign:
            if old_sign == -1.0:
                offset -= 1
            if new_sign == -1.0:
                offset += 1
            M_row[pidx] = new_sign
    return M_row, offset


def compute_all_disagreements_fast(
    U_all: np.ndarray,
    M: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Compute disagreements[i][c] for all assessors and clusters via matmul.

    Returns ndarray of shape (N, C).
    """
    return U_all @ M.T + offsets[np.newaxis, :]


# ── Gibbs candidate building ──────────────────────────────────

def _build_base(
    blocks: List[List[int]], block_l: int, elem_idx: int
) -> List[List[int]]:
    """Build the base block structure with item l removed (drop empty blocks)."""
    base: List[List[int]] = []
    for i, blk in enumerate(blocks):
        if i == block_l:
            remaining = blk[:elem_idx] + blk[elem_idx + 1:]
            if remaining:
                base.append(remaining)
        else:
            base.append(list(blk))
    return base


def _build_candidate(
    base: List[List[int]], l: int, idx: int, K_minus: int
) -> List[List[int]]:
    """Construct the single candidate at index *idx* from the base.

    Indices 0..K_minus  → "create" (insert singleton [l] at position idx).
    Indices K_minus+1.. → "add" (append l to block idx - K_minus - 1).
    """
    if idx <= K_minus:
        return [list(b) for b in base[:idx]] + [[l]] + [list(b) for b in base[idx:]]
    else:
        k = idx - (K_minus + 1)
        cand = [list(b) for b in base]
        cand[k] = cand[k] + [l]
        return cand


# ── main fast Gibbs entry point ───────────────────────────────

def fast_gibbs_reassign_one_item(
    rankings: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    H: Optional[np.ndarray] = None,
    U_all: Optional[np.ndarray] = None,
    cluster_mask: Optional[np.ndarray] = None,
    include_uniform_order_prior: bool = True,
    rng: Optional[random.Random] = None,
    tie_penalty: float = 0.5,
) -> Tuple[List[List[int]], int, int]:
    """Gibbs-sample a new block assignment for one randomly chosen item.

    Return signature ``(new_blocks, n_proposals, n_accepts)`` matches the old
    ``gibbs_reassign_one_item`` but runs in O(n + K) per call vs O(2K · N n log K).

    Parameters
    ----------
    H : ndarray, optional
        Precomputed cluster-level pairwise preferences, shape ``(comb(n, 2),)``.
        ``H[pair_index(a, b)] = #{assessors in cluster: r[a] > r[b]}``.
        If *None*, derived from *U_all* + *cluster_mask* or computed from scratch.
    """
    if rng is None:
        rng = random.Random()
    if not rankings:
        return [b[:] for b in blocks], 0, 0

    N = len(rankings)
    n = len(rankings[0])

    if H is None:
        if U_all is not None and cluster_mask is not None:
            H = U_all[cluster_mask].sum(axis=0)
        else:
            U = compute_U_all(rankings, n)
            H = U.sum(axis=0)

    l = rng.randrange(n)
    block_index = blocks_to_block_index(blocks, n)
    block_l = block_index[l]
    elem_idx = blocks[block_l].index(l)

    base = _build_base(blocks, block_l, elem_idx)
    K_minus = len(base)

    base_sizes = [len(b) for b in base]
    base_block_idx = np.empty(n, dtype=np.intp)
    base_block_idx[l] = -1
    for k, blk in enumerate(base):
        for item in blk:
            base_block_idx[item] = k

    items = np.arange(n)
    xs = items[items != l]
    a_arr = np.minimum(l, xs)
    b_arr = np.maximum(l, xs)
    pidx_arr = a_arr * (2 * n - a_arr - 1) // 2 + (b_arr - a_arr - 1)

    h_vals = H[pidx_arr].copy()
    flip_mask = l > xs
    h_vals[flip_mask] = N - h_vals[flip_mask]

    block_ids = base_block_idx[xs]
    H_gt_block = np.zeros(K_minus)
    np.add.at(H_gt_block, block_ids, h_vals)
    H_lt_block = N * np.array(base_sizes, dtype=float) - H_gt_block

    prefix_Hlt = np.empty(K_minus + 1)
    prefix_Hlt[0] = 0.0
    np.cumsum(H_lt_block, out=prefix_Hlt[1:])
    suffix_Hgt = np.empty(K_minus + 1)
    suffix_Hgt[K_minus] = 0.0
    np.cumsum(H_gt_block[::-1], out=suffix_Hgt[:K_minus])
    suffix_Hgt[:K_minus] = suffix_Hgt[:K_minus][::-1]

    q = math.exp(-theta)
    log_qfact = build_log_qfactorials(n, q)
    log_gamma_1md = lgamma(1.0 - delta)
    log_py_denom = lgamma(gamma + n) - lgamma(gamma + 1)

    log_py_tables_base = 0.0
    for i in range(1, K_minus):
        log_py_tables_base += log(gamma + i * delta)

    log_py_sizes_base = 0.0
    for s in base_sizes:
        log_py_sizes_base += lgamma(s - delta) - log_gamma_1md

    base_Tm = sum(s * (s - 1) // 2 for s in base_sizes)
    base_logP = sum(lgamma(s + 1) for s in base_sizes)
    base_log_qf_sum = sum(log_qfact[s] for s in base_sizes)

    log_py_create = (log_py_tables_base + log(gamma + K_minus * delta)
                     + log_py_sizes_base - log_py_denom)
    log_Z_create = (-theta * tie_penalty * base_Tm + base_logP
                    + log_qfact[n] - base_log_qf_sum - log_qfact[1])
    log_ord_create = -lgamma(K_minus + 2) if include_uniform_order_prior else 0.0

    log_py_add_common = log_py_tables_base + log_py_sizes_base - log_py_denom
    log_Z_add_base = (-theta * tie_penalty * base_Tm + base_logP
                      + log_qfact[n] - base_log_qf_sum)
    log_ord_add = -lgamma(K_minus + 1) if include_uniform_order_prior else 0.0

    log_weights: List[float] = []

    lw_create_common = log_py_create + log_ord_create - N * log_Z_create
    for p in range(K_minus + 1):
        contrib_l = prefix_Hlt[p] + suffix_Hgt[p]
        log_weights.append(lw_create_common - theta * contrib_l)

    for k in range(K_minus):
        s_k = base_sizes[k]
        contrib_l = prefix_Hlt[k] + suffix_Hgt[k + 1] + tie_penalty * N * s_k
        log_py_k = log_py_add_common + log(s_k - delta)
        log_Z_k = (log_Z_add_base - theta * tie_penalty * s_k
                   + log(s_k + 1) - (log_qfact[s_k + 1] - log_qfact[s_k]))
        log_weights.append(log_py_k + log_ord_add - theta * contrib_l - N * log_Z_k)

    idx = sample_categorical_from_logweights(log_weights, rng)
    chosen = _build_candidate(base, l, idx, K_minus)
    return chosen, 1, 1


# ── greedy (ICM) variant ─────────────────────────────────────

def _compute_item_logweights(
    blocks: List[List[int]],
    l: int,
    theta: float,
    gamma: float,
    delta: float,
    H: np.ndarray,
    N: int,
    n: int,
    tie_penalty: float = 0.5,
    include_uniform_order_prior: bool = True,
) -> Tuple[List[float], List[List[int]], int]:
    """Compute log-weights for all candidate placements of item *l*."""
    block_index = blocks_to_block_index(blocks, n)
    block_l = block_index[l]
    elem_idx = blocks[block_l].index(l)

    base = _build_base(blocks, block_l, elem_idx)
    K_minus = len(base)

    base_sizes = [len(b) for b in base]
    base_block_idx = np.empty(n, dtype=np.intp)
    base_block_idx[l] = -1
    for k, blk in enumerate(base):
        for item in blk:
            base_block_idx[item] = k

    items = np.arange(n)
    xs = items[items != l]
    a_arr = np.minimum(l, xs)
    b_arr = np.maximum(l, xs)
    pidx_arr = a_arr * (2 * n - a_arr - 1) // 2 + (b_arr - a_arr - 1)
    h_vals = H[pidx_arr].copy()
    h_vals[l > xs] = N - h_vals[l > xs]

    H_gt_block = np.zeros(K_minus)
    np.add.at(H_gt_block, base_block_idx[xs], h_vals)
    H_lt_block = N * np.array(base_sizes, dtype=float) - H_gt_block

    prefix_Hlt = np.empty(K_minus + 1)
    prefix_Hlt[0] = 0.0
    np.cumsum(H_lt_block, out=prefix_Hlt[1:])
    suffix_Hgt = np.empty(K_minus + 1)
    suffix_Hgt[K_minus] = 0.0
    np.cumsum(H_gt_block[::-1], out=suffix_Hgt[:K_minus])
    suffix_Hgt[:K_minus] = suffix_Hgt[:K_minus][::-1]

    q = math.exp(-theta)
    log_qfact = build_log_qfactorials(n, q)
    log_gamma_1md = lgamma(1.0 - delta)
    log_py_denom = lgamma(gamma + n) - lgamma(gamma + 1)

    log_py_tables_base = sum(log(gamma + i * delta) for i in range(1, K_minus))
    log_py_sizes_base = sum(lgamma(s - delta) - log_gamma_1md for s in base_sizes)

    base_Tm = sum(s * (s - 1) // 2 for s in base_sizes)
    base_logP = sum(lgamma(s + 1) for s in base_sizes)
    base_log_qf_sum = sum(log_qfact[s] for s in base_sizes)

    log_py_create = (log_py_tables_base + log(gamma + K_minus * delta)
                     + log_py_sizes_base - log_py_denom)
    log_Z_create = (-theta * tie_penalty * base_Tm + base_logP
                    + log_qfact[n] - base_log_qf_sum - log_qfact[1])
    log_ord_create = -lgamma(K_minus + 2) if include_uniform_order_prior else 0.0

    log_py_add_common = log_py_tables_base + log_py_sizes_base - log_py_denom
    log_Z_add_base = (-theta * tie_penalty * base_Tm + base_logP
                      + log_qfact[n] - base_log_qf_sum)
    log_ord_add = -lgamma(K_minus + 1) if include_uniform_order_prior else 0.0

    log_weights: List[float] = []
    lw_create_common = log_py_create + log_ord_create - N * log_Z_create
    for p in range(K_minus + 1):
        log_weights.append(lw_create_common - theta * (prefix_Hlt[p] + suffix_Hgt[p]))
    for k in range(K_minus):
        s_k = base_sizes[k]
        contrib_l = prefix_Hlt[k] + suffix_Hgt[k + 1] + tie_penalty * N * s_k
        log_py_k = log_py_add_common + log(s_k - delta)
        log_Z_k = (log_Z_add_base - theta * tie_penalty * s_k
                   + log(s_k + 1) - (log_qfact[s_k + 1] - log_qfact[s_k]))
        log_weights.append(log_py_k + log_ord_add - theta * contrib_l - N * log_Z_k)

    return log_weights, base, K_minus


def greedy_reassign_one_item(
    blocks: List[List[int]],
    l: int,
    theta: float,
    gamma: float,
    delta: float,
    H: np.ndarray,
    N: int,
    n: int,
    tie_penalty: float = 0.5,
) -> Tuple[List[List[int]], bool]:
    """Deterministic ICM move: place item *l* at its MAP position."""
    log_weights, base, K_minus = _compute_item_logweights(
        blocks, l, theta, gamma, delta, H, N, n, tie_penalty)
    idx = int(np.argmax(log_weights))
    new_blocks = _build_candidate(base, l, idx, K_minus)
    changed = (tuple(tuple(sorted(b)) for b in new_blocks)
               != tuple(tuple(sorted(b)) for b in blocks))
    return new_blocks, changed


def icm_sweep_cluster(
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    H: np.ndarray,
    N: int,
    n: int,
    tie_penalty: float = 0.5,
    max_sweeps: int = 50,
) -> Tuple[List[List[int]], int, int]:
    """Run ICM sweeps over all items until convergence.

    Returns ``(blocks, total_moves, sweeps_done)``.
    """
    total_moves = 0
    for sweep in range(max_sweeps):
        sweep_moves = 0
        for l in range(n):
            blocks, changed = greedy_reassign_one_item(
                blocks, l, theta, gamma, delta, H, N, n, tie_penalty)
            if changed:
                sweep_moves += 1
        total_moves += sweep_moves
        if sweep_moves == 0:
            return blocks, total_moves, sweep + 1
    return blocks, total_moves, max_sweeps
