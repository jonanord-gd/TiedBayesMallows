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

try:
    from scipy.special import gammaln as _vec_lgamma
except ImportError:
    _vec_lgamma = np.vectorize(math.lgamma)

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

    Returns U_all of shape ``(N, comb(n, 2))`` with dtype **int8** where
    ``U_all[i, pair_index(a, b)] = 1`` iff item *a* is ranked **after**
    item *b* by assessor *i* (i.e. position_of(a) > position_of(b)).

    Values are binary (0/1) so int8 is lossless and uses 8× less memory
    than float64.  At N=2500, n=1200 this reduces U_all from ~14 GB to
    ~1.8 GB.
    """
    a_idx, b_idx = np.triu_indices(n, k=1)
    # inv_r[i, item] = position of that item for assessor i
    R = np.array(rankings, dtype=np.intp)          # (N, n)  position→item
    inv_r = np.argsort(R, axis=1).astype(np.intp)  # (N, n)  item→position
    U = (inv_r[:, a_idx] > inv_r[:, b_idx]).view(np.uint8).astype(np.int8)
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

    U_all is expected to be float32 (as stored by MixtureRankingModel._U_all).
    M may be float64; it will be cast to float32 for an efficient sgemm.

    Returns ndarray of shape (N, C), float64.
    """
    M_f32 = M.T.astype(np.float32) if M.dtype != np.float32 else M.T
    return (U_all @ M_f32).astype(np.float64) + offsets[np.newaxis, :]


# ── Gibbs candidate building ──────────────────────────────────

def build_pair_cache(n: int) -> np.ndarray:
    """Precompute flat pair indices for each item l.

    Returns ndarray of shape ``(n, n-1)`` where ``cache[l]`` holds the
    pair indices for all other items in the order ``[0..l-1, l+1..n-1]``.
    """
    items = np.arange(n)
    cache = np.empty((n, n - 1), dtype=np.intp)
    for l in range(n):
        xs = np.concatenate([items[:l], items[l + 1:]])
        a = np.minimum(l, xs)
        b = np.maximum(l, xs)
        cache[l] = a * (2 * n - a - 1) // 2 + (b - a - 1)
    return cache


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
    log_qfact: Optional[List[float]] = None,
    block_index: Optional[List[int]] = None,
    pair_cache: Optional[np.ndarray] = None,
    use_py_prior: bool = True,
) -> Tuple[List[List[int]], int, int, int]:
    """Gibbs-sample a new block assignment for one randomly chosen item.

    Return signature ``(new_blocks, n_proposals, n_accepts, moved_item)``.
    Runs in O(n + K) per call vs O(2K · N n log K) for the old version.

    Parameters
    ----------
    H : ndarray, optional
        Precomputed cluster-level pairwise preferences, shape ``(comb(n, 2),)``.
        ``H[pair_index(a, b)] = #{assessors in cluster: r[a] > r[b]}``.
        If *None*, derived from *U_all* + *cluster_mask* or computed from scratch.
    log_qfact : list of float, optional
        Precomputed ``build_log_qfactorials(n, exp(-theta))``.
        Pass this when calling in a loop with fixed theta to avoid recomputing.
    block_index : list of int, optional
        Item → block mapping for *blocks*.  Avoids redundant
        ``blocks_to_block_index`` calls when the caller already has it.
    pair_cache : ndarray, shape (n, n-1), optional
        Precomputed flat pair indices for each item (from ``build_pair_cache``).
    """
    if rng is None:
        rng = random.Random()
    if not rankings:
        return [b[:] for b in blocks], 0, 0, -1

    N = len(rankings)
    n = len(rankings[0])

    # ── Step 1: resolve H (cluster-level pairwise preference counts) ──────────
    # H[pair_index(a, b)] = number of assessors in this cluster for whom item a
    # is ranked *after* item b (i.e. position_of(a) > position_of(b)).
    # Three resolution paths, from cheapest to most expensive:
    #   (a) caller supplied H directly → use as-is
    #   (b) U_all (full N×pairs matrix) + boolean cluster_mask → row-sum subset
    #   (c) neither available → recompute U_all from scratch and sum all rows
    if H is None:
        if U_all is not None and cluster_mask is not None:
            H = U_all[cluster_mask].sum(axis=0)
        else:
            U = compute_U_all(rankings, n)
            H = U.sum(axis=0)

    # ── Step 2: pick the item to reassign ─────────────────────────────────────
    l = rng.randrange(n)                      # item l is selected uniformly at random
    if block_index is None:
        block_index = blocks_to_block_index(blocks, n, validate=False)
    block_l = block_index[l]                  # which block l currently sits in
    elem_idx = blocks[block_l].index(l)       # position of l within that block

    # ── Step 3: build the "base" structure (blocks minus item l) ──────────────
    # base is the block partition after removing l (empty blocks are dropped).
    # K_minus is the number of blocks that remain.
    base = _build_base(blocks, block_l, elem_idx)
    K_minus = len(base)

    # Block sizes and a per-item block-index array for the base structure.
    # base_block_idx[l] is set to -1 as a sentinel (l is unplaced).
    base_sizes = [len(b) for b in base]
    K = len(blocks)
    # Fast path: derive base_block_idx from the passed-in block_index via numpy,
    # avoiding the O(n) Python double for-loop.  block_index maps every item to
    # its current block before item l was removed.
    if block_index is not None:
        block_idx_arr = np.asarray(block_index, dtype=np.intp)
        base_block_idx = block_idx_arr.copy()
        base_block_idx[l] = -1
        if K_minus < K:  # block_l was a singleton and was dropped
            base_block_idx[base_block_idx > block_l] -= 1
    else:
        base_block_idx = np.empty(n, dtype=np.intp)
        base_block_idx[l] = -1
        for k, blk in enumerate(base):
            for item in blk:
                base_block_idx[item] = k

    # ── Step 4: compute per-pair preference counts involving l ────────────────
    # For every other item x we need h_l_gt_x = #{assessors: pos(l) > pos(x)},
    # i.e. the number of times l is ranked *after* x.
    # H stores counts for pairs (a, b) with a < b as "#{pos(a) > pos(b)}".
    # When l < x the stored value is already h_l_gt_x.
    # When l > x the stored value is h_x_gt_l = N - h_l_gt_x, so we flip.
    if pair_cache is not None:
        pidx_arr = pair_cache[l]
    else:
        items = np.arange(n)
        xs = items[items != l]
        a_arr = np.minimum(l, xs)
        b_arr = np.maximum(l, xs)
        pidx_arr = a_arr * (2 * n - a_arr - 1) // 2 + (b_arr - a_arr - 1)

    h_vals = H[pidx_arr].copy()               # raw counts from H for each pair (l, x)
    if l > 0:
        h_vals[:l] = N - h_vals[:l]           # flip pairs where l is the larger index

    # ── Step 5: aggregate h_vals by block using prefix/suffix arrays ──────────
    # H_gt_block[k] = sum of h_vals for items x residing in base-block k.
    # This is the total "l is ranked after x" signal contributed by block k.
    # H_lt_block[k] = N * |block k| - H_gt_block[k], the complementary count.
    # base_block_idx maps each item to its block in the base structure;
    # we need indices for all items except l, i.e. [0..l-1, l+1..n-1].
    block_ids = np.delete(base_block_idx, l)
    H_gt_block = np.zeros(K_minus)
    np.add.at(H_gt_block, block_ids, h_vals)
    H_lt_block = N * np.array(base_sizes, dtype=float) - H_gt_block

    # prefix_Hlt[p]  = sum of H_lt_block[0..p-1]  (blocks strictly before p)
    # suffix_Hgt[p]  = sum of H_gt_block[p..K_minus-1]  (blocks at p or after)
    # Together they let us evaluate the distance contribution for any candidate
    # placement of l in O(1) using: contrib_l = prefix_Hlt[p] + suffix_Hgt[p].
    prefix_Hlt = np.empty(K_minus + 1)
    prefix_Hlt[0] = 0.0
    np.cumsum(H_lt_block, out=prefix_Hlt[1:])
    suffix_Hgt = np.empty(K_minus + 1)
    suffix_Hgt[K_minus] = 0.0
    np.cumsum(H_gt_block[::-1], out=suffix_Hgt[:K_minus])
    suffix_Hgt[:K_minus] = suffix_Hgt[:K_minus][::-1]

    # ── Step 6: precompute terms that are shared across all candidates ─────────
    q = math.exp(-theta)
    if log_qfact is None:
        log_qfact = build_log_qfactorials(n, q)
    log_gamma_1md = lgamma(1.0 - delta)       # log Γ(1−δ), used in Pitman-Yor size factor
    if use_py_prior:
        # Denominator of the Pitman-Yor (PY) table prior: log[Γ(γ+n)/Γ(γ+1)]
        log_py_denom = lgamma(gamma + n) - lgamma(gamma + 1)
        # PY table-count factor for the base K_minus blocks:
        log_py_tables_base = sum(log(gamma + i * delta) for i in range(1, K_minus))
        # PY block-size factor:
        log_py_sizes_base = sum(lgamma(s - delta) - log_gamma_1md for s in base_sizes)

    # Mallows normalisation base terms:
    base_logP = sum(lgamma(s + 1) for s in base_sizes)
    base_log_qf_sum = sum(log_qfact[s] for s in base_sizes)
    log_qfact_arr = np.asarray(log_qfact)
    sizes_arr = np.asarray(base_sizes, dtype=np.float64)
    sizes_arr_int = sizes_arr.astype(np.intp)

    # ── Step 7: "create" candidate log-weights (vectorised over K_minus+1 positions) ───
    # There are K_minus+1 positions where l can be inserted as a new singleton.
    if use_py_prior:
        log_py_create = (log_py_tables_base + log(gamma + K_minus * delta)
                         + log_py_sizes_base - log_py_denom)
    else:
        log_py_create = 0.0
    log_Z_create = base_logP + float(log_qfact_arr[n]) - base_log_qf_sum - float(log_qfact_arr[1])
    log_ord_create = -lgamma(K_minus + 2) if include_uniform_order_prior else 0.0

    lw_create_common = log_py_create + log_ord_create - N * log_Z_create
    lw_create = lw_create_common - theta * (prefix_Hlt + suffix_Hgt)  # (K_minus+1,)

    # ── Step 8: "add" candidate log-weights (vectorised over K_minus blocks) ──────────
    # There are K_minus candidates, one for each existing block l can join.
    if use_py_prior:
        log_py_add_common = log_py_tables_base + log_py_sizes_base - log_py_denom
    else:
        log_py_add_common = 0.0
    log_Z_add_base = base_logP + float(log_qfact_arr[n]) - base_log_qf_sum
    log_ord_add = -lgamma(K_minus + 1) if include_uniform_order_prior else 0.0

    contrib_l_arr = prefix_Hlt[:K_minus] + suffix_Hgt[1:]  # (K_minus,)
    log_Z_add_arr = (log_Z_add_base + np.log(sizes_arr + 1)
                     - (log_qfact_arr[sizes_arr_int + 1] - log_qfact_arr[sizes_arr_int]))
    if use_py_prior:
        lw_add = (log_py_add_common + np.log(sizes_arr - delta)
                  + log_ord_add - theta * contrib_l_arr - N * log_Z_add_arr)
    else:
        lw_add = log_ord_add - theta * contrib_l_arr - N * log_Z_add_arr

    # ── Step 9: sample from the categorical distribution ──────────────────────
    idx = sample_categorical_from_logweights(np.concatenate([lw_create, lw_add]), rng)

    chosen = _build_candidate(base, l, idx, K_minus)
    return chosen, 1, 1, l


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
    include_uniform_order_prior: bool = True,
    use_py_prior: bool = True,
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
    log_qfact_arr = np.asarray(log_qfact)
    sizes_arr = np.asarray(base_sizes, dtype=np.float64)
    sizes_arr_int = sizes_arr.astype(np.intp)

    base_logP = sum(lgamma(s + 1) for s in base_sizes)
    base_log_qf_sum = sum(log_qfact[s] for s in base_sizes)

    if use_py_prior:
        log_gamma_1md = lgamma(1.0 - delta)
        log_py_denom = lgamma(gamma + n) - lgamma(gamma + 1)
        log_py_tables_base = (
            float(np.log(gamma + np.arange(1, K_minus, dtype=np.float64) * delta).sum())
            if K_minus > 1 else 0.0
        )
        log_py_sizes_base = sum(lgamma(s - delta) - log_gamma_1md for s in base_sizes)

    if use_py_prior:
        log_py_create = (log_py_tables_base + log(gamma + K_minus * delta)
                         + log_py_sizes_base - log_py_denom)
    else:
        log_py_create = 0.0
    log_Z_create = base_logP + float(log_qfact_arr[n]) - base_log_qf_sum - float(log_qfact_arr[1])
    log_ord_create = -lgamma(K_minus + 2) if include_uniform_order_prior else 0.0

    if use_py_prior:
        log_py_add_common = log_py_tables_base + log_py_sizes_base - log_py_denom
    else:
        log_py_add_common = 0.0
    log_Z_add_base = base_logP + float(log_qfact_arr[n]) - base_log_qf_sum
    log_ord_add = -lgamma(K_minus + 1) if include_uniform_order_prior else 0.0

    lw_create_common = log_py_create + log_ord_create - N * log_Z_create
    lw_create = lw_create_common - theta * (prefix_Hlt + suffix_Hgt)  # (K_minus+1,)

    contrib_l_arr = prefix_Hlt[:K_minus] + suffix_Hgt[1:]  # (K_minus,)
    log_Z_add_arr = (log_Z_add_base + np.log(sizes_arr + 1)
                     - (log_qfact_arr[sizes_arr_int + 1] - log_qfact_arr[sizes_arr_int]))
    if use_py_prior:
        lw_add = (log_py_add_common + np.log(sizes_arr - delta)
                  + log_ord_add - theta * contrib_l_arr - N * log_Z_add_arr)
    else:
        lw_add = log_ord_add - theta * contrib_l_arr - N * log_Z_add_arr

    return np.concatenate([lw_create, lw_add]), base, K_minus


def greedy_reassign_one_item(
    blocks: List[List[int]],
    l: int,
    theta: float,
    gamma: float,
    delta: float,
    H: np.ndarray,
    N: int,
    n: int,
    use_py_prior: bool = True,
    include_uniform_order_prior: bool = True,
) -> Tuple[List[List[int]], bool]:
    """Deterministic ICM move: place item *l* at its MAP position."""
    log_weights, base, K_minus = _compute_item_logweights(
        blocks, l, theta, gamma, delta, H, N, n,
        use_py_prior=use_py_prior,
        include_uniform_order_prior=include_uniform_order_prior)
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
    max_sweeps: int = 50,
    use_py_prior: bool = True,
    include_uniform_order_prior: bool = True,
) -> Tuple[List[List[int]], int, int]:
    """Run ICM sweeps over all items until convergence.

    Returns ``(blocks, total_moves, sweeps_done)``.
    """
    total_moves = 0
    for sweep in range(max_sweeps):
        sweep_moves = 0
        for l in range(n):
            blocks, changed = greedy_reassign_one_item(
                blocks, l, theta, gamma, delta, H, N, n,
                use_py_prior=use_py_prior,
                include_uniform_order_prior=include_uniform_order_prior)
            if changed:
                sweep_moves += 1
        total_moves += sweep_moves
        if sweep_moves == 0:
            return blocks, total_moves, sweep + 1
    return blocks, total_moves, max_sweeps
