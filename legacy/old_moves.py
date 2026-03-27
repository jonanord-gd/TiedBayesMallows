"""Legacy MH block-update moves removed from model/moves.py.

Includes:
 - mh_py_prior_reassign_one_item  : PY-prior based single-item MH reassignment
 - mh_adjacent_split_merge        : Adjacent split / merge MH
 - mh_adjacent_item_transfer      : Transfer one item between adjacent blocks MH
 - mh_ordering_swap_or_shift      : Reorder blocks via swap or shift MH
 - helpers: _swap_adjacent, _move_block_to_index, _feasible_shift_positions

Also re-exports the basic block helpers needed to run the above:
 - remove_item_from_blocks, apply_move_existing_block, apply_move_new_block
"""

import math
import random
import time
from typing import List, Tuple, Optional

try:
    from model.blocks import blocks_to_block_index
    from model.distance import total_distance_fast, cross_block_disagreements_fast
    from model.priors import log_Z_star_from_sizes, log_py_eppf_from_sizes, log_blocks_posterior
    from model.utils import sample_categorical_from_logweights
    from model.profiling import get_profiler
except ImportError:
    from TiedBayesMallows.model.blocks import blocks_to_block_index
    from TiedBayesMallows.model.distance import total_distance_fast, cross_block_disagreements_fast
    from TiedBayesMallows.model.priors import log_Z_star_from_sizes, log_py_eppf_from_sizes, log_blocks_posterior
    from TiedBayesMallows.model.utils import sample_categorical_from_logweights
    from TiedBayesMallows.model.profiling import get_profiler


# ── basic block helpers (duplicated here for standalone use) ──────────────

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


# ── MH move: PY-prior item reassignment ──────────────────────────────────

def mh_py_prior_reassign_one_item(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    *,
    rng: random.Random,
    blocks_old: Optional[List[List[int]]] = None,
    distance_calculator=None,
    use_parallel: bool = False,
    posterior_cache: Optional[dict] = None,
) -> Tuple[List[List[int]], Optional[int], Optional[int]]:
    """MH update: remove a random item and reassign it using PY prior weights."""
    if not rankings_c:
        return blocks, 0, 0
    K = len(blocks)
    if K == 0:
        return blocks, 0, 0

    profiler = get_profiler()

    n = len(rankings_c[0])
    x = rng.randrange(n)
    blocks_minus, _ = remove_item_from_blocks(blocks, x)
    K_minus = len(blocks_minus)
    sizes_minus = [len(b) for b in blocks_minus]

    weights: List[float] = []
    candidates: List[Tuple[str, Optional[int]]] = []
    for k in range(K_minus):
        w_py = sizes_minus[k] - delta
        if w_py > 0:
            candidates.append(("existing", k))
            weights.append(w_py)
    w_new = gamma + delta * K_minus
    if w_new > 0:
        candidates.append(("new", None))
        weights.append(w_new)

    if not candidates:
        return blocks, 0, 0

    if profiler:
        t_start = time.time()
    idx = sample_categorical_from_logweights([math.log(w) for w in weights], rng)
    if profiler:
        profiler.record_operation("sampling", time.time() - t_start, "mh_py_reassign")

    kind, where = candidates[idx]
    if kind == "existing":
        prop = apply_move_existing_block(blocks_minus, x, where)  # type: ignore
    else:
        pos = rng.randrange(K_minus + 1)
        prop = apply_move_new_block(blocks_minus, x, pos)

    cache_key = (tuple(tuple(b) for b in blocks), theta, gamma, delta)
    if posterior_cache is not None and cache_key in posterior_cache:
        lp_old = posterior_cache[cache_key]
    else:
        if profiler:
            t_start = time.time()
        lp_old = log_blocks_posterior(rankings_c, blocks, theta, gamma, delta)
        if profiler:
            profiler.record_operation("posterior_calculation", time.time() - t_start, "mh_py_reassign")
        if posterior_cache is not None:
            posterior_cache[cache_key] = lp_old

    if profiler:
        t_start = time.time()
    lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)
    if profiler:
        profiler.record_operation("posterior_calculation", time.time() - t_start, "mh_py_reassign")

    new_cache_key = (tuple(tuple(b) for b in prop), theta, gamma, delta)
    if posterior_cache is not None:
        posterior_cache[new_cache_key] = lp_new

    log_acc = lp_new - lp_old
    if math.log(rng.random()) < log_acc:
        return prop, 1, 1
    return blocks, 1, 0


# ── MH move: adjacent split / merge ──────────────────────────────────────

def mh_adjacent_split_merge(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    *,
    rng: random.Random,
    p_merge: float = 0.5,
) -> Tuple[List[List[int]], Optional[int], Optional[int]]:
    K = len(blocks)
    if K == 0:
        return blocks, 0, 0

    profiler = get_profiler()

    if profiler:
        t_start = time.time()
    lp_old = log_blocks_posterior(rankings_c, blocks, theta, gamma, delta)
    if profiler:
        profiler.record_operation("posterior_calculation", time.time() - t_start, "mh_splitmerge")

    splittable = [j for j, b in enumerate(blocks) if len(b) >= 2]
    can_split = bool(splittable)
    can_merge = (K >= 2)
    if not can_split and not can_merge:
        return blocks, 0, 0

    do_merge = (rng.random() < p_merge)
    if do_merge and not can_merge:
        do_merge = False
    if (not do_merge) and not can_split:
        do_merge = True

    if do_merge:
        j = rng.randrange(K - 1)
        bL = blocks[j]
        bR = blocks[j + 1]
        prop = [b[:] for b in blocks]
        prop[j] = bL[:] + bR[:]
        del prop[j + 1]

        if profiler:
            t_start = time.time()
        lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)
        if profiler:
            profiler.record_operation("posterior_calculation", time.time() - t_start, "mh_splitmerge")

        log_q_fwd = math.log(p_merge) + math.log(1.0 / (K - 1))

        s = len(prop[j])
        splittable2 = [jj for jj, bb in enumerate(prop) if len(bb) >= 2]
        if not splittable2:
            return blocks, 0, 0

        a = len(bL)
        log_q_bwd = (
            math.log(1.0 - p_merge)
            + math.log(1.0 / len(splittable2))
            + math.log(1.0 / (s - 1))
            - math.log(math.comb(s, a))
        )
    else:
        j = rng.choice(splittable)
        block = blocks[j]
        s = len(block)
        a = rng.randrange(1, s)
        A = rng.sample(block, a)
        Aset = set(A)
        B = [x for x in block if x not in Aset]

        prop = [b[:] for b in blocks]
        prop[j] = A[:]
        prop.insert(j + 1, B[:])

        if profiler:
            t_start = time.time()
        lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)
        if profiler:
            profiler.record_operation("posterior_calculation", time.time() - t_start, "mh_splitmerge")

        log_q_fwd = (
            math.log(1.0 - p_merge)
            + math.log(1.0 / len(splittable))
            + math.log(1.0 / (s - 1))
            - math.log(math.comb(s, a))
        )
        K2 = len(prop)
        log_q_bwd = math.log(p_merge) + math.log(1.0 / (K2 - 1))

    log_acc = (lp_new - lp_old) + (log_q_bwd - log_q_fwd)
    if math.log(rng.random()) < log_acc:
        return prop, 1, 1
    return blocks, 1, 0


# ── MH move: adjacent item transfer ──────────────────────────────────────

def mh_adjacent_item_transfer(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    *,
    rng: random.Random,
    blocks_old: Optional[List[List[int]]] = None,
    distance_calculator=None,
    use_parallel: bool = False,
    posterior_cache: Optional[dict] = None,
) -> Tuple[List[List[int]], Optional[int], Optional[int]]:
    K = len(blocks)
    if K < 2:
        return blocks, 0, 0

    profiler = get_profiler()

    moves = []
    for j in range(K - 1):
        if len(blocks[j]) >= 2:
            moves.append((j, +1))
        if len(blocks[j + 1]) >= 2:
            moves.append((j, -1))
    if not moves:
        return blocks, 0, 0

    j, direction = rng.choice(moves)
    donor, recv = (j, j + 1) if direction == +1 else (j + 1, j)
    x = rng.choice(blocks[donor])

    prop = [b[:] for b in blocks]
    prop[donor].remove(x)
    prop[recv].append(x)

    cache_key = (tuple(tuple(b) for b in blocks), theta, gamma, delta)
    if posterior_cache is not None and cache_key in posterior_cache:
        lp_old = posterior_cache[cache_key]
    else:
        if profiler:
            t_start = time.time()
        lp_old = log_blocks_posterior(rankings_c, blocks, theta, gamma, delta)
        if profiler:
            profiler.record_operation("posterior_calculation", time.time() - t_start, "mh_transfer")
        if posterior_cache is not None:
            posterior_cache[cache_key] = lp_old

    if profiler:
        t_start = time.time()
    lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)
    if profiler:
        profiler.record_operation("posterior_calculation", time.time() - t_start, "mh_transfer")

    new_cache_key = (tuple(tuple(b) for b in prop), theta, gamma, delta)
    if posterior_cache is not None:
        posterior_cache[new_cache_key] = lp_new

    log_q_fwd = -math.log(len(moves)) - math.log(len(blocks[donor]))

    moves2 = []
    for jj in range(K - 1):
        if len(prop[jj]) >= 2:
            moves2.append((jj, +1))
        if len(prop[jj + 1]) >= 2:
            moves2.append((jj, -1))
    if not moves2 or (j, -direction) not in moves2:
        return blocks, 0, 0

    rev_donor = recv
    log_q_bwd = -math.log(len(moves2)) - math.log(len(prop[rev_donor]))

    log_acc = (lp_new - lp_old) + (log_q_bwd - log_q_fwd)
    if math.log(rng.random()) < log_acc:
        return prop, 1, 1
    return blocks, 1, 0


# ── ordering helpers ──────────────────────────────────────────────────────

def _swap_adjacent(blocks: List[List[int]], j: int) -> List[List[int]]:
    nb = [b[:] for b in blocks]
    nb[j], nb[j + 1] = nb[j + 1], nb[j]
    return nb


def _move_block_to_index(blocks: List[List[int]], j_from: int, j_to_final: int) -> List[List[int]]:
    K = len(blocks)
    if not (0 <= j_from < K and 0 <= j_to_final < K):
        raise ValueError("Invalid indices for move_block.")
    if j_from == j_to_final:
        return [b[:] for b in blocks]
    nb = [b[:] for b in blocks]
    blk = nb.pop(j_from)
    ins = j_to_final - 1 if j_to_final > j_from else j_to_final
    nb.insert(ins, blk)
    return nb


def _feasible_shift_positions(K: int, j: int, max_step: int) -> List[int]:
    lo = max(0, j - max_step)
    hi = min(K - 1, j + max_step)
    return [p for p in range(lo, hi + 1) if p != j]


# ── MH move: ordering swap / shift ───────────────────────────────────────

def mh_ordering_swap_or_shift(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    *,
    rng: random.Random,
    p_short: float = 0.75,
    n_swap_steps: Optional[int] = None,
    max_long_step: Optional[int] = None,
    blocks_old: Optional[List[List[int]]] = None,
    distance_calculator=None,
    use_parallel: bool = False,
    posterior_cache: Optional[dict] = None,
) -> Tuple[List[List[int]], Optional[int], Optional[int]]:
    """Reorder blocks with minimal moves: 1 adjacent swap OR 1-position shift."""
    K = len(blocks)
    if K <= 1:
        return blocks, 0, 0

    if n_swap_steps is None:
        n_swap_steps = 1
    if max_long_step is None:
        max_long_step = 1

    profiler = get_profiler()

    cache_key = (tuple(tuple(b) for b in blocks), theta, gamma, delta)
    if posterior_cache is not None and cache_key in posterior_cache:
        lp_old = posterior_cache[cache_key]
    else:
        if profiler:
            t_start = time.time()
        lp_old = log_blocks_posterior(rankings_c, blocks, theta, gamma, delta)
        if profiler:
            profiler.record_operation("posterior_calculation", time.time() - t_start, "mh_swapshift")
        if posterior_cache is not None:
            posterior_cache[cache_key] = lp_old

    if rng.random() < p_short:
        prop = [b[:] for b in blocks]
        for _ in range(n_swap_steps):
            j = rng.randrange(K - 1)
            prop = _swap_adjacent(prop, j)

        if profiler:
            t_start = time.time()
        lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)
        if profiler:
            profiler.record_operation("posterior_calculation", time.time() - t_start, "mh_swapshift")

        new_cache_key = (tuple(tuple(b) for b in prop), theta, gamma, delta)
        if posterior_cache is not None:
            posterior_cache[new_cache_key] = lp_new

        if math.log(rng.random()) < (lp_new - lp_old):
            return prop, 1, 1
        return blocks, 1, 0

    j_from = rng.randrange(K)
    feasible = _feasible_shift_positions(K, j_from, max_long_step)
    if not feasible:
        return blocks, 0, 0

    j_to = rng.choice(feasible)
    prop = _move_block_to_index(blocks, j_from, j_to)

    if profiler:
        t_start = time.time()
    lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)
    if profiler:
        profiler.record_operation("posterior_calculation", time.time() - t_start, "mh_swapshift")

    new_cache_key = (tuple(tuple(b) for b in prop), theta, gamma, delta)
    if posterior_cache is not None:
        posterior_cache[new_cache_key] = lp_new

    feasible_rev = _feasible_shift_positions(K, j_to, max_long_step)
    if not feasible_rev:
        return blocks, 0, 0

    log_q_fwd = math.log(1.0 - p_short) - math.log(K) - math.log(len(feasible))
    log_q_bwd = math.log(1.0 - p_short) - math.log(K) - math.log(len(feasible_rev))

    log_acc = (lp_new - lp_old) + (log_q_bwd - log_q_fwd)
    if math.log(rng.random()) < log_acc:
        return prop, 1, 1
    return blocks, 1, 0


# ── slow collapsed Gibbs (O(2K · N n log K) per call) ────────────────────

def gibbs_reassign_one_item(
    rankings: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    include_uniform_order_prior: bool = True,
    rng: Optional[random.Random] = None,
    tie_penalty: float = 0.5,
) -> Tuple[List[List[int]], int, int]:
    """Collapsed Gibbs reassignment using full distance recomputation per candidate.

    Superseded by ``fast_gibbs_reassign_one_item`` in ``model/moves.py``, which
    uses precomputed U_all and prefix-sum H vectors for O(1) per candidate.
    Kept here for reference and comparison.
    """
    if rng is None:
        rng = random.Random()
    if not rankings:
        return [b[:] for b in blocks], 0, 0

    profiler = get_profiler()
    N = len(rankings)
    n = len(rankings[0])

    x = rng.randrange(n)
    blocks_minus, _ = remove_item_from_blocks(blocks, x)
    K_minus = len(blocks_minus)
    sizes_minus = [len(b) for b in blocks_minus]

    candidates: List[Tuple[str, int]] = []
    logW: List[float] = []

    for k in range(K_minus):
        w_py = sizes_minus[k] - delta
        if w_py <= 0:
            continue
        prop = apply_move_existing_block(blocks_minus, x, k)
        if profiler:
            t_start = time.time()
        S = total_distance_fast(rankings, prop, tie_penalty=tie_penalty)
        if profiler:
            profiler.record_operation("distance_calculation", time.time() - t_start, "gibbs_reassign")
        sizes_cand = sizes_minus[:]
        sizes_cand[k] += 1
        logZ = log_Z_star_from_sizes(sizes_cand, theta, None, tie_penalty=tie_penalty)
        lw = math.log(w_py) - theta * S - N * logZ
        if include_uniform_order_prior:
            lw += -math.lgamma(K_minus + 1)
        candidates.append(("existing", k))
        logW.append(lw)

    w_new = gamma + delta * K_minus
    if w_new > 0:
        for pos in range(K_minus + 1):
            prop = apply_move_new_block(blocks_minus, x, pos)
            if profiler:
                t_start = time.time()
            S = total_distance_fast(rankings, prop, tie_penalty=tie_penalty)
            if profiler:
                profiler.record_operation("distance_calculation", time.time() - t_start, "gibbs_reassign")
            sizes_cand = sizes_minus[:]
            sizes_cand.insert(pos, 1)
            K_cand = K_minus + 1
            logZ = log_Z_star_from_sizes(sizes_cand, theta, None, tie_penalty=tie_penalty)
            lw = math.log(w_new) - theta * S - N * logZ
            if include_uniform_order_prior:
                lw += -math.lgamma(K_cand + 1)
            candidates.append(("new", pos))
            logW.append(lw)

    if not candidates:
        return [b[:] for b in blocks], 0, 0

    idx = sample_categorical_from_logweights(logW, rng)
    kind, where = candidates[idx]
    out = (apply_move_existing_block(blocks_minus, x, where)
           if kind == "existing"
           else apply_move_new_block(blocks_minus, x, where))
    return out, 1, 1