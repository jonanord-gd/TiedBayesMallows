import math, random, time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
from functools import lru_cache

# try to import numba for speedups
try:
    from numba import njit
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False

# ============================================================
# ---------- small utilities ----------
# ============================================================


def logsumexp(logw: List[float]) -> float:
    m = max(logw)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(x - m) for x in logw))

def sample_categorical_from_logweights(logw: List[float], rng: random.Random) -> int:
    lse = logsumexp(logw)
    probs = [math.exp(x - lse) for x in logw]
    u = rng.random()
    s = 0.0
    for k, p in enumerate(probs):
        s += p
        if u <= s:
            return k
    return len(probs) - 1

def dirichlet_sample(alpha: List[float], rng: random.Random) -> List[float]:
    xs = [rng.gammavariate(a, 1.0) for a in alpha]
    s = sum(xs)
    return [x / s for x in xs]

def invert_perm(perm: List[int]) -> List[int]:
    inv = [0] * len(perm)
    for pos, item in enumerate(perm):
        inv[item] = pos
    return inv


def init_blocks_borda_threshold(
    rankings: List[List[int]],
    n_clusters: int,
    *,
    initial_ranking: Optional[List[int]] = None,
    gap_threshold: float = 0.35,
    rng: Optional[random.Random] = None,
) -> List[List[List[int]]]:
    """Generate initial tied rankings for each cluster via Borda consensus.

    Parameters
    ----------
    rankings : list of rankings
        Observed data; used to compute Borda consensus if initial_ranking
        is not provided.
    n_clusters : int
        Number of clusters; returns this many slightly-different tied rankings.
    initial_ranking : list of int, optional
        Fixed ranking (permutation of 0..n-1) to use for all clusters.
        If not provided, Borda consensus is computed from the data.
    gap_threshold : float
        Threshold for tying adjacent items in the ranking (in rank units).
        Smaller => fewer ties, larger => more ties.
    rng : random.Random, optional
        RNG for adding per-cluster variation. If None, uses default RNG.

    Returns
    -------
    list of tied-ranking blocks
        One entry per cluster. Each is a list-of-lists (tied ranking).
        If initial_ranking is given, all clusters get the same blocks.
        If not, each cluster gets a slightly perturbed version.
    """
    if rng is None:
        rng = random.Random()

    n = len(rankings[0])

    # Compute Borda consensus from data if no initial ranking provided
    if initial_ranking is None:
        N = len(rankings)
        pos_sum = [0.0] * n
        for r in rankings:
            inv = [0] * n
            for p, item in enumerate(r):
                inv[item] = p
            for item in range(n):
                pos_sum[item] += inv[item]
        mean_pos = [s / N for s in pos_sum]
        initial_ranking = sorted(range(n), key=lambda i: mean_pos[i])

    def make_tied_ranking(ranking: List[int]) -> List[List[int]]:
        """Convert strict ranking to tied blocks using gap_threshold."""
        # compute mean positions for this ranking
        pos_map = {item: pos for pos, item in enumerate(ranking)}
        pos_vals = [pos_map[item] for item in ranking]

        blocks = []
        cur = [ranking[0]]
        for a, b in zip(ranking, ranking[1:]):
            if abs(pos_map[b] - pos_map[a]) <= gap_threshold:
                cur.append(b)
            else:
                blocks.append(cur)
                cur = [b]
        blocks.append(cur)
        return blocks

    # Generate per-cluster rankings with random perturbations
    cluster_rankings = []
    for c in range(n_clusters):
        if n_clusters == 1:
            # Single cluster: use initial ranking as-is
            perturbed = initial_ranking[:]
        else:
            # Multiple clusters: add random variation via small swaps
            perturbed = initial_ranking[:]
            # Perform a few random adjacent swaps to create variation
            # (keeps the ranking close but not identical)
            n_swaps = max(1, n // 5)  # ~20% of items get shuffled nearby
            for _ in range(n_swaps):
                i = rng.randrange(n - 1)
                # Swap with neighbor with some probability
                if rng.random() < 0.5:
                    perturbed[i], perturbed[i + 1] = perturbed[i + 1], perturbed[i]

        blocks = make_tied_ranking(perturbed)
        cluster_rankings.append(blocks)

    return cluster_rankings


# ============================================================
# ---------- blocks helpers ----------
# ============================================================

def blocks_to_block_index(blocks: List[List[int]], n: int, *, validate: bool = True) -> List[int]:
    """Returns block index for each item in 0..n-1."""
    block_idx = [-1] * n
    for b, block in enumerate(blocks):
        for item in block:
            if validate and (item < 0 or item >= n):
                raise ValueError(f"Item {item} out of range 0..{n-1}.")
            if block_idx[item] != -1:
                raise ValueError("Item appears twice in blocks.")
            block_idx[item] = b
    if validate and any(x < 0 for x in block_idx):
        raise ValueError("Blocks do not cover all items.")
    return block_idx

def T_of_sizes(sizes: List[int]) -> int:
    return sum(s * (s - 1) // 2 for s in sizes)

def T_of_blocks(blocks: List[List[int]]) -> int:
    return T_of_sizes([len(b) for b in blocks])

# ============================================================
# ------- Fast cross-block disagreements via inversions ------
# ============================================================

class Fenwick:
    __slots__ = ("n", "bit")
    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, i: int, delta: int) -> None:
        i += 1
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def sum_prefix(self, i: int) -> int:
        i += 1
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s

# numba-accelerated inversion count if available
if _USE_NUMBA:
    @njit
    def _cross_block_disagreements_fast_nb(strict_r, block_idx, K):
        # Fenwick tree implemented with local array
        fw = [0] * (K + 1)
        seen = 0
        inv = 0
        for idx in range(len(strict_r)):
            item = strict_r[idx]
            b = block_idx[item]
            # prefix sum
            s = 0
            i = b + 1
            while i <= K:
                s += fw[i]
                i += i & -i
            inv += seen - s
            # add
            i = b + 1
            while i <= K:
                fw[i] += 1
                i += i & -i
            seen += 1
        return inv

def cross_block_disagreements_fast(strict_r: List[int], block_idx: List[int], K: int) -> int:
    """# inversions in block-index sequence along strict_r."""
    if _USE_NUMBA:
        # convert to simple lists for Numba
        return _cross_block_disagreements_fast_nb(strict_r, block_idx, K)
    fw = Fenwick(K)
    seen = 0
    inv = 0
    for item in strict_r:
        b = block_idx[item]
        leq = fw.sum_prefix(b)
        inv += seen - leq
        fw.add(b, 1)
        seen += 1
    return inv

def total_distance_given_block_index(
    rankings: List[List[int]],
    block_index_fn: Callable[[int], int],
    K: int,
    Tm: int
) -> int:
    """Compute sum_i (2*disc_i + Tm) with disc_i via inversion count."""
    total = 0
    for r in rankings:
        fw = Fenwick(K)
        seen = 0
        inv = 0
        for item in r:
            b = block_index_fn(item)
            leq = fw.sum_prefix(b)
            inv += seen - leq
            fw.add(b, 1)
            seen += 1
        total += 2 * inv + Tm
    return total

# ============================================================
# ---------- Z* helpers ----------
# ============================================================

def build_log_qfactorials(n: int, q: float) -> List[float]:
    """log([k]_q!) for k=0..n in O(n) for fixed q."""
    if n < 0:
        raise ValueError("n must be >= 0")
    out = [0.0] * (n + 1)
    if n == 0:
        return out
    if q <= 0.0:
        return out
    if abs(q - 1.0) < 1e-12:
        for k in range(1, n + 1):
            out[k] = math.lgamma(k + 1)
        return out

    log_denom = math.log(1.0 - q)
    qpow = q
    acc = 0.0
    for i in range(1, n + 1):
        acc += math.log(1.0 - qpow) - log_denom
        out[i] = acc
        qpow *= q
    return out

# internal cached computation when log_qfact is not provided
@lru_cache(maxsize=None)
def _log_Z_star_core(sizes_tuple: Tuple[int, ...], theta: float) -> float:
    # builds qfactorials internally; sizes_tuple is tuple of ints
    sizes = list(sizes_tuple)
    if theta <= 0:
        return float("-inf")
    n = sum(sizes)
    q = math.exp(-2.0 * theta)

    Tm = T_of_sizes(sizes)
    logP = sum(math.lgamma(s + 1) for s in sizes)

    log_qfact = build_log_qfactorials(n, q)
    return (-theta * Tm) + logP + (log_qfact[n] - sum(log_qfact[s] for s in sizes))


def log_Z_star_from_sizes(sizes: List[int], theta: float, log_qfact: Optional[List[float]] = None) -> float:
    # if caller didn't supply precomputed log_qfactorials, use cached core
    if log_qfact is None:
        return _log_Z_star_core(tuple(sizes), theta)

    if theta <= 0:
        return float("-inf")
    n = sum(sizes)
    q = math.exp(-2.0 * theta)

    Tm = T_of_sizes(sizes)
    logP = sum(math.lgamma(s + 1) for s in sizes)

    if len(log_qfact) < n + 1:
        log_qfact = build_log_qfactorials(n, q)

    return (-theta * Tm) + logP + (log_qfact[n] - sum(log_qfact[s] for s in sizes))

def log_Z_star(blocks: List[List[int]], theta: float) -> float:
    return log_Z_star_from_sizes([len(b) for b in blocks], theta, None)

# ============================================================
# ---------- PY prior ----------
# ============================================================

def log_py_eppf_from_sizes(sizes: List[int], gamma: float, delta: float) -> float:
    """""Delta is the discount, gamma the concentration. See Pitman-Yor EPPF formula."""
    if gamma <= -delta:
        return float("-inf")
    if not (0.0 <= delta < 1.0):
        return float("-inf")
    n = sum(sizes)
    K = len(sizes)
    if n == 0 or K == 0:
        return 0.0

    log_num_tables = 0.0
    for i in range(1, K):
        term = gamma + i * delta
        if term <= 0:
            return float("-inf")
        log_num_tables += math.log(term)

    log_num_sizes = 0.0
    log_gamma_1md = math.lgamma(1.0 - delta)
    for s in sizes:
        if s <= 0:
            return float("-inf")
        log_num_sizes += math.lgamma(s - delta) - log_gamma_1md

    log_denom = math.lgamma(gamma + n) - math.lgamma(gamma + 1.0)
    return log_num_tables + log_num_sizes - log_denom

# ============================================================
# ---------- log posterior for blocks ----------
# ============================================================

def total_distance_fast(rankings: List[List[int]], blocks: List[List[int]]) -> int:
    """Sum_i d(r_i, blocks). Uses inversion counting (O(N n log K)).

    A numba-accelerated variant is used when available, which speeds the
    per-ranking loop substantially.  The logic is otherwise identical to
    the original implementation.
    """
    if not rankings:
        return 0
    n = len(rankings[0])
    sizes = [len(b) for b in blocks]
    K = len(sizes)
    blk = blocks_to_block_index(blocks, n, validate=False)
    Tm = T_of_sizes(sizes)

    if _USE_NUMBA:
        # convert python lists to simple arrays for njit call
        # numba supports typed List; easiest is to use same cross-count function
        total = 0
        for r in rankings:
            inv = _cross_block_disagreements_fast_nb(r, blk, K)
            total += 2 * inv + Tm
        return total

    total = 0
    for r in rankings:
        disc = cross_block_disagreements_fast(r, blk, K)
        total += 2 * disc + Tm
    return total

def log_blocks_posterior(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
) -> float:
    if not rankings_c:
        return float("-inf")
    sizes = [len(b) for b in blocks]
    K = len(sizes)
    S = total_distance_fast(rankings_c, blocks)
    logZ = log_Z_star_from_sizes(sizes, theta, None)
    logpy = log_py_eppf_from_sizes(sizes, gamma, delta)
    return (-theta * S) - (len(rankings_c) * logZ) + logpy - math.lgamma(K + 1)

# ============================================================
# ---------- Gibbs reassignment (single item) ----------
# ============================================================

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

def gibbs_reassign_one_item(
    rankings: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    include_uniform_order_prior: bool = True,
    rng: Optional[random.Random] = None
) -> List[List[int]]:
    if rng is None:
        rng = random.Random()
    if not rankings:
        return [b[:] for b in blocks], 0, 0

    N = len(rankings)
    n = len(rankings[0])

    x = rng.randrange(n)
    blocks_minus, _ = remove_item_from_blocks(blocks, x)
    K_minus = len(blocks_minus)
    sizes_minus = [len(b) for b in blocks_minus]

    candidates: List[Tuple[str, int]] = []
    logW: List[float] = []

    # existing blocks: compute full distance using Numba-optimized code
    for k in range(K_minus):
        w_py = sizes_minus[k] - delta
        if w_py <= 0:
            continue

        prop = apply_move_existing_block(blocks_minus, x, k)
        S = total_distance_fast(rankings, prop)
        sizes_cand = sizes_minus[:]
        sizes_cand[k] += 1

        logZ = log_Z_star_from_sizes(sizes_cand, theta, None)

        lw = math.log(w_py) - theta * S - N * logZ
        if include_uniform_order_prior:
            lw += -math.lgamma(K_minus + 1)
        candidates.append(("existing", k))
        logW.append(lw)

    # new singleton at each position
    w_new = gamma + delta * K_minus
    if w_new > 0:
        for pos in range(K_minus + 1):
            prop = apply_move_new_block(blocks_minus, x, pos)
            S = total_distance_fast(rankings, prop)
            sizes_cand = sizes_minus[:]
            sizes_cand.insert(pos, 1)
            K_cand = K_minus + 1

            logZ = log_Z_star_from_sizes(sizes_cand, theta, None)

            lw = math.log(w_new) - theta * S - N * logZ
            if include_uniform_order_prior:
                lw += -math.lgamma(K_cand + 1)
            candidates.append(("new", pos))
            logW.append(lw)

    if not candidates:
        return [b[:] for b in blocks], 0, 0

    idx = sample_categorical_from_logweights(logW, rng)
    kind, where = candidates[idx]
    out = apply_move_existing_block(blocks_minus, x, where) if kind == "existing" else apply_move_new_block(blocks_minus, x, where)
    # Gibbs is a single proposal and is always considered accepted (collapsed Gibbs)
    return out, 1, 1


# ------------------------------------------------------------
# New MH move: propose item reassignment from PY prior only
# ------------------------------------------------------------
def mh_py_prior_reassign_one_item(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    *,
    rng: random.Random,
) -> List[List[int]]:
    """MH update: remove a random item and reassign it using PY prior weights.

    The proposal ignores the likelihood (distance) term and samples new block
    placement according to the Pitman–Yor prior.  The acceptance probability is
    based solely on the ratio of posterior probabilities (likelihood+prior).

    Returns a tuple ``(blocks_out, n_proposals, n_accepts)`` as with other
    MH helpers.
    """
    if not rankings_c:
        return blocks, 0, 0
    K = len(blocks)
    if K == 0:
        return blocks, 0, 0

    # pick a random item and remove it
    n = len(rankings_c[0])
    x = rng.randrange(n)
    blocks_minus, old_blk = remove_item_from_blocks(blocks, x)
    K_minus = len(blocks_minus)
    sizes_minus = [len(b) for b in blocks_minus]

    # compute PY prior weights
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

    # sample candidate according to log weights
    idx = sample_categorical_from_logweights([math.log(w) for w in weights], rng)
    kind, where = candidates[idx]
    if kind == "existing":
        prop = apply_move_existing_block(blocks_minus, x, where)  # type: ignore
    else:
        # choose a random insertion position for the new singleton
        pos = rng.randrange(K_minus + 1)
        prop = apply_move_new_block(blocks_minus, x, pos)

    # compute MH acceptance based on full posterior
    lp_old = log_blocks_posterior(rankings_c, blocks, theta, gamma, delta)
    lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)
    log_acc = lp_new - lp_old
    if math.log(rng.random()) < min(0.0, log_acc):
        return prop, 1, 1
    return blocks, 1, 0

# ============================================================
# ---------- MH moves ----------
# ============================================================

def mh_adjacent_split_merge(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    *,
    rng: random.Random,
    p_merge: float = 0.5,
) -> List[List[int]]:
    K = len(blocks)
    if K == 0:
        return blocks, 0, 0

    lp_old = log_blocks_posterior(rankings_c, blocks, theta, gamma, delta)

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

        lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)

        log_q_fwd = math.log(p_merge) + math.log(1.0 / (K - 1))

        # reverse split must pick merged block and reconstruct exact (bL, bR)
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

        lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)

        log_q_fwd = (
            math.log(1.0 - p_merge)
            + math.log(1.0 / len(splittable))
            + math.log(1.0 / (s - 1))
            - math.log(math.comb(s, a))
        )
        K2 = len(prop)
        log_q_bwd = math.log(p_merge) + math.log(1.0 / (K2 - 1))

    log_acc = (lp_new - lp_old) + (log_q_bwd - log_q_fwd)
    if math.log(rng.random()) < min(0.0, log_acc):
        return prop, 1, 1
    return blocks, 1, 0

def mh_adjacent_item_transfer(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    *,
    rng: random.Random,
) -> List[List[int]]:
    K = len(blocks)
    if K < 2:
        return blocks, 0, 0

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

    # build proposal
    prop = [b[:] for b in blocks]
    prop[donor].remove(x)
    prop[recv].append(x)

    # compute likelihoods using Numba-optimized distance
    lp_old = log_blocks_posterior(rankings_c, blocks, theta, gamma, delta)
    lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)

    log_q_fwd = -math.log(len(moves)) - math.log(len(blocks[donor]))

    # check if reverse move is possible
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
    if math.log(rng.random()) < min(0.0, log_acc):
        return prop, 1, 1
    return blocks, 1, 0

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
) -> List[List[int]]:
    K = len(blocks)
    if K <= 1:
        return blocks, 0, 0
    if n_swap_steps is None:
        n_swap_steps = max(1, int(round(math.sqrt(K))))
    if max_long_step is None:
        max_long_step = min(K - 1, max(2, int(round(K / 2))))

    lp_old = log_blocks_posterior(rankings_c, blocks, theta, gamma, delta)

    if rng.random() < p_short:
        prop = [b[:] for b in blocks]
        for _ in range(n_swap_steps):
            j = rng.randrange(K - 1)
            prop = _swap_adjacent(prop, j)

        lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)
        if math.log(rng.random()) < min(0.0, lp_new - lp_old):
            return prop, 1, 1
        return blocks, 1, 0

    j_from = rng.randrange(K)
    feasible = _feasible_shift_positions(K, j_from, max_long_step)
    if not feasible:
        return blocks, 0, 0

    j_to = rng.choice(feasible)
    prop = _move_block_to_index(blocks, j_from, j_to)
    lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)

    feasible_rev = _feasible_shift_positions(K, j_to, max_long_step)
    if not feasible_rev:
        return blocks, 0, 0

    log_q_fwd = math.log(1.0 - p_short) - math.log(K) - math.log(len(feasible))
    log_q_bwd = math.log(1.0 - p_short) - math.log(K) - math.log(len(feasible_rev))

    log_acc = (lp_new - lp_old) + (log_q_bwd - log_q_fwd)
    if math.log(rng.random()) < min(0.0, log_acc):
        return prop, 1, 1
    return blocks, 1, 0

# ============================================================
# ---------- parameter/state dataclasses ----------
# ============================================================

@dataclass
class ClusterParams:
    blocks: List[List[int]]
    theta: float
    gamma: float
    delta: float

@dataclass
class MixtureState:
    clusters: List[ClusterParams]
    z: List[int]
    tau: List[float]

@dataclass
class MCMCSamples:
    z_samples: List[List[int]]
    blocks_samples: List[List[List[List[int]]]]  # [t][c][block][item]
    tau_samples: Optional[List[List[float]]] = None
    theta_samples: Optional[List[List[float]]] = None
    logp: Optional[List[float]] = None          # length T
    K: Optional[List[List[int]]] = None         # [t][c] number of blocks per cluster
    # Acceptance indicators saved per stored iteration: [t][c] 0/1
    theta_accepts: Optional[List[List[int]]] = None
    block_accepts: Optional[List[List[int]]] = None
    # Detailed per-proposal counters (optional, may be large)
    theta_proposals: Optional[List[List[int]]] = None
    theta_accept_counts: Optional[List[List[int]]] = None
    block_proposals: Optional[List[List[int]]] = None
    block_accept_counts: Optional[List[List[int]]] = None

from dataclasses import dataclass

@dataclass
class SamplerConfig:
    n_item_moves_per_cluster: int = 2   # number of item-level PY moves to attempt per cluster each iteration

    # PY prior parameters used by the various block moves.  Clusters have
    # their own `gamma`/`delta` fields, but setting these values here overrides
    # them for every cluster (with ``None`` meaning "use the cluster value").
    gamma: Optional[float] = None   # Pitman–Yor discount parameter (0<=delta<1)
    delta: Optional[float] = None   # Pitman–Yor strength/concentration parameter

    # mixture weights for the four block-update types; they are normalized
    # internally so only their ratios matter.
    p_gibbs: float = 0.0            # probability of Gibbs reassign-one-item move
    p_transfer: float = 0.4         # probability of adjacent-item-transfer move
    p_swapshift: float = 0.4        # probability of ordering swap/shift move
    p_splitmerge: float = 0.2       # probability of adjacent split/merge move
    p_py: float = 0.0               # probability of PY-prior MH single-item reassign move

    # parameters specific to the ordering swap/shift move
    ordering_p_short: float = 0.75   # chance of proposing a "short" swap vs long shift
    ordering_n_swap_steps: Optional[int] = None  # max number of random swap steps when using short move
    ordering_max_long_step: Optional[int] = None  # max displacement for long shift move

    # parameter for split/merge move probability of proposing merge given a split choice
    splitmerge_p_merge: float = 0.5

    # Metropolis–Hastings parameters for updating theta
    a_theta: float = 2.0            # shape of Gamma prior on theta
    b_theta: float = 1.0            # rate of Gamma prior on theta
    theta_step: float = 0.25        # log-normal proposal stddev for theta updates


# ============================================================
# ---------- MAP / summaries ----------
# ============================================================

def _canonicalize_blocks(blocks: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(sorted(block)) for block in blocks)

def estimate_z_from_frequency(z_samples: List[List[int]], *, C: int) -> Dict[str, Any]:
    T = len(z_samples)
    N = len(z_samples[0])
    counts = [[0] * C for _ in range(N)]
    for z in z_samples:
        for i, c in enumerate(z):
            counts[i][c] += 1
    p_ic = [[counts[i][c] / T for c in range(C)] for i in range(N)]
    z_hat = [max(range(C), key=lambda c: p_ic[i][c]) for i in range(N)]
    return {"p_ic": p_ic, "z_hat": z_hat}

def _posterior_mode_from_counts(counts: Dict[Any, int]) -> Tuple[Any, float, int]:
    total = sum(counts.values())
    mode_val = max(counts.keys(), key=lambda k: counts[k])
    mode_count = counts[mode_val]
    mode_prob = mode_count / total if total else float("nan")
    return mode_val, mode_prob, mode_count

def summarize_theta(theta_samples_c: List[float], *, ci: float = 0.95, map_bins: int = 50) -> Dict[str, float]:
    xs = sorted(theta_samples_c)
    n = len(xs)
    mean = sum(xs) / n
    median = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    a = (1 - ci) / 2
    lo = xs[int(math.floor(a * (n - 1)))]
    hi = xs[int(math.floor((1 - a) * (n - 1)))]

    x_min, x_max = xs[0], xs[-1]
    if x_max == x_min:
        map_est = x_min
    else:
        binw = (x_max - x_min) / map_bins
        bins = [0] * map_bins
        for x in xs:
            k = min(map_bins - 1, int((x - x_min) / binw))
            bins[k] += 1
        kmax = max(range(map_bins), key=lambda k: bins[k])
        map_est = x_min + (kmax + 0.5) * binw
    return {"mean": mean, "median": median, "ci_lo": lo, "ci_hi": hi, "map": map_est}

# ============================================================
# ---------- model object with caching ----------
# ============================================================

class MixtureRankingModel:
    """
    Single entry-point object:
      - initialize(...)
      - run_mcmc(...)
      - estimate_map(...)
    Includes lightweight caches so we don't rebuild block indices / T(m) repeatedly.
    """

    def __init__(
        self,
        rankings: List[List[int]],
        *,
        init_clusters: Optional[List[ClusterParams]] = None,
        n_clusters: Optional[int] = None,
        init_mu: Optional[List[float]] = None,
        # convenience shortcuts for cluster hyperparameters; if provided
        # they override the corresponding field of every element in
        # ``init_clusters``.  This mirrors ``init_mu`` which is separate from the
        # cluster definitions.
        init_theta: Optional[float] = None,
        init_gamma: Optional[float] = None,
        init_delta: Optional[float] = None,
        # parameters for auto-generating clusters (used when init_clusters is None)
        borda_gap_threshold: float = 0.35,
        seed: int = 123,
        verbose: bool = False,
    ):
        """Create a mixture model from observed rankings.

        Parameters
        ----------
        rankings : list of permutations
            The observed strict rankings (each a permutation of 0..n-1).
        init_clusters : list of ClusterParams, optional
            Initial configuration for each cluster (blocks, theta, gamma, delta).
            If omitted, clusters are auto-generated via Borda consensus with
            per-cluster variation. The list length determines the number of 
            clusters C.
        n_clusters : int, optional
            Number of clusters to create (ignored if init_clusters is provided).
            Required if init_clusters is None.
        init_mu : list of float, optional
            Dirichlet prior parameters for the cluster weights. If omitted a
            symmetric prior with all entries 1.0 is used.
        init_theta : float, optional
            Initial theta value for all clusters. If given, overrides cluster
            values.
        init_gamma : float, optional
            Common PY gamma to use for all clusters.
        init_delta : float, optional
            Common PY delta to use for all clusters.
        borda_gap_threshold : float
            Threshold for tying adjacent items in Borda-derived rankings.
            Only used when init_clusters is None. Smaller => fewer ties.
        seed : int
            Random seed for reproducibility.
        verbose : bool
            Enable verbose logging of model setup and MCMC progress.
        """
        if not rankings:
            raise ValueError("rankings must be non-empty")
        self.rankings = rankings
        self.N = len(rankings)
        self.n = len(rankings[0])
        if any(len(r) != self.n for r in rankings):
            raise ValueError("All rankings must have same length")

        self.rng = random.Random(seed)

        # Auto-generate clusters if not provided
        if init_clusters is None:
            if n_clusters is None:
                raise ValueError("Either init_clusters or n_clusters must be provided")
            if n_clusters <= 0:
                raise ValueError("n_clusters must be positive")
            
            # Generate per-cluster block rankings via Borda consensus with variation
            cluster_blocks_list = init_blocks_borda_threshold(
                rankings,
                n_clusters,
                gap_threshold=borda_gap_threshold,
                rng=self.rng,
            )
            
            # Set default values for theta, gamma, delta
            theta_val = init_theta if init_theta is not None else 1.0
            gamma_val = init_gamma if init_gamma is not None else 1.0
            delta_val = init_delta if init_delta is not None else 0.5
            
            init_clusters = [
                ClusterParams(
                    blocks=cluster_blocks_list[c],
                    theta=theta_val,
                    gamma=gamma_val,
                    delta=delta_val,
                )
                for c in range(n_clusters)
            ]
        else:
            # Use provided clusters
            if n_clusters is not None:
                raise ValueError("Cannot specify both init_clusters and n_clusters")
            # Apply any overrides to the provided clusters
            if init_theta is not None or init_gamma is not None or init_delta is not None:
                for cl in init_clusters:
                    if init_theta is not None:
                        cl.theta = init_theta
                    if init_gamma is not None:
                        cl.gamma = init_gamma
                    if init_delta is not None:
                        cl.delta = init_delta

        self.C = len(init_clusters)
        if self.C <= 0:
            raise ValueError("Need at least one cluster")

        self.init_mu = init_mu if init_mu is not None else [1.0] * self.C
        if len(self.init_mu) != self.C:
            raise ValueError("init_mu must have length C")

        # state
        z0 = [self.rng.randrange(self.C) for _ in range(self.N)]
        tau0 = dirichlet_sample(self.init_mu, self.rng)
        self.state = MixtureState(clusters=init_clusters, z=z0, tau=tau0)

        # per-cluster caches (updated when blocks change)
        self._rebuild_all_cluster_caches()

        # Initiate samples form MCMC run
        self.samples: Optional[MCMCSamples] = None

        # Sampler config
        self.cfg = SamplerConfig()
        
        # Verbose logging flag
        self.verbose = verbose
        if self.verbose:
            print(f"[Model] Initialized ({self.N} assessors, {self.n} items, {self.C} clusters)")
            print(f"[Model] Numba JIT: {'enabled' if _USE_NUMBA else 'disabled'}")
            for c, cl in enumerate(self.state.clusters):
                K = len(cl.blocks)
                print(f"  Cluster {c}: {K} blocks, theta={cl.theta:.3f}, gamma={cl.gamma:.3f}, delta={cl.delta:.3f}")
    # -----------------------
    # caching
    # -----------------------
    @dataclass
    class _ClusterCache:
        sizes: List[int]
        K: int
        Tm: int
        block_idx: List[int]   # length n

    def _rebuild_cluster_cache(self, c: int) -> None:
        cl = self.state.clusters[c]
        sizes = [len(b) for b in cl.blocks]
        K = len(sizes)
        Tm = T_of_sizes(sizes)
        blk = blocks_to_block_index(cl.blocks, self.n, validate=False)
        self._cache[c] = MixtureRankingModel._ClusterCache(
            sizes=sizes, K=K, Tm=Tm, block_idx=blk
        )

    def _rebuild_all_cluster_caches(self) -> None:
        self._cache: List[MixtureRankingModel._ClusterCache] = [None] * self.C  # type: ignore
        for c in range(self.C):
            self._rebuild_cluster_cache(c)

    # -----------------------
    # core updates
    # -----------------------
    def _update_z(self) -> None:
        """Uses cached block_idx, K, Tm; only logZ depends on theta each step."""
        log_tau = [math.log(t) for t in self.state.tau]
        logZ = [log_Z_star_from_sizes(self._cache[c].sizes, self.state.clusters[c].theta, None) for c in range(self.C)]

        for i, r_i in enumerate(self.rankings):
            logw = []
            for c in range(self.C):
                cc = self._cache[c]
                theta = self.state.clusters[c].theta
                disc = cross_block_disagreements_fast(r_i, cc.block_idx, cc.K)
                d_ic = 2 * disc + cc.Tm
                logw.append(log_tau[c] - theta * d_ic - logZ[c])
            self.state.z[i] = sample_categorical_from_logweights(logw, self.rng)

    def _update_tau(self) -> None:
        counts = [0] * self.C
        for zi in self.state.z:
            counts[zi] += 1
        post = [self.init_mu[c] + counts[c] for c in range(self.C)]
        self.state.tau = dirichlet_sample(post, self.rng)

    def _cluster_rankings(self, c: int) -> List[List[int]]:
        return [r for r, zi in zip(self.rankings, self.state.z) if zi == c]

    @staticmethod
    def _log_gamma_pdf(x: float, a: float, b: float) -> float:
        if x <= 0:
            return float("-inf")
        return (a - 1.0) * math.log(x) - b * x + a * math.log(b) - math.lgamma(a)

    def _update_cluster_theta(
        self,
        c: int,
        rankings_c: List[List[int]],
        *,
        a_theta: float = 2.0,
        b_theta: float = 1.0,
        step: float = 0.25
    ) -> bool:
        n_c = len(rankings_c)
        if n_c == 0:
            return 0, 0

        cl = self.state.clusters[c]
        cc = self._cache[c]

        theta_old = cl.theta
        theta_new = math.exp(math.log(theta_old) + self.rng.gauss(0.0, step))

        # S_c using cached block_idx/K/Tm; delegate to vectorized helper
        S_c = total_distance_fast(rankings_c, cl.blocks)

        lp_old = (-theta_old * S_c) - n_c * log_Z_star_from_sizes(cc.sizes, theta_old, None) + self._log_gamma_pdf(theta_old, a_theta, b_theta)
        lp_new = (-theta_new * S_c) - n_c * log_Z_star_from_sizes(cc.sizes, theta_new, None) + self._log_gamma_pdf(theta_new, a_theta, b_theta)

        log_hast = math.log(theta_new) - math.log(theta_old)
        log_acc = (lp_new - lp_old) + log_hast

        accepted = math.log(self.rng.random()) < min(0.0, log_acc)
        if accepted:
            cl.theta = theta_new
        # (theta changes do not invalidate cache, since cache is only blocks-derived)
        # One proposal was attempted for theta; return proposal and accept counts
        return 1, (1 if accepted else 0)

    def _update_cluster_blocks(
        self,
        c: int,
        rankings_c: List[List[int]],
        *,
        n_item_moves: int = 2,
        p_gibbs: float = 0.0,
        p_transfer: float = 0.4,
        p_swapshift: float = 0.4,
        p_splitmerge: float = 0.2,
        p_py: float = 0.0,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
    ) -> bool:
        if not rankings_c:
            return 0, 0, False
        cl = self.state.clusters[c]

        # allow sampler-wide overrides for the PY hyperparameters
        if gamma is None:
            gamma = cl.gamma
        if delta is None:
            delta = cl.delta

        s = p_gibbs + p_transfer + p_swapshift + p_splitmerge + p_py
        p_gibbs, p_transfer, p_swapshift, p_splitmerge, p_py = (
            p_gibbs / s,
            p_transfer / s,
            p_swapshift / s,
            p_splitmerge / s,
            p_py / s,
        )

        # record before state to detect whether any MH move was accepted
        before_key = _canonicalize_blocks(cl.blocks)

        proposals = 0
        accepts = 0

        for _ in range(n_item_moves):
            u = self.rng.random()
            if u < p_gibbs:
                out_blocks, p, a = gibbs_reassign_one_item(
                    rankings=rankings_c,
                    blocks=cl.blocks,
                    theta=cl.theta,
                    gamma=gamma,
                    delta=delta,
                    rng=self.rng
                )
                # Gibbs considered a single proposal and treated as accepted (a==1)
            elif u < p_gibbs + p_transfer:
                out_blocks, p, a = mh_adjacent_item_transfer(
                    rankings_c=rankings_c,
                    blocks=cl.blocks,
                    theta=cl.theta,
                    gamma=gamma,
                    delta=delta,
                    rng=self.rng
                )
            elif u < p_gibbs + p_transfer + p_swapshift:
                out_blocks, p, a = mh_ordering_swap_or_shift(
                    rankings_c=rankings_c,
                    blocks=cl.blocks,
                    theta=cl.theta,
                    gamma=gamma,
                    delta=delta,
                    rng=self.rng,
                    p_short=self.cfg.ordering_p_short,
                    n_swap_steps=self.cfg.ordering_n_swap_steps,
                    max_long_step=self.cfg.ordering_max_long_step,
                )
            else:
                out_blocks, p, a = mh_adjacent_split_merge(
                    rankings_c=rankings_c,
                    blocks=cl.blocks,
                    theta=cl.theta,
                    gamma=gamma,
                    delta=delta,
                    rng=self.rng,
                    p_merge=self.cfg.splitmerge_p_merge,
                )

            proposals += (p if p is not None else 0)
            accepts += (a if a is not None else 0)
            cl.blocks = out_blocks

        # blocks changed -> refresh cache
        after_key = _canonicalize_blocks(cl.blocks)
        self._rebuild_cluster_cache(c)

        # return proposal/accept counts and whether any block changed
        return proposals, accepts, (before_key != after_key)

    # -----------------------
    # public API
    # -----------------------
    def set_sampler(self, **kwargs) -> None:
        """ 
        Configures the SamplerConfig parameters for the MCMC sampler.
        This allows users to adjust the behavior of the sampler by passing keyword arguments corresponding to the fields in SamplerConfig.
        For example, set_sampler(p_gibbs=0.2) would set the probability of Gibbs moves to 0.2 while keeping other parameters at their default values.

        The same keyword arguments may now also be supplied directly to
        :meth:`run_mcmc` (see its docstring) so that it is no longer necessary
        to call :meth:`set_sampler` ahead of time unless you want a persistent
        configuration between multiple runs.
        """
        for k, v in kwargs.items():
            if not hasattr(self.cfg, k):
                raise ValueError(f"Unknown sampler setting: {k}")
            setattr(self.cfg, k, v)

    def step(self, **overrides) -> Dict[str, List[int]]:
        """Performs one MCMC step, updating z, tau, and cluster blocks/theta.
        Accepts optional overrides for sampler settings (e.g. step(p_gibbs=0.2)) 
        that temporarily adjust the behavior of this step without changing the underlying SamplerConfig.
        
        If you specify any of the block-move probabilities (p_gibbs, p_transfer,
        p_swapshift, p_splitmerge), the unspecified ones are automatically set to
        zero.  This makes it easy to specify a clean sampling regime, e.g.
        step(p_gibbs=1.0) uses only Gibbs moves.
        
        For longer runs it is usually more convenient to supply sampler
        options directly to :meth:`run_mcmc` so they apply across many steps
        (see that method's docstring).  In particular, you can control which
        block-update moves are used via ``p_gibbs``, ``p_transfer``,
        ``p_swapshift``, ``p_splitmerge`` and the new ``p_py`` parameter that
        activates a Metropolis-Hastings move sampling a single-item reassignment
        from the Pitman–Yor prior.        """
        cfg = self.cfg
        
        # If user specifies any block-move probability, auto-zero the unspecified ones
        block_move_keys = {'p_gibbs', 'p_transfer', 'p_swapshift', 'p_splitmerge', 'p_py'}
        specified_keys = block_move_keys & set(overrides.keys())
        if specified_keys:
            for key in block_move_keys - specified_keys:
                overrides[key] = 0.0
        
        # allow per-call overrides (e.g. step(p_gibbs=0.2))
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown sampler setting: {k}")
            setattr(cfg, k, v)

        self._update_z()
        self._update_tau()

        theta_accepts: List[int] = [0] * self.C
        block_accepts: List[int] = [0] * self.C

        # detailed per-cluster proposal/accept counts
        theta_proposals: List[int] = [0] * self.C
        theta_accept_counts: List[int] = [0] * self.C
        block_proposals: List[int] = [0] * self.C
        block_accept_counts: List[int] = [0] * self.C

        for c in range(self.C):
            Rc = self._cluster_rankings(c)
            # blocks: returns (n_proposals, n_accepts, changed_bool)
            bp, ba, block_changed = self._update_cluster_blocks(
                c, Rc,
                n_item_moves=cfg.n_item_moves_per_cluster,
                p_gibbs=cfg.p_gibbs,
                p_transfer=cfg.p_transfer,
                p_swapshift=cfg.p_swapshift,
                p_splitmerge=cfg.p_splitmerge,
                gamma=cfg.gamma,
                delta=cfg.delta,
            )
            block_proposals[c] = int(bp)
            block_accept_counts[c] = int(ba)
            block_accepts[c] = 1 if block_changed else 0

            # theta: returns (n_proposals, n_accepts)
            tp, ta = self._update_cluster_theta(
                c, Rc,
                a_theta=cfg.a_theta,
                b_theta=cfg.b_theta,
                step=cfg.theta_step,
            )
            theta_proposals[c] = int(tp)
            theta_accept_counts[c] = int(ta)
            theta_accepts[c] = 1 if ta > 0 else 0

        return {
            "theta_accepts": theta_accepts,
            "block_accepts": block_accepts,
            "theta_proposals": theta_proposals,
            "theta_accept_counts": theta_accept_counts,
            "block_proposals": block_proposals,
            "block_accept_counts": block_accept_counts,
        }

    def run_mcmc(
        self,
        n_iter: int,
        *,
        burn_in: int = 0,
        thin: int = 1,
        save_samples: bool = True,
        save_tau: bool = False,
        save_theta: bool = False,
        save_logp: bool = True,
        save_acceptance_details: bool = False,
        n_item_moves_per_cluster: int = 2,
        **sampler_kwargs,
    ) -> Tuple[MixtureState, Optional[MCMCSamples]]:
        """Runs MCMC for ``n_iter`` iterations, with optional burn-in and thinning.

        Any keyword arguments beyond the ones explicitly listed above are
        interpreted as sampler configuration parameters.  They are passed to
        :meth:`set_sampler` and therefore correspond exactly to the fields of
        :class:`SamplerConfig` (```gamma```, ```delta```, ```p_gibbs```,
        ```theta_step```, ```n_item_moves_per_cluster```, etc.).  This means you can specify the
        full sampler behaviour when calling :meth:`run_mcmc` without needing to
        call :meth:`set_sampler` in advance.  Unknown sampler keywords will
        result in a :class:`ValueError`.

        **Smart defaults for block-move probabilities**: If you specify any of
        the five block move probabilities (```p_gibbs```, ```p_transfer```,
        ```p_swapshift```, ```p_splitmerge```, ```p_py```), the unspecified ones
        are automatically set to zero.  This provides a clean way to define a
        specific sampling regime, e.g.
        ``run_mcmc(..., p_gibbs=1.0)`` uses only Gibbs moves or
        ``run_mcmc(..., p_transfer=0.5, p_swapshift=0.5)`` uses a 50–50 mix.  The
        new ``p_py`` option activates the PY-prior MH proposal described above.
        The explicit parameters ``burn_in``, ``thin``, ``save_samples`` and
        friends control which iterations are stored and what diagnostics are
        recorded.  ``n_item_moves_per_cluster`` is kept as a named argument for
        backwards compatibility, but it may also be supplied via the sampler
        kwargs.

        Returns the final state and the collected samples (if any)."""
        if n_iter <= 0:
            raise ValueError("n_iter must be positive")
        if thin <= 0:
            raise ValueError("thin must be >= 1")
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0")

        # apply any sampler-specific options upfront; this updates ``self.cfg``
        if sampler_kwargs:
            # detect if user specified any block-move probabilities;
            # if so, auto-zero the unspecified ones for clarity
            block_move_keys = {'p_gibbs', 'p_transfer', 'p_swapshift', 'p_splitmerge', 'p_py'}
            specified_keys = block_move_keys & set(sampler_kwargs.keys())
            if specified_keys:
                for key in block_move_keys - specified_keys:
                    sampler_kwargs[key] = 0.0
            
            # we treat ``n_item_moves_per_cluster`` specially since it is
            # both an explicit argument and a config field; pop it if present
            if "n_item_moves_per_cluster" in sampler_kwargs:
                n_item_moves_per_cluster = sampler_kwargs.pop("n_item_moves_per_cluster")
            # remaining keys should all correspond to SamplerConfig fields
            self.set_sampler(**sampler_kwargs)

        samples: Optional[MCMCSamples] = None
        if save_samples:
            samples = MCMCSamples(
                z_samples=[],
                blocks_samples=[],
                tau_samples=[] if save_tau else None,
                theta_samples=[] if save_theta else None,
                K = [],
                logp = [],
                theta_accepts = [],
                block_accepts = [],
                theta_proposals = [] if save_acceptance_details else None,
                theta_accept_counts = [] if save_acceptance_details else None,
                block_proposals = [] if save_acceptance_details else None,
                block_accept_counts = [] if save_acceptance_details else None,
            )

        def snapshot() -> None:
            """""Helper to save current state to samples at each saved iteration."""""
            assert samples is not None
            samples.z_samples.append(self.state.z[:])
            samples.blocks_samples.append([[b[:] for b in cl.blocks] for cl in self.state.clusters])
            if save_tau and samples.tau_samples is not None:
                samples.tau_samples.append(self.state.tau[:])
            if save_theta and samples.theta_samples is not None:
                samples.theta_samples.append([cl.theta for cl in self.state.clusters])
            # K per cluster
            samples.K.append([len(cl.blocks) for cl in self.state.clusters])

            # optional: joint log posterior (simple version; see helper below)
            if save_logp and samples.logp is not None:
                samples.logp.append(self.log_joint())

        save_details = save_acceptance_details

        # Startup diagnostics
        if self.verbose:
            print(f"\n[MCMC] Starting run: n_iter={n_iter}, burn_in={burn_in}, thin={thin}")
            print(f"[MCMC] Sampler config: p_gibbs={self.cfg.p_gibbs:.3f}, p_transfer={self.cfg.p_transfer:.3f}, p_swapshift={self.cfg.p_swapshift:.3f}, p_splitmerge={self.cfg.p_splitmerge:.3f}, p_py={self.cfg.p_py:.3f}")
            print(f"[MCMC] Item moves per cluster: {n_item_moves_per_cluster}")
            saved_iters = (n_iter - burn_in + thin - 1) // thin if save_samples else 0
            print(f"[MCMC] Will save {saved_iters} iterations after burn-in\n")

        t_start = time.time()
        theta_accepts_per_cluster = [0] * self.C
        block_accepts_per_cluster = [0] * self.C
        theta_proposals_per_cluster = [0] * self.C
        block_proposals_per_cluster = [0] * self.C

        for it in range(n_iter):
            # ``step`` will respect the (possibly updated) ``self.cfg``
            info = self.step(n_item_moves_per_cluster=n_item_moves_per_cluster)
            
            # accumulate acceptance statistics
            for c in range(self.C):
                theta_accepts_per_cluster[c] += info["theta_accepts"][c]
                block_accepts_per_cluster[c] += info["block_accepts"][c]
                theta_proposals_per_cluster[c] += info.get("theta_proposals", [0]*self.C)[c]
                block_proposals_per_cluster[c] += info.get("block_proposals", [0]*self.C)[c]
            
            if save_samples and it >= burn_in and ((it - burn_in) % thin == 0):
                snapshot()
                # record acceptance indicators for this saved iteration
                assert samples is not None
                samples.theta_accepts.append(info["theta_accepts"])
                samples.block_accepts.append(info["block_accepts"])
                if save_details:
                    # detailed counts (per-proposal)
                    if samples.theta_proposals is not None:
                        samples.theta_proposals.append(info.get("theta_proposals", [0]*self.C))
                    if samples.theta_accept_counts is not None:
                        samples.theta_accept_counts.append(info.get("theta_accept_counts", [0]*self.C))
                    if samples.block_proposals is not None:
                        samples.block_proposals.append(info.get("block_proposals", [0]*self.C))
                    if samples.block_accept_counts is not None:
                        samples.block_accept_counts.append(info.get("block_accept_counts", [0]*self.C))

            # Progress reporting: at 10%, 25%, 50%, 75%, 90%, and end
            if self.verbose and it > 0:
                milestones = [int(0.1*n_iter), int(0.25*n_iter), int(0.5*n_iter), int(0.75*n_iter), int(0.9*n_iter), n_iter-1]
                if it in milestones:
                    elapsed = time.time() - t_start
                    pct = 100.0 * (it + 1) / n_iter
                    rate = (it + 1) / elapsed
                    eta_sec = (n_iter - it - 1) / rate if rate > 0 else 0
                    eta_min = eta_sec / 60
                    print(f"[MCMC] {pct:6.1f}% | iter {it+1:6d}/{n_iter} | {elapsed:7.1f}s elapsed | ETA {eta_min:5.1f} min")

        t_elapsed = time.time() - t_start

        # Final summary
        if self.verbose:
            print(f"\n[MCMC] Run complete in {t_elapsed:.1f}s ({t_elapsed/60:.2f} min)")
            if samples and samples.logp:
                logp_vals = samples.logp
                # Format logp values with scientific notation if too long
                def format_logp(x):
                    fixed = f"{x:.2f}"
                    return f"{x:.2e}" if len(fixed) > 15 else fixed
                
                final_str = format_logp(logp_vals[-1])
                min_str = format_logp(min(logp_vals))
                max_str = format_logp(max(logp_vals))
                print(f"[MCMC] Final logp: {final_str} (min={min_str}, max={max_str})")
            
            # Per-cluster acceptance rates
            print(f"[MCMC] Acceptance rates (theta | blocks):")
            for c in range(self.C):
                theta_rate = theta_accepts_per_cluster[c] / max(1, theta_proposals_per_cluster[c]) if theta_proposals_per_cluster[c] > 0 else 0
                block_rate = block_accepts_per_cluster[c] / max(1, block_proposals_per_cluster[c]) if block_proposals_per_cluster[c] > 0 else 0
                print(f"  Cluster {c}: {theta_rate:.1%} | {block_rate:.1%}")

        self.samples = samples
        return self.state, samples

    def estimate_map(
        self,
        samples: Optional[MCMCSamples] = None,
        *,
        ci: float = 0.95
    ) -> Dict[str, Any]:
        """Frequency MAP-like summaries (no label switching handling).

        ``samples`` can either be supplied explicitly (the object returned by
        ``run_mcmc``) or omitted, in which case the method uses
        ``self.samples`` stored on the model.  Providing ``samples`` makes it
        easier to work with the result of ``run_mcmc`` in a single expression.

        Summaries include:
          - *z*: posterior mode via argmax of frequencies (soft probs also
            returned)
          - *blocks* per cluster: most frequent sampled ordered partition
          - *theta*: mean/median/CI + crude histogram MAP (requires
            ``theta_samples``)
        """
        if samples is None:
            if self.samples is None:
                raise RuntimeError("No samples available. Run run_mcmc(...) first with save_samples=True.")
            samples = self.samples

        z_samples = samples.z_samples
        blocks_samples = samples.blocks_samples
        if not z_samples or not blocks_samples:
            raise ValueError("Need z_samples and blocks_samples")

        out: Dict[str, Any] = {}
        out["N"] = len(z_samples[0])
        out["C"] = self.C
        out["T"] = len(z_samples)

        out["labels"] = estimate_z_from_frequency(z_samples, C=self.C)

        consensus_blocks = []
        for c in range(self.C):
            counts: Dict[Tuple[Tuple[int, ...], ...], int] = {}
            for t in range(len(z_samples)):
                key = _canonicalize_blocks(blocks_samples[t][c])
                counts[key] = counts.get(key, 0) + 1
            mode_key, mode_prob, mode_count = _posterior_mode_from_counts(counts)
            blocks_hat = [list(block) for block in mode_key]
            consensus_blocks.append({
                "cluster": c,
                "blocks_hat": blocks_hat,
                "posterior_prob": mode_prob,
                "count": mode_count,
                "n_unique": len(counts),
            })
        out["consensus_blocks"] = consensus_blocks

        if samples.theta_samples is not None:
            theta_summary = []
            for c in range(self.C):
                theta_c = [samples.theta_samples[t][c] for t in range(len(z_samples))]
                theta_summary.append({"cluster": c, **summarize_theta(theta_c, ci=ci)})
            out["theta_summary"] = theta_summary
        else:
            out["theta_summary"] = None

        return out
    
    def acceptance_statistics(self, parameter: Optional[str] = None) -> Dict[str, Any]:
        """Compute acceptance statistics after a run.

        This delegates to AcceptanceProbabilities.acceptance_probabilities
        and returns a dictionary of acceptance summaries. Requires that
        `run_mcmc(..., save_logp=True)` was used so that `self.samples.logp`
        is available.
        """
        from AcceptanceProbabilities import acceptance_probabilities

        if self.samples is None:
            raise RuntimeError("No samples available. Run run_mcmc() first with save_samples=True.")

        return acceptance_probabilities(self.samples, self.C, parameter=parameter)

    def print_acceptance_summary(self) -> None:
        """Print a human-readable acceptance summary to stdout.

        Requires `run_mcmc(..., save_logp=True)` so log-posterior trace is present.
        """
        from AcceptanceProbabilities import print_acceptance_summary

        if self.samples is None:
            raise RuntimeError("No samples available. Run run_mcmc() first with save_samples=True.")

        print_acceptance_summary(self.samples, self.C)
    
    def log_joint(self) -> float:
        """
        Unnormalized log posterior (up to constants) of current state.
        Useful as a trace diagnostic.
        """
        lp = 0.0

        # tau prior: Dirichlet(mu) has log p(tau) = const + sum((mu_c-1)log tau_c)
        # (const omitted)
        for c, t in enumerate(self.state.tau):
            if t <= 0:
                return float("-inf")
            lp += (self.init_mu[c] - 1.0) * math.log(t)

        # z likelihood part
        # p(z|tau) = prod_i tau_{z_i}
        for zi in self.state.z:
            lp += math.log(self.state.tau[zi])

        # cluster blocks + theta likelihood + priors (as in your updates)
        for c, cl in enumerate(self.state.clusters):
            Rc = self._cluster_rankings(c)
            if Rc:
                lp += log_blocks_posterior(Rc, cl.blocks, cl.theta, cl.gamma, cl.delta)

                # theta prior (match update_cluster_theta defaults if you want)
                # if you used a_theta=2, b_theta=1:
                lp += self._log_gamma_pdf(cl.theta, 2.0, 1.0)

        return lp
    
    def _require_samples(self) -> None:
        if self.samples is None:
            raise RuntimeError("Run run_mcmc(..., save_samples=True) first.")

    def plot_trace_theta(self, *, burn: int = 0) -> None:
        self._require_samples()
        import matplotlib.pyplot as plt

        assert self.samples is not None
        assert self.samples.theta_samples is not None, "Run with save_theta=True"

        T = len(self.samples.theta_samples)
        xs = list(range(burn, T))

        for c in range(self.C):
            ys = [self.samples.theta_samples[t][c] for t in xs]
            plt.figure()
            plt.plot(xs, ys)
            plt.title(f"Theta trace (cluster {c})")
            plt.xlabel("Saved iteration")
            plt.ylabel("theta")
            plt.show()

    def plot_trace_tau(self, *, burn: int = 0) -> None:
        self._require_samples()
        import matplotlib.pyplot as plt

        assert self.samples is not None
        assert self.samples.tau_samples is not None, "Run with save_tau=True"

        T = len(self.samples.tau_samples)
        xs = list(range(burn, T))

        for c in range(self.C):
            ys = [self.samples.tau_samples[t][c] for t in xs]
            plt.figure()
            plt.plot(xs, ys)
            plt.title(f"Tau trace (cluster {c})")
            plt.xlabel("Saved iteration")
            plt.ylabel("tau")
            plt.show()

    def plot_trace_K(self, *, burn: int = 0) -> None:
        self._require_samples()
        import matplotlib.pyplot as plt

        assert self.samples.K is not None

        T = len(self.samples.K)
        xs = list(range(burn, T))

        for c in range(self.C):
            ys = [self.samples.K[t][c] for t in xs]
            plt.figure()
            plt.plot(xs, ys)
            plt.title(f"#Blocks K trace (cluster {c})")
            plt.xlabel("Saved iteration")
            plt.ylabel("K")
            plt.show()

    def plot_trace_logp(self, *, burn: int = 0) -> None:
        self._require_samples()
        import matplotlib.pyplot as plt

        assert self.samples.logp is not None

        xs = list(range(burn, len(self.samples.logp)))
        ys = self.samples.logp[burn:]

        plt.figure()
        plt.plot(xs, ys)
        plt.title("Log joint trace (unnormalized)")
        plt.xlabel("Saved iteration")
        plt.ylabel("logp")
        plt.show()


