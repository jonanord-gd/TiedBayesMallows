import math, random
from dataclasses import dataclass
from typing import List, Dict, Any,  Optional, Tuple
from itertools import permutations
try:
    from numba import njit
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False
    def njit(f): return f

# ---------- small utilities ----------
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
    # simple Dirichlet via Gamma
    xs = [rng.gammavariate(a, 1.0) for a in alpha]
    s = sum(xs)
    return [x / s for x in xs]

def invert_perm(perm: List[int]) -> List[int]:
    # Returns the position of each item, given a ranking "perm"
    inv = [0] * len(perm)
    for pos, item in enumerate(perm):
        inv[item] = pos
    return inv


# ============================================================
# ------------- Old helpers ----------------------------------
# ============================================================
def blocks_to_block_index(blocks: List[List[int]], n: int) -> List[int]:
    # Returns the block index of each item
    block_idx = [-1] * n
    for b, block in enumerate(blocks):
        for item in block:
            if block_idx[item] != -1:
                raise ValueError("Item appears twice in blocks.")
            block_idx[item] = b
    if any(x < 0 for x in block_idx):
        raise ValueError("Blocks do not cover all items.")
    return block_idx


def T_of_blocks(blocks: List[List[int]]) -> int:
    # Number of ties given the block structure
    return sum(len(b) * (len(b) - 1) // 2 for b in blocks)


def cross_block_disagreements(strict_r: List[int], blocks: List[List[int]]) -> int:
    n = len(strict_r)
    pos_r = invert_perm(strict_r)
    blk = blocks_to_block_index(blocks, n)

    disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            bi, bj = blk[i], blk[j]
            if bi == bj:
                continue
            rho_says_i_before_j = (bi < bj)
            r_says_i_before_j = (pos_r[i] < pos_r[j])
            if rho_says_i_before_j != r_says_i_before_j:
                disc += 1
    return disc



def kemeny_distance_strict_vs_tied(strict_r: List[int], blocks: List[List[int]]) -> int:
    # Your decomposition: d(r, rho_m) = 2*disc + T(m)
    disc = cross_block_disagreements(strict_r, blocks)
    return 2 * disc + T_of_blocks(blocks)


def total_distance(rankings: List[List[int]], blocks: List[List[int]]) -> int:
    return sum(kemeny_distance_strict_vs_tied(r, blocks) for r in rankings)


# ============================================================
# --------- Partition / block helpers ------------------------
# ============================================================

def blocks_to_block_index(blocks: List[List[int]], n: int) -> List[int]:
    # Returns the block index of each item
    block_idx = [-1] * n
    for b, block in enumerate(blocks):
        for item in block:
            if block_idx[item] != -1:
                raise ValueError("Item appears twice in blocks.")
            block_idx[item] = b
    if any(x < 0 for x in block_idx):
        raise ValueError("Blocks do not cover all items.")
    return block_idx


def T_of_sizes(sizes: List[int]) -> int:
    return sum(s * (s - 1) // 2 for s in sizes)

def T_of_blocks(blocks: List[List[int]]) -> int:
    return T_of_sizes([len(b) for b in blocks])


# ============================================================
# ------- Fast cross-block disagreements via inversions ------
# ============================================================

# Fenwick tree helper functions (Numba-compatible)
def fenwick_build(n: int) -> List[int]:
    """Initialize Fenwick tree array."""
    return [0] * (n + 1)

@njit
def fenwick_add(bit: List[int], i: int, delta: int, n: int) -> None:
    """Add delta to position i (0-based externally)."""
    i += 1
    while i <= n:
        bit[i] += delta
        i += i & -i

@njit
def fenwick_sum_prefix(bit: List[int], i: int) -> int:
    """Sum for indices [0..i] inclusive (0-based)."""
    i += 1
    s = 0
    while i > 0:
        s += bit[i]
        i -= i & -i
    return s

# Legacy class interface (kept for backward compatibility)
class Fenwick:
    __slots__ = ("n", "bit")
    def __init__(self, n: int):
        self.n = n
        self.bit = fenwick_build(n)

    def add(self, i: int, delta: int) -> None:
        fenwick_add(self.bit, i, delta, self.n)

    def sum_prefix(self, i: int) -> int:
        return fenwick_sum_prefix(self.bit, i)
    

@njit
def cross_block_disagreements_fast(
    strict_r: List[int],
    block_idx: List[int],
    K: int
) -> int:
    """
    Count disagreements between strict ranking r and tied center defined by ordered blocks.

    This is the number of inversions in the sequence of block indices when items are read
    in the strict order strict_r, ignoring within-block pairs (equal block index).
    Complexity: O(n log K). JIT-compiled for speed.
    """
    # Initialize Fenwick tree
    bit = [0] * (K + 1)
    seen = 0
    inv = 0
    
    for item in strict_r:
        b = block_idx[item]
        # Count items with block index <= b seen so far
        i = b + 1
        leq = 0
        while i > 0:
            leq += bit[i]
            i -= i & -i
        
        inv += seen - leq
        
        # Add this block index to Fenwick tree
        i = b + 1
        while i <= K:
            bit[i] += 1
            i += i & -i
        
        seen += 1
    
    return inv

def kemeny_distance_strict_vs_tied_fast(
    strict_r: List[int],
    blocks: List[List[int]],
    *,
    validate_blocks: bool = False
) -> int:
    n = len(strict_r)
    sizes = [len(b) for b in blocks]
    K = len(sizes)
    block_idx = blocks_to_block_index(blocks, n)
    # cross_block_disagreements_fast is now JIT-compiled
    disc = cross_block_disagreements_fast(strict_r, block_idx, K)
    return 2 * disc + T_of_sizes(sizes)

def total_distance_fast(rankings: List[List[int]], blocks: List[List[int]]) -> int:
    """Sum_i d(r_i, blocks). Uses inversion counting (O(N n log K)). JIT-friendly."""
    n = len(rankings[0])
    sizes = [len(b) for b in blocks]
    K = len(sizes)
    block_idx = blocks_to_block_index(blocks, n)
    Tm = T_of_sizes(sizes)
    total = 0
    for r in rankings:
        # This inner function call is JIT-compiled
        disc = cross_block_disagreements_fast(r, block_idx, K)
        total += 2 * disc + Tm
    return total

# -----------------------------
# Normalization constant Z*_n(m, theta) (Old)
# -----------------------------

def qfactorial(n: int, q: float) -> float:
    # [n]_q! = prod_{i=1}^n (1 + q + ... + q^(i-1))
    if n <= 0:
        return 1.0
    if q <= 0.0:
        return 1.0 
    if abs(q - 1.0) < 1e-12:
        return math.factorial(n)

    log_val = 0.0
    log_denom = math.log(1.0 - q)
    for i in range(1, n + 1):
        log_val += math.log(1.0 - q**i) - log_denom
    return math.exp(log_val)

def log_Z_star(blocks: List[List[int]], theta: float) -> float:
    """
    log Z*_n(m,theta) for strict observed rankings and tied center.
    Uses block sizes only (equivalently the multiplicity vector m_s).
    """
    if theta <= 0:
        return float("-inf")

    n = sum(len(b) for b in blocks)
    q = math.exp(-2.0 * theta)

    # T(m)
    Tm = T_of_blocks(blocks)

    # log P(m) = sum log(s!)
    logP = sum(math.lgamma(len(b) + 1) for b in blocks)

    # q-factorial ratio
    log_qfact_n = math.log(qfactorial(n, q))
    log_qfact_blocks = sum(math.log(qfactorial(len(b), q)) for b in blocks)

    return (-theta * Tm) + logP + (log_qfact_n - log_qfact_blocks)


# ============================================================
# ---------------- Z*_n(m, theta)  ---------------------------
# ============================================================

# We don't need to precompute ALL qfactorials, only does of spesific sizes. Might be faster though

def build_log_qfactorials(n: int, q: float) -> List[float]:
    """
    Precompute log([k]_q!) for k=0..n in O(n) for fixed q.
    [k]_q! = prod_{i=1}^k (1 + q + ... + q^(i-1)) = prod_{i=1}^k (1 - q^i)/(1-q)
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    out = [0.0] * (n + 1)
    if n == 0:
        return out

    if q <= 0.0:
        # degenerate-ish; treat as 1
        return out

    if abs(q - 1.0) < 1e-12:
        # log factorials
        for k in range(1, n + 1):
            out[k] = math.lgamma(k + 1)
        return out

    log_denom = math.log(1.0 - q)
    qpow = q  # q^1
    acc = 0.0
    for i in range(1, n + 1):
        # acc += log(1 - q^i) - log(1 - q)
        acc += math.log(1.0 - qpow) - log_denom
        out[i] = acc
        qpow *= q
    return out

def log_Z_star_from_sizes(sizes: List[int], theta: float, log_qfact: Optional[List[float]] = None) -> float:
    """
    log Z*_n(m, theta), depends only on block sizes (not labels).
    Pass log_qfact precomputed for n and q = exp(-2theta) for speed in tight loops.
    """
    if theta <= 0:
        return float("-inf")

    n = sum(sizes)
    q = math.exp(-2.0 * theta)

    Tm = T_of_sizes(sizes)
    logP = sum(math.lgamma(s + 1) for s in sizes)

    if log_qfact is None or len(log_qfact) < n + 1:
        log_qfact = build_log_qfactorials(n, q)

    log_qfact_n = log_qfact[n]
    log_qfact_blocks = sum(log_qfact[s] for s in sizes)

    return (-theta * Tm) + logP + (log_qfact_n - log_qfact_blocks)

def log_Z_star(blocks: List[List[int]], theta: float) -> float:
    return log_Z_star_from_sizes([len(b) for b in blocks], theta, None)

# -----------------------------
# Single-item reassignment Gibbs move under PY prior 
# -----------------------------


def remove_item_from_blocks(blocks: List[List[int]], x: int) -> Tuple[List[List[int]], int]:
    """
    Remove item x from blocks. Returns (new_blocks, old_block_index_before_removal).
    Removes empty blocks.
    """
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

def total_distance_given_block_index(rankings: List[List[int]], block_index_fn, K: int, Tm: int) -> int:
    """
    Compute sum_i (2*disc_i + Tm) where disc_i is inversion count of block indices along ranking i.
    block_index_fn(item)-> block index in 0..K-1.
    """
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

def gibbs_reassign_one_item(
    rankings: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    include_uniform_order_prior: bool = True,
    rng: Optional[random.Random] = None
) -> List[List[int]]:
    """
    Collapsed Gibbs-like reassignment of one item x under PY(gamma, delta),
    targeting posterior ∝ exp(-theta S(rho)) * Z*(m,theta)^(-N) * PY_prior(partition) * (1/K!) (optional).

    """
    if rng is None:
        rng = random.Random()

    N = len(rankings)
    n = len(rankings[0])

    # Pick an item uniformly
    x = rng.randrange(n)

    # Remove x
    blocks_minus, _ = remove_item_from_blocks(blocks, x)
    K_minus = len(blocks_minus)
    sizes_minus = [len(b) for b in blocks_minus]

    # Precompute log q-factorials for this theta once (used by all candidates)
    log_qfact = build_log_qfactorials(n, math.exp(-2.0 * theta))

    # Candidate moves:
    #   ("existing", k) for k in 0..K_minus-1
    #   ("new", pos) for pos in 0..K_minus
    candidates: List[Tuple[str, int]] = []
    logW: List[float] = []

    # -----------------
    # existing blocks
    # -----------------
    for k in range(K_minus):
        w_py = sizes_minus[k] - delta
        if w_py <= 0:
            continue

        # Candidate sizes
        sizes_cand = sizes_minus[:]
        sizes_cand[k] += 1
        K_cand = K_minus
        Tm = T_of_sizes(sizes_cand)

        # block index function for this candidate:
        # all items except x map to their block in blocks_minus; x maps to k
        # Need item->block for blocks_minus:
        blk_minus = blocks_to_block_index(blocks_minus, n)

        def block_index_fn(item, _blk=blk_minus, _x=x, _k=k):
            return _k if item == _x else _blk[item]

        S = total_distance_given_block_index(rankings, block_index_fn, K_cand, Tm)
        logZ = log_Z_star_from_sizes(sizes_cand, theta, log_qfact)

        lw = math.log(w_py) - theta * S - N * logZ
        if include_uniform_order_prior:
            lw += -math.lgamma(K_cand + 1)  # -log(K!)
        candidates.append(("existing", k))
        logW.append(lw)

    # -----------------
    # new singleton block at each position
    # -----------------
    w_new = gamma + delta * K_minus
    if w_new > 0:
        blk_minus = blocks_to_block_index(blocks_minus, n)

        for pos in range(K_minus + 1):
            # sizes: insert 1 at pos
            sizes_cand = sizes_minus[:]
            sizes_cand.insert(pos, 1)
            K_cand = K_minus + 1
            Tm = T_of_sizes(sizes_cand)

            # mapping: x -> pos; existing block b shifts by +1 if b >= pos
            def block_index_fn(item, _blk=blk_minus, _x=x, _pos=pos):
                if item == _x:
                    return _pos
                b = _blk[item]
                return b if b < _pos else b + 1

            S = total_distance_given_block_index(rankings, block_index_fn, K_cand, Tm)
            logZ = log_Z_star_from_sizes(sizes_cand, theta, log_qfact)

            lw = math.log(w_new) - theta * S - N * logZ
            if include_uniform_order_prior:
                lw += -math.lgamma(K_cand + 1)  # -log(K!)
            candidates.append(("new", pos))
            logW.append(lw)

    if not candidates:
        return [b[:] for b in blocks]

    idx = sample_categorical_from_logweights(logW, rng)
    kind, where = candidates[idx]

    # Materialize chosen candidate blocks only once
    if kind == "existing":
        return apply_move_existing_block(blocks_minus, x, where)
    else:
        return apply_move_new_block(blocks_minus, x, where)


# ---------- cluster state ----------
@dataclass
class ClusterParams:
    blocks: List[List[int]]   # rho_{m_c} as ordered blocks
    theta: float              # theta_c
    gamma: float              # PY strength (λ_c in notes)
    delta: float              # PY discount (α_c in notes)


@dataclass
class MixtureState:
    clusters: List[ClusterParams]  # length C
    z: List[int]                   # length N_assessors, values 0..C-1
    tau: List[float]               # length C



# ---------- updates ----------
def update_z(rankings: List[List[int]], state: MixtureState, rng: random.Random) -> None:
    """
    For each assessor i:
      log w_ic = log tau_c - theta_c * d(r_i, blocks_c) - log Z*(blocks_c, theta_c)
    Sampling z_i from categorical proportional to w_ic.

    """
    C = len(state.clusters)
    log_tau = [math.log(t) for t in state.tau]
    logZ = [log_Z_star(cl.blocks, cl.theta) for cl in state.clusters]

    # Precompute block indices + Tm for each cluster to avoid recomputing inside the i loop
    # (still O(C*n), cheap compared to N loop)
    cluster_cached = []
    n = len(rankings[0])
    for cl in state.clusters:
        sizes = [len(b) for b in cl.blocks]
        K = len(sizes)
        blk = blocks_to_block_index(cl.blocks, n)
        Tm = T_of_sizes(sizes)
        cluster_cached.append((K, blk, Tm, cl.theta))

    for i, r_i in enumerate(rankings):
        logw = []
        for c in range(C):
            K, blk, Tm, theta = cluster_cached[c]
            disc = cross_block_disagreements_fast(r_i, blk, K)
            d_ic = 2 * disc + Tm
            logw.append(log_tau[c] - theta * d_ic - logZ[c])
        state.z[i] = sample_categorical_from_logweights(logw, rng)  


def cluster_rankings(rankings: List[List[int]], z: List[int], c: int) -> List[List[int]]:
    return [r for r, zi in zip(rankings, z) if zi == c]

def update_tau(state: MixtureState, mu: List[float], rng: random.Random) -> None:
    """
    tau | z ~ Dirichlet(mu_c + n_c).  (eq. 26) :contentReference[oaicite:9]{index=9}
    """
    C = len(state.clusters)
    counts = [0] * C
    for zi in state.z:
        counts[zi] += 1
    post = [mu[c] + counts[c] for c in range(C)]
    state.tau = dirichlet_sample(post, rng)

def log_gamma_pdf(x: float, a: float, b: float) -> float:
    if x <= 0:
        return float("-inf")
    return (a - 1.0) * math.log(x) - b * x + a * math.log(b) - math.lgamma(a)


def update_cluster_theta(
    rankings_c: List[List[int]],
    cl: ClusterParams,
    rng: random.Random,
    a_theta: float = 2.0,
    b_theta: float = 1.0,
    step: float = 0.25
) -> None:
    """
    MH update for theta_c with cluster-specific likelihood:
      log post(theta) = -theta S_c - n_c log Z*(blocks,theta) + log GammaPrior(theta) + const

    """
    n_c = len(rankings_c)
    if n_c == 0:
        return

    theta_old = cl.theta
    theta_new = math.exp(math.log(theta_old) + rng.gauss(0.0, step))

    # compute S_c once using fast distance pieces
    n = len(rankings_c[0])
    sizes = [len(b) for b in cl.blocks]
    K = len(sizes)
    blk = blocks_to_block_index(cl.blocks, n)
    Tm = T_of_sizes(sizes)

    S_c = 0
    for r in rankings_c:
        disc = cross_block_disagreements_fast(r, blk, K)
        S_c += 2 * disc + Tm

    lp_old = (-theta_old * S_c) - n_c * log_Z_star_from_sizes(sizes, theta_old, None) + log_gamma_pdf(theta_old, a_theta, b_theta)
    lp_new = (-theta_new * S_c) - n_c * log_Z_star_from_sizes(sizes, theta_new, None) + log_gamma_pdf(theta_new, a_theta, b_theta)

    # Hastings correction for log-normal RW
    log_hast = math.log(theta_new) - math.log(theta_old)
    log_acc = (lp_new - lp_old) + log_hast

    if math.log(rng.random()) < min(0.0, log_acc):
        cl.theta = theta_new


# ============================================================
# --------- PY prior (EPPF) and log-posterior ----------------
# ============================================================

def log_rising_factorial(a: float, m: int) -> float:
    """log (a)_{m} = log Gamma(a+m) - log Gamma(a). m is nonnegative int."""
    if m < 0:
        return float("-inf")
    return math.lgamma(a + m) - math.lgamma(a)

def log_py_eppf_from_sizes(sizes: List[int], gamma: float, delta: float) -> float:
    """
    Pitman–Yor EPPF for an *unordered* partition with block sizes 'sizes'.
    p(π) ∝ [prod_{i=1}^{K-1} (gamma + i*delta)] * prod_j (1-delta)_{n_j-1} / (gamma+1)_{n-1}
    """
    if gamma <= -delta:
        return float("-inf")
    if not (0.0 <= delta < 1.0):
        return float("-inf")
    n = sum(sizes)
    K = len(sizes)
    if n == 0 or K == 0:
        return 0.0

    # numerator: prod_{i=1}^{K-1} (gamma + i*delta)
    log_num_tables = 0.0
    for i in range(1, K):
        term = gamma + i * delta
        if term <= 0:
            return float("-inf")
        log_num_tables += math.log(term)

    # numerator: prod_j (1-delta)_{n_j-1} = prod_j Gamma(n_j - delta) / Gamma(1-delta)
    log_num_sizes = 0.0
    log_gamma_1md = math.lgamma(1.0 - delta)
    for s in sizes:
        if s <= 0:
            return float("-inf")
        log_num_sizes += math.lgamma(s - delta) - log_gamma_1md

    # denominator: (gamma+1)_{n-1} = Gamma(gamma+n) / Gamma(gamma+1)
    log_denom = math.lgamma(gamma + n) - math.lgamma(gamma + 1.0)

    return log_num_tables + log_num_sizes - log_denom

def log_blocks_posterior(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
) -> float:
    """
    log target for blocks (up to constant):
      -theta * S_c(blocks) - n_c * log Z*(blocks, theta) + log PY_EPPF(sizes)
      + (optional) log p(order) where p(order)=1/K!  => -log(K!)
    """
    if not rankings_c:
        return float("-inf")

    sizes = [len(b) for b in blocks]
    K = len(sizes)

    S = total_distance_fast(rankings_c, blocks)
    logZ = log_Z_star_from_sizes(sizes, theta, None)
    logpy = log_py_eppf_from_sizes(sizes, gamma, delta)

    lp = (-theta * S) - (len(rankings_c) * logZ) + logpy -math.lgamma(K + 1) 
    return lp

# ============================================================
# --------- MH move split merge------------------------
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
    """
    Propose either:
      - merge two adjacent blocks, OR
      - split one block into two adjacent blocks
    and accept/reject with MH.

    Split proposal:
      choose a block j with size>=2 uniformly
      choose a split size a in {1,...,s-1} uniformly
      choose subset A of size a uniformly; B = rest
      new blocks: ... [A], [B] ... at position j

    Merge proposal:
      choose boundary between j and j+1 uniformly
      merge them into one block at position j

    This move changes K and sizes locally (good mixing in tie-structure).
    """
    K = len(blocks)
    if K == 0:
        return blocks

    lp_old = log_blocks_posterior(rankings_c, blocks, theta, gamma, delta)

    # Decide merge vs split (but only if feasible)
    splittable = [j for j, b in enumerate(blocks) if len(b) >= 2]
    can_split = len(splittable) > 0
    can_merge = K >= 2

    if not can_split and not can_merge:
        return blocks

    do_merge = (rng.random() < p_merge)
    if do_merge and not can_merge:
        do_merge = False
    if (not do_merge) and not can_split:
        do_merge = True

    if do_merge:
        # ---- MERGE ----
        j = rng.randrange(K - 1)  # boundary between j and j+1
        bL = blocks[j]
        bR = blocks[j + 1]
        prop = [b[:] for b in blocks]
        prop[j] = bL[:] + bR[:]
        del prop[j + 1]

        lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)

        # proposal probs
        # q(old->new): choose merge (p_merge) * choose boundary 1/(K-1)
        log_q_fwd = math.log(p_merge) + math.log(1.0 / (K - 1))

        # q(new->old): must choose split (1-p_merge) * choose splittable block (the merged one)
        # In new state, that merged block has size s = |bL|+|bR|
        # split size a must equal |bL| and subset must equal exactly bL
        K2 = len(prop)
        s = len(prop[j])
        splittable2 = [jj for jj, bb in enumerate(prop) if len(bb) >= 2]
        if len(splittable2) == 0:
            return blocks  # should not happen

        a = len(bL)
        # number of subsets of size a from s
        # IMPORTANT: choose exact subset bL => prob = 1 / C(s,a)
        log_q_backsplit = (
            math.log(1.0 - p_merge)
            + math.log(1.0 / len(splittable2))
            + math.log(1.0 / (s - 1))
            - math.log(math.comb(s, a))
        )
        log_q_bwd = log_q_backsplit

    else:
        # ---- SPLIT ----
        j = rng.choice(splittable)
        block = blocks[j]
        s = len(block)
        # choose split size
        a = rng.randrange(1, s)  # 1..s-1
        # choose subset A of size a uniformly
        A = rng.sample(block, a)
        Aset = set(A)
        B = [x for x in block if x not in Aset]

        prop = [b[:] for b in blocks]
        prop[j] = A[:]
        prop.insert(j + 1, B[:])

        lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)

        # q(old->new): split (1-p_merge) * choose splittable block 1/|splittable|
        #            * choose a 1/(s-1) * choose subset 1/C(s,a)
        log_q_fwd = (
            math.log(1.0 - p_merge)
            + math.log(1.0 / len(splittable))
            + math.log(1.0 / (s - 1))
            - math.log(math.comb(s, a))
        )

        # q(new->old): merge (p_merge) * choose boundary that merges A and B
        K2 = len(prop)
        # boundary count in new state is K2-1, and our target boundary is exactly at j
        log_q_bwd = math.log(p_merge) + math.log(1.0 / (K2 - 1))

    log_acc = (lp_new - lp_old) + (log_q_bwd - log_q_fwd)
    if math.log(rng.random()) < min(0.0, log_acc):
        return prop
    return blocks

# ============================================================
# --------- MH move adjacent item transfer ------------------------
# ============================================================

def mh_adjacent_item_transfer(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    *,
    rng: random.Random,
) -> List[List[int]]:
    """
    Move 1 item from a donor block to an adjacent receiver block.
    Pick a boundary j between blocks j and j+1 uniformly, then a direction.
    Donor must have size >= 2 (to avoid empty block).
    """
    K = len(blocks)
    if K < 2:
        return blocks

    # feasible oriented moves: (j, dir) where dir=+1 means move from j to j+1, dir=-1 from j+1 to j
    moves = []
    for j in range(K - 1):
        if len(blocks[j]) >= 2:
            moves.append((j, +1))
        if len(blocks[j + 1]) >= 2:
            moves.append((j, -1))
    if not moves:
        return blocks

    lp_old = log_blocks_posterior(rankings_c, blocks, theta, gamma, delta)

    j, direction = rng.choice(moves)
    if direction == +1:
        donor, recv = j, j + 1
    else:
        donor, recv = j + 1, j

    x = rng.choice(blocks[donor])

    prop = [b[:] for b in blocks]
    prop[donor].remove(x)
    prop[recv].append(x)

    lp_new = log_blocks_posterior(rankings_c, prop, theta, gamma, delta)

    # proposal prob: choose oriented move uniformly from 'moves', then choose x uniformly from donor
    log_q_fwd = -math.log(len(moves)) - math.log(len(blocks[donor]))

    # reverse: in proposed state, oriented reverse move must be available and choose same x from its new donor
    # donor in reverse is 'recv' (since we moved x into recv)
    blocks2 = prop
    K2 = len(blocks2)
    moves2 = []
    for jj in range(K2 - 1):
        if len(blocks2[jj]) >= 2:
            moves2.append((jj, +1))
        if len(blocks2[jj + 1]) >= 2:
            moves2.append((jj, -1))
    if not moves2:
        return blocks  # should not happen

    # reverse oriented move is same boundary j, opposite direction
    rev_dir = -direction
    if (j, rev_dir) not in moves2:
        return blocks  # should not happen

    rev_donor = recv  # where x currently sits
    log_q_bwd = -math.log(len(moves2)) - math.log(len(blocks2[rev_donor]))

    log_acc = (lp_new - lp_old) + (log_q_bwd - log_q_fwd)
    if math.log(rng.random()) < min(0.0, log_acc):
        return prop
    return blocks


# ============================================================
# --------- Ordering change proposals (swap + long shift) -----
# ============================================================

def _swap_adjacent(blocks: List[List[int]], j: int) -> List[List[int]]:
    """Swap blocks[j] and blocks[j+1]."""
    nb = [b[:] for b in blocks]
    nb[j], nb[j + 1] = nb[j + 1], nb[j]
    return nb

def _move_block_to_index(blocks: List[List[int]], j_from: int, j_to_final: int) -> List[List[int]]:
    """
    Move block at index j_from to final index j_to_final (0..K-1).
    'final index' means its position in the resulting list.
    """
    K = len(blocks)
    if not (0 <= j_from < K and 0 <= j_to_final < K):
        raise ValueError("Invalid indices for move_block.")

    if j_from == j_to_final:
        return [b[:] for b in blocks]

    nb = [b[:] for b in blocks]
    blk = nb.pop(j_from)  # now length K-1
    ins = j_to_final
    if j_to_final > j_from:
        ins = j_to_final - 1
    nb.insert(ins, blk)
    return nb

def _feasible_shift_positions(K: int, j: int, max_step: int) -> List[int]:
    """
    Feasible final indices j_to for moving the block currently at j,
    constrained by max_step in the original index coordinates.
    """
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
    """
    Ordering move for ordered partitions (weak orders):

    With prob p_short:
      - do a short move = n_swap_steps adjacent swaps (random boundaries).
        This is symmetric because each adjacent-swap step is symmetric.

    With prob 1-p_short:
      - do a long move = pick a block j uniformly, move it to a new position p within
        distance <= max_long_step (in index space). This needs a Hastings correction
        because the number of feasible destinations depends on position.

    Defaults scale with K (#blocks):
      - n_swap_steps defaults to max(1, round(sqrt(K)))
      - max_long_step defaults to max(2, round(K/2)) capped at K-1
    """
    K = len(blocks)
    if K <= 1:
        return blocks

    # sensible defaults that scale with K
    if n_swap_steps is None:
        n_swap_steps = max(1, int(round(math.sqrt(K))))
    if max_long_step is None:
        max_long_step = min(K - 1, max(2, int(round(K / 2))))

    lp_old = log_blocks_posterior(
        rankings_c, blocks, theta, gamma, delta
    )

    # -------------------------
    # SHORT: adjacent swap RW
    # -------------------------
    if rng.random() < p_short:
        prop = [b[:] for b in blocks]
        # n_swap_steps times: pick a boundary uniformly and swap
        # Each step is symmetric => entire RW is symmetric (no Hastings term)
        for _ in range(n_swap_steps):
            j = rng.randrange(K - 1)
            prop = _swap_adjacent(prop, j)

        lp_new = log_blocks_posterior(
            rankings_c, prop, theta, gamma, delta
        )

        log_acc = lp_new - lp_old  # symmetric proposal
        if math.log(rng.random()) < min(0.0, log_acc):
            return prop
        return blocks

    # -------------------------
    # LONG: bounded block shift
    # -------------------------
    j_from = rng.randrange(K)
    feasible = _feasible_shift_positions(K, j_from, max_long_step)
    if not feasible:
        return blocks

    j_to = rng.choice(feasible)
    prop = _move_block_to_index(blocks, j_from, j_to)

    lp_new = log_blocks_posterior(
        rankings_c, prop, theta, gamma, delta
    )

    # Proposal:
    # q_fwd = (1-p_short) * (1/K) * (1/|F(j_from)|)
    # Reverse: moved block ends up at index j_to in proposed state.
    # q_bwd = (1-p_short) * (1/K) * (1/|F'(j_to)|)
    # => Hastings factor = |F'(j_to)| / |F(j_from)|
    feasible_rev = _feasible_shift_positions(K, j_to, max_long_step)

    # (Should never be empty since j_from is within max_long_step of j_to by construction,
    #  but keep safe.)
    if not feasible_rev:
        return blocks

    log_q_fwd = math.log(1.0 - p_short) - math.log(K) - math.log(len(feasible))
    log_q_bwd = math.log(1.0 - p_short) - math.log(K) - math.log(len(feasible_rev))

    log_acc = (lp_new - lp_old) + (log_q_bwd - log_q_fwd)
    if math.log(rng.random()) < min(0.0, log_acc):
        return prop
    return blocks



# ============================================================
# --------- Update Cluster Blocks ------------------------
# ============================================================
# ============================================================
# --------- Update Cluster Blocks with Acceptance Tracking ---
# ============================================================

def mh_adjacent_item_transfer_tracked(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    rng: random.Random,
) -> Tuple[List[List[int]], bool]:
    """Wrapper around mh_adjacent_item_transfer that returns (blocks, accepted)."""
    original_blocks = [b[:] for b in blocks]
    new_blocks = mh_adjacent_item_transfer(rankings_c, blocks, theta, gamma, delta, rng=rng)
    accepted = new_blocks != original_blocks
    return new_blocks, accepted

def mh_ordering_swap_or_shift_tracked(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    rng: random.Random,
    p_short: float = 0.75,
    n_swap_steps: Optional[int] = None,
    max_long_step: Optional[int] = None,
) -> Tuple[List[List[int]], bool]:
    """Wrapper around mh_ordering_swap_or_shift that returns (blocks, accepted)."""
    original_blocks = [b[:] for b in blocks]
    new_blocks = mh_ordering_swap_or_shift(
        rankings_c, blocks, theta, gamma, delta, rng=rng, p_short=p_short, n_swap_steps=n_swap_steps, max_long_step=max_long_step
    )
    accepted = new_blocks != original_blocks
    return new_blocks, accepted

def mh_adjacent_split_merge_tracked(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    rng: random.Random,
    p_merge: float = 0.5,
) -> Tuple[List[List[int]], bool]:
    """Wrapper around mh_adjacent_split_merge that returns (blocks, accepted)."""
    original_blocks = [b[:] for b in blocks]
    new_blocks = mh_adjacent_split_merge(rankings_c, blocks, theta, gamma, delta, rng=rng, p_merge=p_merge)
    accepted = new_blocks != original_blocks
    return new_blocks, accepted

def update_cluster_blocks(
    rankings_c: List[List[int]],
    cl: ClusterParams,
    rng: random.Random,
    n_item_moves: int = 1,
    p_gibbs: float = 0.,
    p_transfer: float = 0.4,
    p_swapshift: float = 0.4,
    p_splitmerge: float = 0.20,
    acceptance_stats: Optional["AcceptanceStats"] = None,
) -> None:
    """Update cluster blocks with optional acceptance tracking."""
    if not rankings_c:
        return

    # normalize probs defensively
    s = p_gibbs + p_transfer + p_swapshift + p_splitmerge
    p_gibbs, p_transfer, p_swapshift, p_splitmerge = p_gibbs/s, p_transfer/s, p_swapshift/s, p_splitmerge/s

    for _ in range(n_item_moves):
        u = rng.random()

        if u < p_gibbs:
            # existing collapsed Gibbs reassignment (your current move)
            cl.blocks = gibbs_reassign_one_item(
                rankings=rankings_c,
                blocks=cl.blocks,
                theta=cl.theta,
                gamma=cl.gamma,
                delta=cl.delta,
                rng=rng
            )

        elif u < p_gibbs + p_transfer:
            # MH: move one item across an adjacent boundary
            if acceptance_stats is not None:
                acceptance_stats.transfer_proposals += 1
                cl.blocks, accepted = mh_adjacent_item_transfer_tracked(
                    rankings_c=rankings_c,
                    blocks=cl.blocks,
                    theta=cl.theta,
                    gamma=cl.gamma,
                    delta=cl.delta,
                    rng=rng
                )
                if accepted:
                    acceptance_stats.transfer_accepts += 1
            else:
                cl.blocks = mh_adjacent_item_transfer(
                    rankings_c=rankings_c,
                    blocks=cl.blocks,
                    theta=cl.theta,
                    gamma=cl.gamma,
                    delta=cl.delta,
                    rng=rng
                )
                
        elif u < p_gibbs + p_transfer + p_swapshift:
            if acceptance_stats is not None:
                acceptance_stats.swapshift_proposals += 1
                cl.blocks, accepted = mh_ordering_swap_or_shift_tracked(
                    rankings_c=rankings_c,
                    blocks=cl.blocks,
                    theta=cl.theta,
                    gamma=cl.gamma,
                    delta=cl.delta,
                    rng=rng,
                    p_short=0.75,
                    n_swap_steps=None,
                    max_long_step=None,
                )
                if accepted:
                    acceptance_stats.swapshift_accepts += 1
            else:
                cl.blocks = mh_ordering_swap_or_shift(
                    rankings_c=rankings_c,
                    blocks=cl.blocks,
                    theta=cl.theta,
                    gamma=cl.gamma,
                    delta=cl.delta,
                    rng=rng,
                    p_short=0.75,
                    n_swap_steps=None,
                    max_long_step=None,
                )

        else:
            # MH: split/merge adjacent blocks
            if acceptance_stats is not None:
                acceptance_stats.splitmerge_proposals += 1
                cl.blocks, accepted = mh_adjacent_split_merge_tracked(
                    rankings_c=rankings_c,
                    blocks=cl.blocks,
                    theta=cl.theta,
                    gamma=cl.gamma,
                    delta=cl.delta,
                    rng=rng,
                    p_merge=0.5
                )
                if accepted:
                    acceptance_stats.splitmerge_accepts += 1
            else:
                cl.blocks = mh_adjacent_split_merge(
                    rankings_c=rankings_c,
                    blocks=cl.blocks,
                    theta=cl.theta,
                    gamma=cl.gamma,
                    delta=cl.delta,
                    rng=rng,
                    p_merge=0.5
                )


# ============================================================
# ---------- top-level samplers ------------------------------
# ============================================================

@dataclass
class MCMCSamples:
    z_samples: List[List[int]]
    blocks_samples: List[List[List[List[int]]]]  # [sample][cluster][block][item]
    tau_samples: Optional[List[List[float]]] = None
    theta_samples: Optional[List[List[float]]] = None

@dataclass
class AcceptanceStats:
    """Track acceptance rates for different move types."""
    transfer_proposals: int = 0      # Adjacent item transfer
    transfer_accepts: int = 0
    
    swapshift_proposals: int = 0     # Ordering swap/shift
    swapshift_accepts: int = 0
    
    splitmerge_proposals: int = 0    # Adjacent split/merge
    splitmerge_accepts: int = 0
    
    def transfer_rate(self) -> float:
        return self.transfer_accepts / self.transfer_proposals if self.transfer_proposals > 0 else 0.0
    
    def swapshift_rate(self) -> float:
        return self.swapshift_accepts / self.swapshift_proposals if self.swapshift_proposals > 0 else 0.0
    
    def splitmerge_rate(self) -> float:
        return self.splitmerge_accepts / self.splitmerge_proposals if self.splitmerge_proposals > 0 else 0.0
    
    def summary(self) -> Dict[str, float]:
        """Return summary of acceptance rates."""
        return {
            "transfer": self.transfer_rate(),
            "swapshift": self.swapshift_rate(),
            "splitmerge": self.splitmerge_rate(),
        }

@dataclass
class MCMCResult:
    final_state: "MixtureState"
    samples: Optional[MCMCSamples] = None
    acceptance_stats: Optional[AcceptanceStats] = None


def run_mixture_mcmc(
    rankings: List[List[int]],
    n_iter: int,
    init_clusters: List["ClusterParams"],
    C: int = 1,
    mu: Optional[List[float]] = None,
    seed: int = 123,
    n_item_moves_per_cluster: int = 2,

    # sampling controls (defaults: save the "necessary" samples)
    save_samples: bool = True,
    burn_in: int = 0,
    thin: int = 1,

    # optional extra samples
    save_tau: bool = False,
    save_theta: bool = False,
) -> MCMCResult:
    """
    Run mixture MCMC. Always returns the final state.
    Optionally collects MCMC samples (default: on), with burn-in and thinning.

    "Necessary" samples saved by default when save_samples=True:
      - z_samples
      - blocks_samples

    Optional extras:
      - tau_samples if save_tau=True
      - theta_samples if save_theta=True
    """
    if C <= 0:
        raise ValueError("C must be positive")
    if thin <= 0:
        raise ValueError("thin must be >= 1")
    if burn_in < 0:
        raise ValueError("burn_in must be >= 0")

    if mu is None:
        mu = [1.0] * C
    if len(mu) != C:
        raise ValueError("mu must have length C")
    if len(init_clusters) != C:
        raise ValueError("init_clusters must have length C")

    rng = random.Random(seed)
    N = len(rankings)

    # init
    z = [rng.randrange(C) for _ in range(N)]
    tau = dirichlet_sample(mu, rng)
    state = MixtureState(clusters=init_clusters, z=z, tau=tau)

    # acceptance tracking
    acceptance_stats = AcceptanceStats()

    # storage (optional)
    samples: Optional[MCMCSamples] = None
    if save_samples:
        samples = MCMCSamples(
            z_samples=[],
            blocks_samples=[],
            tau_samples=[] if save_tau else None,
            theta_samples=[] if save_theta else None,
        )

        def snapshot() -> None:
            # copy z
            samples.z_samples.append(state.z[:])
            # deep-copy blocks: clusters -> blocks -> items
            samples.blocks_samples.append([[b[:] for b in cl.blocks] for cl in state.clusters])
            # optional extras
            if save_tau and samples.tau_samples is not None:
                samples.tau_samples.append(state.tau[:])
            if save_theta and samples.theta_samples is not None:
                samples.theta_samples.append([cl.theta for cl in state.clusters])

    # run chain
    for it in range(n_iter):
        update_z(rankings, state, rng)
        update_tau(state, mu, rng)
        for c in range(C):
            Rc = cluster_rankings(rankings, state.z, c)
            update_cluster_blocks(Rc, state.clusters[c], rng, n_item_moves=n_item_moves_per_cluster, acceptance_stats=acceptance_stats)
            update_cluster_theta(Rc, state.clusters[c], rng)

        if save_samples and it >= burn_in and ((it - burn_in) % thin == 0):
            snapshot()

    return MCMCResult(final_state=state, samples=samples, acceptance_stats=acceptance_stats)



# ============================================================
# ---------- Estimate MAP -------------------------------------
# ============================================================

def _canonicalize_blocks(blocks: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    """
    Canonical representation of a weak order (ordered partition):
      - sort items within each block
      - keep block order as given
    Returns a hashable tuple-of-tuples.
    """
    return tuple(tuple(sorted(block)) for block in blocks)

def estimate_z_from_frequency(
    z_samples: List[List[int]],
    *,
    C: int
) -> Dict[str, Any]:
    """
    Returns:
      p_ic: N x C posterior membership probs via frequency
      z_hat: length N hard labels via argmax
    """
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
    """
    Returns (mode_value, mode_prob, mode_count)
    """
    total = sum(counts.values())
    mode_val = max(counts.keys(), key=lambda k: counts[k])
    mode_count = counts[mode_val]
    mode_prob = mode_count / total if total else float("nan")
    return mode_val, mode_prob, mode_count

def summarize_theta(
    theta_samples_c: List[float],
    *,
    ci: float = 0.95,
    map_bins: int = 50
) -> Dict[str, float]:
    """
    Simple posterior summaries for scalar theta:
      mean, median, (ci_lo, ci_hi), and a crude MAP via histogram mode.
    """
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

def estimate_posterior_summaries_simple(
    z_samples: List[List[int]],
    blocks_samples: List[List[List[List[int]]]],  # [t][c] -> blocks
    theta_samples: List[List[float]],             # [t][c] -> theta
    *,
    C: int,
    ci: float = 0.95
) -> Dict[str, Any]:
    """
    Simple frequency-based estimators:
      - soft/hard z from frequencies
      - per-cluster consensus blocks = most frequent sampled weak order
      - per-cluster consensus probability = its frequency
      - theta summarized directly from samples

    No label switching handling by design.
    """
    T = len(z_samples)
    if T == 0:
        raise ValueError("Empty MCMC samples.")
    N = len(z_samples[0])

    # 1) z summaries
    z_out = estimate_z_from_frequency(z_samples, C=C)

    # 2) consensus weak order per cluster = posterior mode
    consensus_blocks = []
    for c in range(C):
        counts: Dict[Tuple[Tuple[int, ...], ...], int] = {}
        for t in range(T):
            key = _canonicalize_blocks(blocks_samples[t][c])
            counts[key] = counts.get(key, 0) + 1

        mode_key, mode_prob, mode_count = _posterior_mode_from_counts(counts)
        # convert back to list-of-lists
        blocks_hat = [list(block) for block in mode_key]

        consensus_blocks.append({
            "cluster": c,
            "blocks_hat": blocks_hat,
            "posterior_prob": mode_prob,
            "count": mode_count,
            "n_unique": len(counts),
        })

    # 3) theta summaries per cluster
    theta_summary = []
    for c in range(C):
        theta_c = [theta_samples[t][c] for t in range(T)]
        theta_summary.append({"cluster": c, **summarize_theta(theta_c, ci=ci)})

    return {
        "N": N,
        "C": C,
        "T": T,
        "labels": z_out,                 # {"p_ic": ..., "z_hat": ...}
        "consensus_blocks": consensus_blocks,
        "theta_summary": theta_summary,
    }
