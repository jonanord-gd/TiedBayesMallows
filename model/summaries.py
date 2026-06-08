"""Helpers for summarizing MCMC output and blocks."""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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


# ─────────────────────────────────────────────────────────────────────────────
# Greedy consensus recovery from posterior block samples
# ─────────────────────────────────────────────────────────────────────────────

def _log_q_binom(n: int, k: int, q: float) -> float:
    """log of the q-binomial coefficient C(n, k)_q.

    Uses the product formula
        C(n,k)_q = ∏_{i=0}^{k-1} (1 - q^{n-i}) / (1 - q^{i+1})

    with k replaced by min(k, n-k) for symmetry and numerical stability.
    Returns 0.0 for boundary cases k=0 or k=n.
    """
    if k == 0 or k == n:
        return 0.0
    k = min(k, n - k)
    # q ≈ 1  (θ ≈ 0): q-binomial → ordinary binomial
    if abs(q - 1.0) < 1e-12:
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    result = 0.0
    for i in range(k):
        # log(1 - q^{n-i}) - log(1 - q^{i+1})
        # Use math.log1p(-x) = log(1-x) for better accuracy when x≈0.
        result += math.log1p(-q ** (n - i)) - math.log1p(-q ** (i + 1))
    return result


def _log_binom(n: int, k: int) -> float:
    """log of ordinary binomial coefficient C(n, k)."""
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def merge_threshold(s_a: int, s_b: int, theta: float) -> float:
    """Threshold on D_cross for merging two adjacent consensus blocks.

    Derived from the likelihood ratio test comparing the Mallows model
    with blocks B_a and B_b kept **separate** (B_a ranked above B_b)
    against the model with them **merged** into one tied block.

    The merge is preferred — and should be performed — when

        D_cross > merge_threshold(s_a, s_b, theta)

    where D_cross is the *total* cross-block Kemeny disagreement summed
    over all s_a * s_b item pairs, averaged over posterior samples
    (i.e., D_cross ∈ [0, s_a * s_b]).

    Formula (from theory)
    ---------------------
        threshold = (1/θ) · log [ C(s_a+s_b, s_a) / C(s_a+s_b, s_a)_{e^{-θ}} ]

    where C(n,k) is the ordinary binomial and C(n,k)_q is the q-binomial
    (= ratio of Mallows normalising constants: merged block Z / separate Z).

    Parameters
    ----------
    s_a, s_b : int  Block sizes (both ≥ 1).
    theta    : float  Concentration parameter (> 0).

    Returns
    -------
    float  The threshold value.  Always ≥ 0.
    """
    if theta <= 0:
        raise ValueError("theta must be strictly positive")
    q = math.exp(-theta)
    n, k = s_a + s_b, min(s_a, s_b)
    log_binom  = _log_binom(n, k)
    log_q_binom = _log_q_binom(n, k, q)
    # log_binom ≥ log_q_binom always (q-binomial ≤ ordinary binomial for q∈(0,1))
    return (log_binom - log_q_binom) / theta


def build_posterior_pairwise_matrix(
    blocks_samples_c: List[List[List[int]]],
    n_items: int,
    *,
    p_tie: float = 0.5,
    thin: int = 1,
) -> np.ndarray:
    """Build an n_items × n_items pairwise preference matrix from posterior.

    P[i, j] = fraction of posterior draws in which item i is in a
    *higher-ranked* (earlier) block than item j.  Ties (same block)
    contribute p_tie to both P[i,j] and P[j,i], so P[i,j] + P[j,i] = 1
    for all i ≠ j.

    Parameters
    ----------
    blocks_samples_c : list of T block structures for ONE cluster.
        Each element is a block partition, e.g. [[0,2],[1],[3,4]].
    n_items : int
    p_tie   : float  Kemeny weight for ties (default 0.5).
    thin    : int    Use every `thin`-th sample (default 1 = use all).

    Returns
    -------
    P : ndarray (n_items, n_items), float64, diagonal = 0.
    """
    samples = blocks_samples_c[::thin]
    T = len(samples)
    P = np.zeros((n_items, n_items), dtype=np.float64)

    for blks in samples:
        # Build block-index vector: bidx[item] = rank of block (0 = highest).
        bidx = np.full(n_items, -1, dtype=np.int32)
        for b_idx, block in enumerate(blks):
            for item in block:
                if 0 <= item < n_items:
                    bidx[item] = b_idx

        # Vectorised pairwise comparison: O(n²) per sample.
        # -1 marks items absent from this cluster's block structure.
        valid = (bidx >= 0)
        b = bidx.astype(np.float64)
        b[~valid] = np.nan   # NaN propagation keeps invalid pairs at 0

        bi = b[:, None]   # (n, 1)
        bj = b[None, :]   # (1, n)

        # i ranked above j  → P[i,j] += 1
        # same block        → P[i,j] += p_tie  (and symmetrically P[j,i])
        i_above = np.nan_to_num(bi < bj, nan=0.0)
        tied    = np.nan_to_num(bi == bj, nan=0.0)

        P += i_above + p_tie * tied

    np.fill_diagonal(P, 0.0)
    P /= T
    return P


def greedy_consensus_recovery(
    blocks_samples_c: List[List[List[int]]],
    theta: float,
    n_items: int,
    *,
    p_tie: float = 0.5,
    thin: int = 1,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """Recover a consensus weak order from posterior block samples.

    Algorithm
    ---------
    1. **Pairwise preference matrix** P[i,j] — fraction of posterior draws
       where item i is in a higher-ranked block than j (ties → p_tie = 0.5).

    2. **Initial strict ordering** — items sorted descending by Borda score
       B[i] = Σ_j P[i,j].  Each item starts as its own singleton block.
       Any cycles present in the Borda order will be resolved by merging.

    3. **Greedy block merging** — repeat until no merge criterion is met:
       a. For each pair of *adjacent* blocks (B_a, B_b) in the current ordering,
          compute the total cross-block Kemeny disagreement::

              D_cross(B_a, B_b) = Σ_{i∈B_a, j∈B_b} P[j, i]

          (sum of preferences for the *lower* block's items over the upper).
       b. Compare D_cross to the theoretical likelihood threshold::

              threshold(s_a, s_b, θ) =
                  (1/θ) · log [ C(s_a+s_b, s_a) / C(s_a+s_b, s_a)_{e^{-θ}} ]

          where C(n,k) is the ordinary binomial and C(n,k)_q the q-binomial.
       c. Merge the adjacent pair with the *highest* D_cross > threshold.
       d. Stop when all adjacent D_cross values are ≤ their thresholds.

    Parameters
    ----------
    blocks_samples_c : list of T block structures for ONE cluster.
    theta     : float  Concentration parameter (posterior mean or MAP).
    n_items   : int    Total number of items.
    p_tie     : float  Kemeny tie penalty (default 0.5).
    thin      : int    Thin factor for pairwise-matrix computation.

    Returns
    -------
    blocks : list of list of int
        Recovered consensus weak order as a block partition (rank 0 = highest).
    info   : dict  Diagnostics:
        ``P``             — (n_items, n_items) pairwise preference matrix.
        ``borda``         — Borda scores per item.
        ``ordered_items`` — Initial Borda ordering.
        ``n_cycles``      — Number of Borda-order cycle violations found.
        ``merge_history`` — List of dicts, one per merge performed.
        ``d_cross_final`` — D_cross values for each adjacent block pair in the
                            final consensus (all ≤ threshold).
        ``threshold_final``— Corresponding thresholds.
    """
    # Step 1: pairwise matrix
    P = build_posterior_pairwise_matrix(
        blocks_samples_c, n_items, p_tie=p_tie, thin=thin
    )

    # Step 2: Borda ordering
    borda = P.sum(axis=1)
    ordered_items: List[int] = list(np.argsort(-borda))

    # Count cycles: pairs (a < b in Borda order) where P[item_a, item_b] < p_tie
    n_cycles = int(sum(
        1
        for a in range(n_items)
        for b in range(a + 1, n_items)
        if P[ordered_items[a], ordered_items[b]] < p_tie
    ))

    # Step 3: start with all singletons
    blocks: List[List[int]] = [[item] for item in ordered_items]
    merge_history: List[Dict[str, Any]] = []

    while True:
        n_blk = len(blocks)
        if n_blk <= 1:
            break

        best_a:   Optional[int]   = None
        best_d:   float           = -1.0
        best_thr: float           = 0.0

        for a in range(n_blk - 1):
            B_a, B_b = blocks[a], blocks[a + 1]
            # Total cross-block disagreement (higher-block b items preferred over a items)
            d = float(sum(P[j, i] for i in B_a for j in B_b))
            thr = merge_threshold(len(B_a), len(B_b), theta)
            if d > thr and d > best_d:
                best_d   = d
                best_a   = a
                best_thr = thr

        if best_a is None:
            break

        a = best_a
        merge_history.append({
            "position":    a,
            "block_a":     list(blocks[a]),
            "block_b":     list(blocks[a + 1]),
            "d_cross":     best_d,
            "threshold":   best_thr,
            "n_blk_before": n_blk,
        })
        blocks = blocks[:a] + [blocks[a] + blocks[a + 1]] + blocks[a + 2:]

    # Final adjacent D_cross values and thresholds
    d_cross_final  = []
    threshold_final = []
    for a in range(len(blocks) - 1):
        B_a, B_b = blocks[a], blocks[a + 1]
        d = float(sum(P[j, i] for i in B_a for j in B_b))
        thr = merge_threshold(len(B_a), len(B_b), theta)
        d_cross_final.append(d)
        threshold_final.append(thr)

    return blocks, {
        "P":               P,
        "borda":           borda,
        "ordered_items":   ordered_items,
        "n_cycles":        n_cycles,
        "merge_history":   merge_history,
        "d_cross_final":   d_cross_final,
        "threshold_final": threshold_final,
    }


def consensus_from_samples(
    samples,
    *,
    p_tie: float = 0.5,
    thin: int = 1,
    theta_method: str = "mean",
    min_cluster_size: int = 1,
) -> Dict[int, Dict[str, Any]]:
    """Recover consensus weak orders for all active clusters from MCMCSamples.

    Convenience wrapper around :func:`greedy_consensus_recovery` that handles
    all bookkeeping automatically:

    * **n_items** is inferred from the block samples.
    * **theta** per cluster is estimated from the posterior theta samples.
    * Block samples are separated by cluster internally.

    Parameters
    ----------
    samples : MCMCSamples
        Output of ``model.run_mcmc(save_samples=True, save_theta=True)``.
    p_tie : float
        Kemeny tie penalty passed to ``greedy_consensus_recovery`` (default 0.5).
    thin : int
        Thinning factor for the pairwise preference matrix (default 1 = use all).
    theta_method : str
        How to estimate θ per cluster from the posterior:

        * ``"mean"``   — posterior mean of θ samples (default; most stable).
        * ``"median"`` — posterior median.
        * ``"map"``    — θ at the single highest-logp draw (mirrors ``find_map``);
                         requires ``samples.logp`` to be saved.
    min_cluster_size : int
        Skip clusters whose size in the final z draw is below this threshold
        (default 1 — include all non-empty clusters).

    Returns
    -------
    dict
        A dict with two kinds of entries:

        * Integer keys (cluster index) → ``{"blocks", "info", "theta",
          "z_map", "membership", "cluster_size"}``.

          - ``"blocks"``       — recovered consensus weak order (list of lists,
                                 rank 0 = highest preference).
          - ``"info"``         — diagnostics from ``greedy_consensus_recovery``.
          - ``"theta"``        — estimated concentration parameter.
          - ``"z_map"``        — boolean array (length N) — True for assessors
                                 whose MAP cluster assignment is this cluster.
          - ``"membership"``   — float array (length N) — posterior probability
                                 each assessor belongs to this cluster.
          - ``"cluster_size"`` — number of assessors in this cluster under MAP.

        * String keys: ``"z_map"`` (int array, length N, MAP cluster per
          assessor) and ``"membership"`` (2-D float array, shape N×C).
    """
    if samples.theta_samples is None:
        raise ValueError(
            "samples.theta_samples is None — re-run MCMC with save_theta=True."
        )
    if not samples.blocks_samples:
        raise ValueError("samples.blocks_samples is empty.")

    T = len(samples.blocks_samples)
    C = len(samples.blocks_samples[0])

    # ── Infer n_items from block samples ─────────────────────────────────────
    max_item = -1
    for t_blocks in samples.blocks_samples:
        for c in range(C):
            for blk in t_blocks[c]:
                for item in blk:
                    if item > max_item:
                        max_item = item
    if max_item < 0:
        raise ValueError("Could not infer n_items: all block samples are empty.")
    n_items = max_item + 1

    # ── Determine active clusters from the final z draw ───────────────────────
    z_final = np.asarray(samples.z_samples[-1], dtype=int)
    cluster_sizes_final = np.bincount(z_final, minlength=C)
    active = [c for c in range(C) if cluster_sizes_final[c] >= min_cluster_size]

    # ── Estimate theta per cluster ────────────────────────────────────────────
    theta_arr = np.asarray(samples.theta_samples)  # (T, C)

    if theta_method == "mean":
        theta_per_cluster = {c: float(theta_arr[:, c].mean()) for c in active}
    elif theta_method == "median":
        theta_per_cluster = {c: float(np.median(theta_arr[:, c])) for c in active}
    elif theta_method == "map":
        if not samples.logp:
            raise ValueError(
                "theta_method='map' requires logp to be saved — "
                "re-run MCMC with save_logp=True."
            )
        best_t = int(np.argmax(np.asarray(samples.logp)))
        theta_per_cluster = {c: float(theta_arr[best_t, c]) for c in active}
    else:
        raise ValueError(
            f"theta_method must be 'mean', 'median', or 'map', got {theta_method!r}"
        )

    # ── Estimate MAP cluster assignment per assessor ──────────────────────────
    z_freq = estimate_z_from_frequency(samples.z_samples, C=C)
    z_map_vec = np.asarray(z_freq["z_hat"], dtype=int)          # (N,)
    membership_mat = np.asarray(z_freq["p_ic"], dtype=float)    # (N, C)

    # ── Run greedy consensus recovery per active cluster ──────────────────────
    results: Dict[Any, Any] = {}
    for c in active:
        bsamp_c = [samples.blocks_samples[t][c] for t in range(T)]
        theta_c = theta_per_cluster[c]

        blocks_hat, info = greedy_consensus_recovery(
            bsamp_c,
            theta=theta_c,
            n_items=n_items,
            p_tie=p_tie,
            thin=thin,
        )
        z_map_c = z_map_vec == c
        results[c] = {
            "blocks": blocks_hat,
            "info": info,
            "theta": theta_c,
            "z_map": z_map_c,
            "membership": membership_mat[:, c],
            "cluster_size": int(z_map_c.sum()),
        }

    # ── Top-level convenience entries ─────────────────────────────────────────
    results["z_map"] = z_map_vec          # shape (N,)  — MAP cluster per assessor
    results["membership"] = membership_mat  # shape (N, C) — full posterior probability matrix

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Medoid consensus from posterior samples
# ─────────────────────────────────────────────────────────────────────────────

def _find_medoid_blocks(
    blocks_samples_c: List[List[List[int]]],
    n_items: int,
    thin: int = 1,
    p_tie: float = 0.5,
) -> Tuple[List[List[int]], float, int]:
    """Find the medoid block ordering from cluster posterior samples.

    The medoid is the actual posterior sample that minimises the total Kemeny
    distance to all other samples in the cluster.  Unlike greedy consensus
    recovery it is guaranteed to be a valid observed weak order.

    Kemeny distance between two *weak* orders for a pair (i, j):

    * **0** — both agree (same strict direction, or both tie the pair).
    * **1** — strict disagreement (one says i ≺ j, the other says j ≺ i).
    * **p_tie** — one strictly orders the pair, the other ties it.

    Algorithm (O(T · n²) time, O(n²) space)
    ----------------------------------------
    1. Build count matrices from the T (thinned) samples:

       * P[i,j] = #{t : i strictly before j}
       * P[j,i] = #{t : j strictly before i}
       * E[i,j] = T − P[i,j] − P[j,i]  (tied)

    2. Score each candidate t over all upper-triangle pairs (i < j)::

         score(t) =  [t: i<j] · (P[j,i] + p_tie·E[i,j])   # t strict, opp. samples disagree
                   + [t: i=j] · p_tie · (P[i,j] + P[j,i])  # t ties, strict samples disagree
                   + [t: i>j] · (P[i,j] + p_tie·E[i,j])    # t strict other way

    3. Return the candidate with the lowest score.

    Returns
    -------
    (medoid_blocks, avg_distance, medoid_index)
    """
    thinned = blocks_samples_c[::thin] if thin > 1 else blocks_samples_c
    T = len(thinned)
    if T == 0:
        raise ValueError("No samples available for this cluster after thinning.")

    # ── Build rank matrix (T × n_items): rank_mat[t, i] = block index of item i ──
    rank_mat = np.zeros((T, n_items), dtype=np.int32)
    for t, blocks in enumerate(thinned):
        for b_idx, block in enumerate(blocks):
            for item in block:
                rank_mat[t, item] = b_idx

    # ── Count matrices (n × n, float64 for p_tie arithmetic) ─────────────────
    # P[i,j]  = #{t : i strictly before j}   (i in earlier block than j)
    # P_T[i,j] = P[j,i]                       (j strictly before i)
    # E[i,j]  = #{t : i and j in the same block (tied)}
    P   = (rank_mat[:, :, None] < rank_mat[:, None, :]).sum(axis=0).astype(np.float64)
    P_T = P.T
    E   = T - P - P_T

    # ── Score each candidate ──────────────────────────────────────────────────
    scores = np.zeros(T, dtype=np.float64)
    for t in range(T):
        r = rank_mat[t]
        before = np.triu(r[:, None] < r[None, :], k=1).astype(np.float64)  # [t: i<j]
        after  = np.triu(r[:, None] > r[None, :], k=1).astype(np.float64)  # [t: i>j]
        tied   = np.triu(r[:, None] == r[None, :], k=1).astype(np.float64) # [t: i=j]

        # Strict disagreement (weight 1)
        strict = (before * P_T).sum() + (after * P).sum()
        # Candidate strictly orders pair, samples tie it (weight p_tie)
        partial_strict_cand = p_tie * ((before + after) * E).sum()
        # Candidate ties pair, samples strictly order it either way (weight p_tie)
        partial_tied_cand   = p_tie * (tied * (P + P_T)).sum()

        scores[t] = strict + partial_strict_cand + partial_tied_cand

    medoid_t = int(np.argmin(scores))
    avg_distance = float(scores[medoid_t]) / T

    return thinned[medoid_t], avg_distance, medoid_t


def medoid_from_samples(
    samples,
    *,
    rankings=None,
    gamma: float = 1.0,
    delta: float = 0.5,
    p_tie: float = 0.5,
    a_theta: float = 1.0,
    b_theta: float = 1e-6,
    thin: int = 1,
    theta_method: str = "mean",
    min_cluster_size: int = 1,
    refine: bool = True,
    max_sweeps: int = 50,
    use_py_prior: bool = False,
    include_order_prior: bool = False,
    verbose: bool = True,
) -> Dict[Any, Any]:
    """Find the medoid weak order for each active cluster, then refine with ICM.

    **Step 1 — Medoid:** finds the actual posterior sample closest (in total
    Kemeny distance) to all other samples in the cluster.  This is guaranteed
    to be a valid observed weak order.

    **Step 2 — ICM refinement** (when ``rankings`` is provided and
    ``refine=True``): runs Iterated Conditional Modes hill-climbing on top of
    the medoid, exactly as ``find_map`` does.  For each item in turn the block
    position is moved to the one that maximises the cluster log-posterior given
    the current MAP cluster assignments (``z_map``).  This can improve beyond
    any visited sample.

    Parameters
    ----------
    samples : MCMCSamples
        Output of ``model.run_mcmc(save_samples=True, save_theta=True)``.
    rankings : array-like, shape (N, n_items), optional
        Original ranking data (position→item format).  Required for ICM
        refinement.  If None, only the medoid is returned.
    gamma : float
        Pitman-Yor discount parameter (default 1.0 = no discount).
    delta : float
        Pitman-Yor strength parameter (default 0.5).
    p_tie : float
        Kemeny tie-vs-strict penalty (default 0.5).  Weight given to a
        disagreement between one strict and one tied ordering of the same
        pair.  Set to 0 to ignore ties entirely, 1 to treat them the same
        as full strict disagreements.
    thin : int
        Thinning factor for the medoid search (default 1 = all samples).
    theta_method : str
        ``"mean"`` | ``"median"`` | ``"map"`` — how to estimate θ per cluster.
    min_cluster_size : int
        Skip clusters smaller than this in the final z draw (default 1).
    refine : bool
        Run ICM sweeps after finding the medoid (default True).
        Ignored when ``rankings`` is None.
    max_sweeps : int
        Maximum ICM sweeps per cluster (default 50).
    use_py_prior : bool
        Include the Pitman-Yor prior in the ICM objective (default False).
    include_order_prior : bool
        Include the uniform order prior in the ICM objective (default False).
    verbose : bool
        Print ICM progress (default True).

    Returns
    -------
    dict
        * Integer keys → ``{"blocks", "theta", "z_map", "membership",
          "cluster_size", "avg_distance", "icm_moves", "icm_sweeps"}``.
        * ``"z_map"``      — int array (N,), MAP cluster per assessor.
        * ``"membership"`` — float array (N × C), posterior probabilities.
    """
    if samples.theta_samples is None:
        raise ValueError(
            "samples.theta_samples is None — re-run MCMC with save_theta=True."
        )
    if not samples.blocks_samples:
        raise ValueError("samples.blocks_samples is empty.")

    T = len(samples.blocks_samples)
    C = len(samples.blocks_samples[0])

    # ── Infer n_items ─────────────────────────────────────────────────────────
    max_item = -1
    for t_blocks in samples.blocks_samples:
        for c in range(C):
            for blk in t_blocks[c]:
                for item in blk:
                    if item > max_item:
                        max_item = item
    if max_item < 0:
        raise ValueError("Could not infer n_items: all block samples are empty.")
    n_items = max_item + 1

    # ── Active clusters from final z draw ─────────────────────────────────────
    z_final = np.asarray(samples.z_samples[-1], dtype=int)
    cluster_sizes_final = np.bincount(z_final, minlength=C)
    active = [c for c in range(C) if cluster_sizes_final[c] >= min_cluster_size]

    # ── Theta estimates ───────────────────────────────────────────────────────
    theta_arr = np.asarray(samples.theta_samples)
    if theta_method == "mean":
        theta_per_cluster = {c: float(theta_arr[:, c].mean()) for c in active}
    elif theta_method == "median":
        theta_per_cluster = {c: float(np.median(theta_arr[:, c])) for c in active}
    elif theta_method == "map":
        if not samples.logp:
            raise ValueError(
                "theta_method='map' requires logp — re-run MCMC with save_logp=True."
            )
        best_t = int(np.argmax(np.asarray(samples.logp)))
        theta_per_cluster = {c: float(theta_arr[best_t, c]) for c in active}
    elif theta_method == "conditional_map":
        theta_per_cluster = {c: float(theta_arr[:, c].mean()) for c in active}
    else:
        raise ValueError(f"theta_method must be 'mean', 'median', or 'map', got {theta_method!r}")

    # ── MAP cluster assignments and membership matrix ─────────────────────────
    z_freq = estimate_z_from_frequency(samples.z_samples, C=C)
    z_map_vec = np.asarray(z_freq["z_hat"], dtype=int)
    membership_mat = np.asarray(z_freq["p_ic"], dtype=float)

    # ── Find medoid per cluster ───────────────────────────────────────────────
    # Optionally build U_all for ICM refinement
    do_refine = refine and rankings is not None
    if do_refine or theta_method == "conditional_map":
        R = np.asarray(rankings, dtype=np.intp)   # (N, n_items) position→item
    if do_refine:
        from .moves import compute_U_all, icm_sweep_cluster
        U_all = compute_U_all(R.tolist(), n_items)  # (N, n_pairs) int8

    results: Dict[Any, Any] = {}
    for c in active:
        bsamp_c = [samples.blocks_samples[t][c] for t in range(T)]
        medoid_blocks, avg_dist, _ = _find_medoid_blocks(bsamp_c, n_items, thin=thin, p_tie=p_tie)
        z_map_c = z_map_vec == c
        theta_c = theta_per_cluster[c]

        icm_moves, icm_sweeps = 0, 0
        final_blocks = medoid_blocks

        if do_refine:
            N_c = int(z_map_c.sum())
            if N_c > 0:
                H_c = U_all[z_map_c].sum(axis=0).astype(np.float64)
                final_blocks, icm_moves, icm_sweeps = icm_sweep_cluster(
                    blocks=medoid_blocks,
                    theta=theta_c,
                    gamma=gamma,
                    delta=delta,
                    H=H_c,
                    N=N_c,
                    n=n_items,
                    max_sweeps=max_sweeps,
                    use_py_prior=use_py_prior,
                    include_uniform_order_prior=include_order_prior,
                )
                if verbose:
                    print(f"  Cluster {c}: {icm_moves} ICM moves in {icm_sweeps} sweeps, "
                          f"K={len(final_blocks)} blocks")

        if theta_method == "conditional_map":
            from .theta_conditional_map import theta_conditional_map_for_cluster
            N_c = int(z_map_c.sum())
            if N_c > 0:
                R_c = R[z_map_c]  # (N_c, n_items) position->item
                theta_c, _Dc = theta_conditional_map_for_cluster(
                    R_c, final_blocks, n_items,
                    a_theta=a_theta, b_theta=b_theta, theta_init=theta_c,
                )

        results[c] = {
            "blocks":        final_blocks,
            "theta":         theta_c,
            "z_map":         z_map_c,
            "membership":    membership_mat[:, c],
            "cluster_size":  int(z_map_c.sum()),
            "avg_distance":  avg_dist,
            "icm_moves":     icm_moves,
            "icm_sweeps":    icm_sweeps,
        }

    results["z_map"]       = z_map_vec
    results["membership"]  = membership_mat
    return results
