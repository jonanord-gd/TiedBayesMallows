"""Identifiability criterion for TiM3 solutions (Section 4.7).

Implements D_min = N · min_{c≠c'} log(1/β_{cc'}) and compares it to log(C),
using the high-θ asymptotic formula from Theorem 4.3 / eq. (51):

    log(1/β_{cc'}) ≈  |D_{cc'}|/2 · θ̄_{cc'}
                    + log(√(F(s^c) · F(s^{c'})) / e(≺_{cc'}))

where  F(s^c) = ∏_k s_k!  and  e(≺_{cc'}) is the number of linear extensions
of the combined partial order induced by both consensus weak orders.

Two regimes
-----------
|D_{cc'}| > 0 :  the combined partial order contains a cycle (the consensuses
    strictly disagree on at least one item pair), so e(≺) = 0 and log(1/β)
    grows linearly with θ.  The pair is asymptotically distinguishable; we
    return the dominant term  |D|/2 · θ̄  +  log √(F^c · F^{c'}).

|D_{cc'}| = 0 :  the consensuses agree on all strict between-block orderings
    and differ only in tie structure.  The combined partial order is consistent
    (no cycle) and e(≺) > 0.  We estimate  log e(≺_{cc'})  via Sequential
    Importance Sampling (SIS) — the same item-by-item sequential pattern used
    in `cross_block_disagreements_fast` (Fenwick tree, distance.py) and the
    candidate-building loop in moves.py.

SIS for linear extensions  (Karzanov & Khachiyan 1991)
------------------------------------------------------
At each step, the *sources* of the remaining partial order (items with all
predecessors already placed) are identified.  One source is chosen uniformly
at random and the sample weight is the product of source-set sizes:

    e(≺) = E_q[ ∏_k s_k ]

where the expectation is under the greedy-uniform proposal q.  Numerically:

    log ê(≺) = logsumexp(log_weights) - log(M)

using the `logsumexp` already available in utils.py.  The estimator is
unbiased and the variance decreases with M and with the tightness of ≺.

Caching
-------
In the posterior-averaged version (eq. 68) the block structure changes
infrequently relative to θ updates, so many MCMC iterations share an
identical block pair.  A lightweight dict cache keyed by canonical block
tuples avoids redundant SIS runs.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .blocks import blocks_to_block_index
from .summaries import _canonicalize_blocks
from .utils import logsumexp


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Pairwise disagreement count
# ─────────────────────────────────────────────────────────────────────────────

def _count_disagreeing_pairs(bidx_c: List[int], bidx_cp: List[int], n: int) -> int:
    """Count item pairs with strictly opposite block orderings in ρ_c vs ρ_{c'}.

    Pair (i, j) disagrees iff
        block_c[i]  < block_c[j]   AND   block_cp[i] > block_cp[j]
    or vice-versa.  Pairs tied in *either* consensus are not strict
    disagreements and are excluded.

    O(n²); n is typically small (≤ 30) so no vectorisation is needed.
    """
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            ci, cj   = bidx_c[i],  bidx_c[j]
            cpi, cpj = bidx_cp[i], bidx_cp[j]
            if ci == cj or cpi == cpj:
                continue        # tied in at least one consensus — not a strict disagreement
            if (ci < cj) != (cpi < cpj):
                count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Combined partial order  (transitive closure)
# ─────────────────────────────────────────────────────────────────────────────

def _build_reachability(
    bidx_c: List[int], bidx_cp: List[int], n: int
) -> Tuple[bool, np.ndarray]:
    """Transitive closure of the combined partial order ≺_{cc'}.

    ≺_{cc'} is the transitive closure of the union of strict pairwise
    constraints from ρ_c and ρ_{c'}: item i must come before j if either
    consensus places i in an *earlier* block than j.

    Returns
    -------
    has_cycle : bool
        True iff the combined order is inconsistent (always the case when
        |D_{cc'}| > 0; in that case e(≺) = 0).
    reach : bool ndarray, shape (n, n)
        reach[i, j] = True  iff  i ≺ j  in the transitive closure.

    Implementation: Floyd-Warshall on a boolean matrix.  O(n³) but n is
    small, and the numpy bitwise ops make each k-iteration a single call.
    """
    bc  = np.asarray(bidx_c,  dtype=np.int32)
    bcp = np.asarray(bidx_cp, dtype=np.int32)

    # Direct edges from each consensus: i → j  iff  bc[i] < bc[j]  (strictly)
    reach = (bc[:, None] < bc[None, :]) | (bcp[:, None] < bcp[None, :])
    np.fill_diagonal(reach, False)

    # Floyd-Warshall transitive closure
    for k in range(n):
        reach |= reach[:, k : k + 1] & reach[k : k + 1, :]

    # Cycle check: a cycle exists iff  ∃ i,j : i ≺ j  AND  j ≺ i
    # (the diagonal stays False by construction but off-diagonal symmetry reveals cycles)
    has_cycle = bool(np.any(reach & reach.T))
    return has_cycle, reach


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SIS estimator for log e(≺)
# ─────────────────────────────────────────────────────────────────────────────

def _sis_log_linear_extensions(
    reach: np.ndarray,
    n: int,
    n_samples: int,
    rng: random.Random,
) -> float:
    """Estimate  log e(≺)  via Sequential Importance Sampling.

    Algorithm
    ---------
    For each sample:
      1. Initialise in_deg[j] = number of items that must precede j  (= reach[:, j].sum()).
      2. Sources = {j : in_deg[j] == 0}.
      3. Repeat n times:
           a. Record s = |sources|;  accumulate  log_w += log(s).
           b. Choose one source uniformly at random.
           c. Remove it; decrement in_deg for its successors (reach[chosen, j]).
      4. log_weight = log_w  for this sample.

    E[exp(log_weight)] = e(≺)  (unbiased estimator of the count), so

        log ê(≺) = logsumexp(log_weights) - log(M).

    Connection to existing code
    ---------------------------
    The sequential item-processing loop mirrors `cross_block_disagreements_fast`
    in distance.py (which sweeps items left-to-right, updating a Fenwick tree).
    The logsumexp aggregation reuses the utility already in utils.py.

    Complexity: O(M · n²) — each of M samples processes n steps each costing
    O(n) for the source scan and successor update.  For n ≤ 30, M = 500
    requires ≈ 450 000 elementary operations; negligible.
    """
    # Predecessor counts — fixed for all samples
    base_in_degrees: List[int] = reach.sum(axis=0).tolist()

    # Precompute successor lists from reach for O(1) lookup per step
    # reach[i, j] == True  →  i is a direct or transitive predecessor of j
    # When i is placed we need to decrement all j still remaining where reach[i,j].
    # Storing as lists avoids repeated numpy indexing inside the hot loop.
    successors: List[List[int]] = [
        [j for j in range(n) if reach[i, j]] for i in range(n)
    ]

    log_weights: List[float] = []

    for _ in range(n_samples):
        in_deg = list(base_in_degrees)          # O(n) copy per sample
        remaining_set = set(range(n))
        log_w = 0.0
        valid = True

        for _step in range(n):
            # Identify sources (items with no remaining predecessors)
            sources = [i for i in remaining_set if in_deg[i] == 0]
            s = len(sources)
            if s == 0:                          # should not happen when |D| = 0
                valid = False
                break
            log_w += math.log(s)

            # Choose a source uniformly at random and "place" it
            chosen = rng.choice(sources)
            remaining_set.discard(chosen)

            # Decrement in-degrees for chosen's successors still remaining
            for j in successors[chosen]:
                if j in remaining_set:
                    in_deg[j] -= 1

        log_weights.append(log_w if valid else float("-inf"))

    return logsumexp(log_weights) - math.log(n_samples)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  log(1/β) for a single cluster pair
# ─────────────────────────────────────────────────────────────────────────────

def _log_sqrt_F(blocks: List[List[int]]) -> float:
    """log √(F(s)) = ½ · Σ_k log(s_k!)."""
    return 0.5 * sum(math.lgamma(len(b) + 1) for b in blocks)


def log_one_over_beta_pair(
    blocks_c:  List[List[int]],
    theta_c:   float,
    blocks_cp: List[List[int]],
    theta_cp:  float,
    n: int,
    *,
    n_samples: int = 500,
    rng: Optional[random.Random] = None,
) -> float:
    """Estimate  log(1/β_{cc'})  from the high-θ asymptotic formula (eq. 51).

    Parameters
    ----------
    blocks_c, blocks_cp : list of list of int
        Block structures for the two consensus weak orders.
    theta_c, theta_cp : float
        Precision parameters.
    n : int
        Total number of items.
    n_samples : int
        SIS draws for estimating e(≺) when |D_{cc'}| = 0.
    rng : random.Random, optional
        For reproducibility.

    Returns
    -------
    float : estimated  log(1/β_{cc'}).
        +inf is returned for pairs that are perfectly distinguishable
        (e.g. identical blocks with infinite consensus).
    """
    if rng is None:
        rng = random.Random()

    bidx_c  = blocks_to_block_index(blocks_c,  n)
    bidx_cp = blocks_to_block_index(blocks_cp, n)

    D = _count_disagreeing_pairs(bidx_c, bidx_cp, n)
    theta_bar    = (theta_c + theta_cp) / 2.0
    log_sqrt_Fcc = _log_sqrt_F(blocks_c) + _log_sqrt_F(blocks_cp)  # log √(F^c · F^{c'})

    if D > 0:
        # e(≺) = 0 because the combined partial order has cycles; log(1/β) grows
        # linearly in θ.  The full formula is D/2·θ̄ + log√(F^c·F^{c'}) - log e(≺),
        # but log e(≺) = -∞ so the normalisation correction is +∞.  We return only
        # the θ-dependent leading term D/2·θ̄; the log_sqrt_Fcc constant is part of
        # the normalisation that diverges and must not be added in isolation.
        return D / 2.0 * theta_bar

    # |D| = 0 — combined partial order is consistent; estimate e(≺) via SIS.
    has_cycle, reach = _build_reachability(bidx_c, bidx_cp, n)
    if has_cycle:
        # Guard for degenerate inputs (should not occur when D == 0).
        return float("inf")

    log_e = _sis_log_linear_extensions(reach, n, n_samples, rng)
    return log_sqrt_Fcc - log_e


# ─────────────────────────────────────────────────────────────────────────────
# 5a.  Plug-in criterion  D̂_min  (eq. 67)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dmin_plugin(
    map_result: Dict,
    N: int,
    n: int,
    *,
    n_samples: int = 500,
    rng: Optional[random.Random] = None,
) -> Dict:
    """Plug-in criterion  D̂_min = N · min_{c≠c'} log(1/β̂_{cc'})  (eq. 67).

    Uses the posterior-mode block and θ estimates from ``find_map()``.

    Parameters
    ----------
    map_result : dict
        Output of ``MixtureRankingModel.find_map()``.  Must contain
        ``consensus_blocks`` and ``theta_summary`` (or theta defaults to 1.0).
    N : int
        Number of assessors.
    n : int
        Number of items.
    n_samples : int
        SIS draws per pair when |D_{cc'}| = 0.
    rng : random.Random, optional

    Returns
    -------
    dict with keys:
        dmin          float   D̂_min value
        log_C         float   log(C) threshold
        recoverable   bool    True iff D̂_min > log(C)
        bottleneck    tuple   (c*, c'*) — hardest cluster pair
        pair_details  list    per-pair breakdown
    """
    if rng is None:
        rng = random.Random(0)

    C          = map_result["C"]
    consensus  = map_result["consensus_blocks"]
    theta_info = map_result.get("theta_summary")

    blocks_list = [consensus[c]["blocks_hat"] for c in range(C)]
    theta_list  = [
        theta_info[c]["map"] if theta_info else 1.0
        for c in range(C)
    ]

    log_C = math.log(C)
    pair_details: List[Dict] = []
    min_val = float("inf")
    bottleneck = (0, 1)

    for c in range(C):
        for cp in range(c + 1, C):
            bidx_c  = blocks_to_block_index(blocks_list[c],  n)
            bidx_cp = blocks_to_block_index(blocks_list[cp], n)
            D = _count_disagreeing_pairs(bidx_c, bidx_cp, n)

            val = log_one_over_beta_pair(
                blocks_list[c],  theta_list[c],
                blocks_list[cp], theta_list[cp],
                n, n_samples=n_samples, rng=rng,
            )
            pair_details.append({
                "c": c, "cp": cp,
                "D": D,
                "log_inv_beta": val,
                "dmin_pair": N * val,
            })
            if val < min_val:
                min_val    = val
                bottleneck = (c, cp)

    dmin = N * min_val
    return {
        "dmin":         dmin,
        "log_C":        log_C,
        "recoverable":  dmin > log_C,
        "bottleneck":   bottleneck,
        "pair_details": pair_details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5b.  Posterior-averaged criterion  D̃_min  (eq. 68)
# ─────────────────────────────────────────────────────────────────────────────

def compute_dmin_posterior_averaged(
    samples,                            # MCMCSamples
    N: int,
    n: int,
    C: int,
    *,
    n_samples_sis: int = 200,
    rng: Optional[random.Random] = None,
    max_mcmc_iters: Optional[int] = None,
) -> Dict:
    """Posterior-averaged criterion  D̃_min = N · min_{c≠c'} E_post[log(1/β_{cc'})]  (eq. 68).

    Averages log(1/β_{cc'}) over post-burn-in MCMC draws.  A dict cache keyed
    by sorted canonical block pairs avoids re-running SIS for block structures
    that repeat across iterations — common because blocks change much less
    frequently than θ does.

    Parameters
    ----------
    samples : MCMCSamples
        Must have ``blocks_samples`` and ``theta_samples``.
    N : int
        Number of assessors.
    n : int
        Number of items.
    C : int
        Number of clusters.
    n_samples_sis : int
        SIS draws per *unique* block pair when |D_{cc'}| = 0.
    rng : random.Random, optional
    max_mcmc_iters : int, optional
        If given, use only the last ``max_mcmc_iters`` stored samples.

    Returns
    -------
    dict with keys:
        dmin                    float
        log_C                   float
        recoverable             bool
        bottleneck              tuple  (c*, c'*)
        pair_details            list
        n_unique_block_pairs    int    (cache size; indicates block mixing)
    """
    if rng is None:
        rng = random.Random(0)

    blocks_samp = samples.blocks_samples
    theta_samp  = samples.theta_samples
    T           = len(blocks_samp)

    start = max(0, T - max_mcmc_iters) if max_mcmc_iters is not None else 0

    # Per-pair running sums for the mean
    pair_keys = [(c, cp) for c in range(C) for cp in range(c + 1, C)]
    pair_accum: Dict[Tuple[int, int], List[float]] = {k: [] for k in pair_keys}

    # SIS cache: sorted canonical block pair → log e(≺) estimate
    sis_cache: Dict[Tuple, float] = {}

    for t in range(start, T):
        blks_t  = blocks_samp[t]                           # list[C] of block structures
        theta_t = theta_samp[t] if theta_samp else [1.0] * C

        for c, cp in pair_keys:
            bidx_c  = blocks_to_block_index(blks_t[c],  n)
            bidx_cp = blocks_to_block_index(blks_t[cp], n)

            D = _count_disagreeing_pairs(bidx_c, bidx_cp, n)
            log_sqrt_Fcc = _log_sqrt_F(blks_t[c]) + _log_sqrt_F(blks_t[cp])
            theta_bar    = (theta_t[c] + theta_t[cp]) / 2.0

            if D > 0:
                val = D / 2.0 * theta_bar + log_sqrt_Fcc
            else:
                # Build a canonical, order-independent cache key
                key_c  = _canonicalize_blocks(blks_t[c])
                key_cp = _canonicalize_blocks(blks_t[cp])
                cache_key = tuple(sorted([key_c, key_cp]))

                if cache_key not in sis_cache:
                    _, reach = _build_reachability(bidx_c, bidx_cp, n)
                    sis_cache[cache_key] = _sis_log_linear_extensions(
                        reach, n, n_samples_sis, rng
                    )
                log_e = sis_cache[cache_key]
                val   = log_sqrt_Fcc - log_e

            pair_accum[(c, cp)].append(val)

    log_C = math.log(C)
    pair_details: List[Dict] = []
    min_mean = float("inf")
    bottleneck = pair_keys[0]

    for c, cp in pair_keys:
        vals = pair_accum[(c, cp)]
        mean_val = sum(vals) / len(vals) if vals else float("inf")
        pair_details.append({
            "c": c, "cp": cp,
            "mean_log_inv_beta": mean_val,
            "dmin_pair": N * mean_val,
        })
        if mean_val < min_mean:
            min_mean   = mean_val
            bottleneck = (c, cp)

    dmin = N * min_mean
    return {
        "dmin":                  dmin,
        "log_C":                 log_C,
        "recoverable":           dmin > log_C,
        "bottleneck":            bottleneck,
        "pair_details":          pair_details,
        "n_unique_block_pairs":  len(sis_cache),
    }
