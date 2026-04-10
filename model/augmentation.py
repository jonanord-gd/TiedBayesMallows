"""Partial ranking augmentation via Metropolis-Hastings data augmentation.

When rankings have missing items (not all n items are ranked by every
assessor), this module provides:

1. Detection of which assessors have incomplete data.
2. Initial random completion of partial rankings.
3. MH proposals that swap positions of two missing items within an
   assessor's ranking, accepting/rejecting based on the Mallows likelihood
   under the assessor's current cluster.
4. Efficient incremental update of U_all rows for affected assessors.

Missing data convention
-----------------------
Rankings are in position→item format: ``ranking[pos] = item``, where items
are ``0..n-1``.  A **partial ranking** lists only the items that were
actually observed, in order.  Its length is ``< n``.  The "missing" items
are those in ``set(range(n)) - set(ranking)``.

At initialisation the partial rankings are padded to length ``n`` by
appending the missing items in a random order.  The augmentation MH step
then explores alternative completions.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np

from .moves import _pair_index


# ── Data structures ───────────────────────────────────────────


@dataclass
class PartialRankingInfo:
    """Pre-computed info about which assessors have missing data."""

    has_missing: bool  # True if ANY ranking is partial
    n_partial: int     # number of assessors with missing items

    # Per-assessor info (length N; entries are empty for complete rankings)
    missing_items: List[List[int]]       # missing_items[i] = list of unranked items
    observed_length: List[int]           # how many items were actually observed
    partial_mask: np.ndarray             # bool (N,): True if ranking i is partial


def detect_missing(rankings: List[List[int]], n: int) -> PartialRankingInfo:
    """Scan rankings and identify partial ones.

    Parameters
    ----------
    rankings : list of lists
        Each ranking is position→item.  Complete rankings have length ``n``
        and contain each of ``0..n-1`` exactly once.  Partial rankings are
        shorter (only observed items, in order).
    n : int
        Total number of items.

    Returns
    -------
    PartialRankingInfo
    """
    N = len(rankings)
    missing_items: List[List[int]] = []
    observed_length: List[int] = []
    partial_mask = np.zeros(N, dtype=bool)

    for i, r in enumerate(rankings):
        obs_len = len(r)
        observed_length.append(obs_len)
        if obs_len < n:
            present = set(r)
            miss = sorted(set(range(n)) - present)
            missing_items.append(miss)
            partial_mask[i] = True
        else:
            missing_items.append([])

    n_partial = int(partial_mask.sum())
    return PartialRankingInfo(
        has_missing=(n_partial > 0),
        n_partial=n_partial,
        missing_items=missing_items,
        observed_length=observed_length,
        partial_mask=partial_mask,
    )


def complete_rankings(
    rankings: List[List[int]],
    info: PartialRankingInfo,
    n: int,
    rng: random.Random,
) -> List[List[int]]:
    """Pad partial rankings to length n by appending missing items in random order.

    Returns a new list of rankings (complete copies).  Rankings that are
    already complete are copied as-is.
    """
    completed = []
    for i, r in enumerate(rankings):
        if info.partial_mask[i]:
            miss = info.missing_items[i][:]
            rng.shuffle(miss)
            completed.append(list(r) + miss)
        else:
            completed.append(list(r))
    return completed


# ── U_all row update ──────────────────────────────────────────


def update_U_rows(
    U_all: np.ndarray,
    rankings: List[List[int]],
    assessor_indices: np.ndarray,
    n: int,
) -> None:
    """Recompute U_all rows for a subset of assessors in-place.

    Parameters
    ----------
    U_all : ndarray (N, n_pairs), float32
        The full pairwise preference matrix.  Modified **in-place**.
    rankings : list of lists
        The current (augmented) complete rankings.
    assessor_indices : 1-D int array
        Indices of assessors whose rows need updating.
    n : int
        Number of items.
    """
    if len(assessor_indices) == 0:
        return
    a_idx, b_idx = np.triu_indices(n, k=1)
    for i in assessor_indices:
        r = rankings[i]
        inv_r = np.empty(n, dtype=np.intp)
        for pos, item in enumerate(r):
            inv_r[item] = pos
        U_all[i] = (inv_r[a_idx] > inv_r[b_idx]).astype(np.float32)


# ── MH augmentation move ─────────────────────────────────────


def _swap_two_missing(
    ranking: List[int],
    missing_items: List[int],
    observed_length: int,
    n: int,
    rng: random.Random,
) -> Tuple[List[int], int, int]:
    """Propose a new completion by swapping two missing items' positions.

    Only positions ``observed_length..n-1`` (the "augmented tail") are
    touched.  The observed prefix is never modified.

    Returns ``(proposed_ranking, pos1, pos2)`` where pos1, pos2 are the
    swapped positions in the ranking.
    """
    n_miss = n - observed_length
    if n_miss < 2:
        # Only one missing item — no swap possible, return unchanged
        return ranking[:], -1, -1

    # Pick two distinct positions in the augmented tail
    tail_start = observed_length
    p1 = rng.randint(tail_start, n - 1)
    p2 = rng.randint(tail_start, n - 2)
    if p2 >= p1:
        p2 += 1

    proposed = ranking[:]
    proposed[p1], proposed[p2] = proposed[p2], proposed[p1]
    return proposed, p1, p2


def _log_mallows_kernel(
    ranking: List[int],
    block_idx: List[int],
    K: int,
    Tm: int,
    theta: float,
    tie_penalty: float,
) -> float:
    """Compute -theta * d(ranking, blocks) for a single ranking.

    Uses a Fenwick tree for O(n log K) inversion counting.
    """
    from .distance import cross_block_disagreements_fast
    inv = cross_block_disagreements_fast(ranking, block_idx, K)
    return -theta * (inv + tie_penalty * Tm)


def _delta_inversions_swap(
    ranking: List[int],
    p1: int,
    p2: int,
    block_idx: List[int],
) -> int:
    """Compute the change in cross-block disagreements when swapping positions p1 and p2.

    Only pairs involving the two swapped items and items between them can
    change.  Cost is O(|p2 - p1|) instead of O(n log K) for full Fenwick.

    Parameters
    ----------
    ranking : list of int
        Current (complete) ranking, position→item.
    p1, p2 : int
        Positions to swap (p1 < p2 assumed; caller should ensure this).
    block_idx : list of int
        Item → block index mapping.

    Returns
    -------
    int
        delta such that ``new_inversions = old_inversions + delta``.
    """
    if p1 > p2:
        p1, p2 = p2, p1

    a = ranking[p1]  # item currently at p1 (moves to p2)
    b = ranking[p2]  # item currently at p2 (moves to p1)
    ba = block_idx[a]
    bb = block_idx[b]

    delta = 0

    # Pair (a, b): currently a at p1 < p2, so a is "before" b.
    # After swap, b is at p1 and a is at p2, so b is "before" a.
    # A cross-block disagreement exists when the earlier-positioned item
    # is in a higher block.  Check if the flip changes the disagreement count.
    if ba != bb:
        # Before: a before b → disagreement if ba > bb
        # After:  b before a → disagreement if bb > ba
        # Net change: +1 if bb > ba, -1 if ba > bb  → sign(ba - bb)
        delta += 1 if bb > ba else -1

    # Items between p1 and p2: each has its relative ordering flipped
    # with both a and b.
    for p in range(p1 + 1, p2):
        c = ranking[p]
        bc = block_idx[c]

        # Pair (a, c): before swap a at p1 < p (a before c).
        # After swap a at p2 > p (a after c). Relative order flips.
        if ba != bc:
            # Before: a before c → disagree if ba > bc (+1 disagreement)
            # After:  c before a → disagree if bc > ba (+1 disagreement)
            if ba > bc:
                delta -= 1  # was disagreement, now concordant
            else:
                delta += 1  # was concordant, now disagreement

        # Pair (b, c): before swap b at p2 > p (b after c).
        # After swap b at p1 < p (b before c). Relative order flips.
        if bb != bc:
            # Before: c before b → disagree if bc > bb
            # After:  b before c → disagree if bb > bc
            if bc > bb:
                delta -= 1  # was disagreement, now concordant
            else:
                delta += 1  # was concordant, now disagreement

    return delta


def augmentation_mh_step(
    rankings: List[List[int]],
    info: PartialRankingInfo,
    z: np.ndarray,
    clusters,  # List[ClusterParams]
    cache,     # List[_ClusterCache]
    tie_penalty: float,
    rng: random.Random,
    n: int,
    method: str = "incremental",
) -> Tuple[int, int]:
    """One sweep of MH augmentation over all partial assessors.

    For each assessor with missing data, propose swapping two missing
    items and accept/reject based on the Mallows likelihood under the
    assessor's current cluster.

    Modifies ``rankings`` in-place for accepted proposals.

    Parameters
    ----------
    method : str, default="incremental"
        Distance computation method for the MH acceptance step:

        ``"incremental"``
            Compute only the *change* in cross-block disagreements from the
            swap.  Cost is O(|p2 − p1|) per proposal — proportional to the
            gap between the two swapped positions, not the full ranking
            length.  Best when ``n`` is large relative to missingness.

        ``"fenwick"``
            Recompute the full distance for both the current and proposed
            rankings using a Fenwick tree.  Cost is O(n log K) per proposal.
            Simpler but heavier when ``n`` is large.

    Returns ``(n_proposals, n_accepts)``.
    """
    partial_indices = np.where(info.partial_mask)[0]
    n_proposals = 0
    n_accepts = 0

    use_incremental = method == "incremental"

    for i in partial_indices:
        obs_len = info.observed_length[i]
        n_miss = n - obs_len
        if n_miss < 2:
            continue

        c = int(z[i])
        cl = clusters[c]
        cc = cache[c]

        # Propose swap
        proposed, p1, p2 = _swap_two_missing(
            rankings[i], info.missing_items[i], obs_len, n, rng
        )
        if p1 < 0:
            continue

        n_proposals += 1

        if use_incremental:
            # O(|p2 - p1|): compute delta in disagreements
            delta_inv = _delta_inversions_swap(rankings[i], p1, p2, cc.block_idx)
            # log_acc = -theta * delta_inv  (Tm cancels out)
            log_acc = -cl.theta * delta_inv
        else:
            # O(n log K): full Fenwick for both
            log_cur = _log_mallows_kernel(
                rankings[i], cc.block_idx, cc.K, cc.Tm, cl.theta, tie_penalty
            )
            log_prop = _log_mallows_kernel(
                proposed, cc.block_idx, cc.K, cc.Tm, cl.theta, tie_penalty
            )
            log_acc = log_prop - log_cur

        # Symmetric proposal → accept with prob min(1, exp(log_acc))
        if math.log(rng.random()) < min(0.0, log_acc):
            rankings[i] = proposed
            n_accepts += 1

    return n_proposals, n_accepts
