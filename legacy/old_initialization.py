"""Legacy Borda-threshold block initialisation helpers, moved from model/initialization.py.

Functions
---------
_make_tied_ranking_with_percentile  - Random tied-block splitting of a ranked list
init_blocks_borda_threshold         - Generate per-cluster initial blocks via Borda consensus
"""

import random
from typing import List, Optional

try:
    from model.dataclasses import ClusterParams
except ImportError:
    from TiedBayesMallows.model.dataclasses import ClusterParams


def _make_tied_ranking_with_percentile(
    ranking: List[int],
    percentile: float,
    rng: random.Random,
) -> List[List[int]]:
    """Create tied blocks based on a random number of split points.

    Parameters
    ----------
    ranking : list
        Items in ranked order.
    percentile : float
        Expected fraction of items that get separate blocks (0 to 1).
    rng : random.Random

    Returns
    -------
    blocks : list of lists
    """
    n = len(ranking)
    if n <= 1:
        return [ranking]

    expected_n_blocks = max(1, int(percentile * n))
    actual_n_blocks = rng.randint(
        max(1, expected_n_blocks - 1),
        min(n, expected_n_blocks + 2),
    )

    if actual_n_blocks >= n:
        return [[item] for item in ranking]
    if actual_n_blocks == 1:
        return [ranking]

    split_positions = sorted(rng.sample(range(1, n), actual_n_blocks - 1))
    blocks = []
    start = 0
    for sp in split_positions:
        blocks.append(ranking[start:sp])
        start = sp
    blocks.append(ranking[start:])
    return blocks


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
    rankings : list of lists
        Observed strict rankings (each a permutation of 0..n-1).
    n_clusters : int
        Number of clusters.
    initial_ranking : list of int, optional
        Seed Borda consensus order. Computed from data if None.
    gap_threshold : float
        Controls block granularity (legacy parameter, currently unused in splitting logic
        — splitting is controlled by ``_make_tied_ranking_with_percentile``).
    rng : random.Random, optional

    Returns
    -------
    list of length n_clusters, each element a list-of-lists block structure.
    """
    if rng is None:
        rng = random.Random()

    n = len(rankings[0])

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

    cluster_rankings = []
    for c in range(n_clusters):
        if n_clusters == 1:
            perturbed = initial_ranking[:]
            percentile = 0.5
        else:
            perturbed = initial_ranking[:]
            n_shuffles = max(2, n // 4)
            for _ in range(n_shuffles):
                start = rng.randrange(n - 1)
                end = min(start + rng.randint(2, max(3, n // 5)), n)
                segment = perturbed[start:end]
                rng.shuffle(segment)
                perturbed[start:end] = segment
            percentile = rng.uniform(0.4, 0.6)

        blocks = _make_tied_ranking_with_percentile(perturbed, percentile, rng)
        cluster_rankings.append(blocks)

    return cluster_rankings
