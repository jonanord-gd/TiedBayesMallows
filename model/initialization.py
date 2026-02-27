"""Functions for generating initial cluster configurations."""

import random
from typing import List, Optional


from .utils import invert_perm


def init_blocks_borda_threshold(
    rankings: List[List[int]],
    n_clusters: int,
    *,
    initial_ranking: Optional[List[int]] = None,
    gap_threshold: float = 0.35,
    rng: Optional[random.Random] = None,
) -> List[List[List[int]]]:
    """Generate initial tied rankings for each cluster via Borda consensus.

    See the original docstring in the monolithic file for details.  This
    helper is used by :class:`MixtureRankingModel` when clusters are
    auto‑generated.  The implementation was moved verbatim from
    ``TiedMallowsObject.py``.
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
