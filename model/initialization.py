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

    def make_tied_ranking_with_percentile(ranking: List[int], percentile: float) -> List[List[int]]:
        """Create tied blocks based on number of block boundaries.
        
        This method identifies natural break points in the consensus ranking
        by randomly selecting split positions. The percentile parameter controls the expected
        number of blocks:
        - percentile=0.1: Very few blocks (mostly tied)
        - percentile=0.5: Moderate blocks (middle ground)  
        - percentile=0.9: Many blocks (mostly separate)
        """
        n = len(ranking)
        if n <= 1:
            return [ranking]
        
        # Decide how many blocks to create: between 1 and n
        # percentile=0.5 means we expect about n/2 blocks on average
        expected_n_blocks = max(1, int(percentile * n))
        actual_n_blocks = rng.randint(max(1, expected_n_blocks - 1), 
                                      min(n, expected_n_blocks + 2))
        
        if actual_n_blocks >= n:
            # All separate blocks
            return [[item] for item in ranking]
        elif actual_n_blocks == 1:
            # All in one block
            return [ranking]
        else:
            # Create actual_n_blocks-1 random split points
            # This creates actual_n_blocks blocks
            split_positions = sorted(rng.sample(range(1, n), actual_n_blocks - 1))
            
            blocks = []
            start = 0
            for split_pos in split_positions:
                blocks.append(ranking[start:split_pos])
                start = split_pos
            blocks.append(ranking[start:])
            
            return blocks

    # Generate per-cluster rankings with random perturbations
    cluster_rankings = []
    for c in range(n_clusters):
        if n_clusters == 1:
            # Single cluster: use initial ranking as-is
            perturbed = initial_ranking[:]
            percentile = 0.5  # Default to medium number of blocks
        else:
            # Multiple clusters: add random perturbations and vary block structure
            perturbed = initial_ranking[:]
            
            # Aggressive perturbation: random shuffling of segments
            n_shuffles = max(2, n // 4)
            for _ in range(n_shuffles):
                start = rng.randrange(n - 1)
                end = min(start + rng.randint(2, max(3, n // 5)), n)
                segment = perturbed[start:end]
                rng.shuffle(segment)
                perturbed[start:end] = segment
            
            # Vary block structure diversity: percentile controls how many blocks
            # IMPORTANT: Keep variance LOW to avoid extreme imbalance between clusters
            # Range from 0.4 to 0.6 (moderate blocks) instead of 0.2-0.9
            # This prevents one cluster from becoming overwhelmingly attractive
            percentile = rng.uniform(0.4, 0.6)
        
        blocks = make_tied_ranking_with_percentile(perturbed, percentile)
        cluster_rankings.append(blocks)

    return cluster_rankings

