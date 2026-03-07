from typing import List, Sequence, Hashable, Tuple
import numpy as np
from sklearn.cluster import SpectralClustering
import random
import math


class Fenwick:
    """Fenwick tree (BIT) for prefix sums over 1..n."""
    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, i: int, delta: int) -> None:
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def sum(self, i: int) -> int:
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s


def kendall_inversions(rank_a: Sequence[int], rank_b: Sequence[int]) -> int:
    """
    Kendall tau distance (number of inversions) between two rankings of the same items.
    rank_a and rank_b must be permutations of the same items (represented as IDs/labels).

    Implementation:
      Map items -> position in rank_a, then translate rank_b into that position list,
      and count inversions in that list.
    """
    n = len(rank_a)
    pos_in_a = {item: idx for idx, item in enumerate(rank_a)}  # 0-based

    # Translate rank_b items into positions in rank_a (0-based -> make 1-based for BIT)
    seq = [pos_in_a[item] + 1 for item in rank_b]

    # Count inversions in seq using BIT
    bit = Fenwick(n)
    inv = 0
    seen = 0
    for x in seq:
        # number of seen elements > x  = seen - (# <= x)
        inv += seen - bit.sum(x)
        bit.add(x, 1)
        seen += 1
    return inv


def agreement_matrix(rankings: List[Sequence[Hashable]]) -> np.ndarray:
    """
    Build an m x m matrix A where A[i,j] = number of agreeing pairwise preferences
    (concordant pairs) between rankings i and j.

    Assumes:
      - Each ranking is a total order (no ties)
      - All rankings contain the same items, just permuted
    """
    m = len(rankings)
    if m == 0:
        return np.zeros((0, 0), dtype=np.int64)

    n = len(rankings[0])
    items0 = set(rankings[0])
    for r in rankings:
        if len(r) != n:
            raise ValueError("All rankings must have the same length.")
        if set(r) != items0:
            raise ValueError("All rankings must contain the same items.")

    total_pairs = n * (n - 1) // 2
    A = np.zeros((m, m), dtype=np.int64)

    # diagonal is max agreement with itself
    np.fill_diagonal(A, total_pairs)

    for i in range(m):
        for j in range(i + 1, m):
            inv = kendall_inversions(rankings[i], rankings[j])
            agree = total_pairs - inv
            A[i, j] = agree
            A[j, i] = agree
    return A


# --- example ---
if __name__ == "__main__":
    rankings = [
        ["a", "b", "c", "d"],
        ["b", "a", "c", "d"],
        ["d", "c", "b", "a"],
    ]
    A = agreement_matrix(rankings)

    A_normalized = A / A.max()
    
    n_clusters = 3

    # Use agreement matrix directly as affinity
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',  # use precomputed similarity matrix
        assign_labels="kmeans",
        random_state=0
    )
    
    labels = sc.fit_predict(A_normalized)


def borda_ranking(rankings: List[Sequence[Hashable]]) -> List[Hashable]:
    """
    Compute Borda ranking: each item gets points equal to its position in each ranking.
    Higher position (lower rank) = higher score.
    
    Returns items sorted by Borda score (descending).
    """
    if not rankings or not rankings[0]:
        return []
    
    items = set(rankings[0])
    borda_scores = {item: 0 for item in items}
    
    # Each ranking votes: last item gets n points, first item gets 1 point
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            borda_scores[item] += (len(ranking) - pos)  # reversed: position n gets n points
    
    # Sort by score descending
    sorted_items = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, score in sorted_items]


def preference_strength_matrix(borda_rank: List[Hashable], n_items: int) -> np.ndarray:
    """
    Build matrix M where M[i,j] = preference difference strength between items i and j.
    
    Higher value = items are far apart in preference (strong difference)
    Lower value = items are close in preference (weak difference)
    
    Strength is based on position gap in the central Borda ranking.
    For adjacent items (consecutive in ranking), strength = 1.
    For items k positions apart, strength = k.
    """
    M = np.zeros((n_items, n_items), dtype=np.float64)
    
    # Create position map: item -> position in Borda ranking
    pos_map = {item: idx for idx, item in enumerate(borda_rank)}
    
    for i in range(n_items):
        for j in range(i + 1, n_items):
            # Position difference = how far apart in borda ranking
            pos_i = pos_map.get(i, i)  # fallback if item not in ranking
            pos_j = pos_map.get(j, j)
            strength = abs(pos_i - pos_j)
            M[i, j] = strength
            M[j, i] = strength
    
    return M


def agglomerative_block_formation(borda_rank: List[Hashable], n_blocks: int) -> List[List[Hashable]]:
    """
    Form blocks by agglomerative clustering in the Borda ranking order.
    
    Start with each item in its own block, then merge adjacent blocks with
    smallest preference difference until we have n_blocks blocks.
    
    This preserves locality: items close in the central ranking stay together.
    
    Parameters
    ----------
    borda_rank : list
        Central ranking (Borda voting result)
    n_blocks : int
        Desired number of blocks
    
    Returns
    -------
    blocks : list of lists
        Items grouped into n_blocks blocks, maintaining Borda order within blocks
    """
    n_items = len(borda_rank)
    if n_blocks >= n_items:
        return [[item] for item in borda_rank]
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    
    # Start: each item is in its own block
    blocks = [[item] for item in borda_rank]
    
    # Agglomerative: repeatedly merge adjacent blocks with smallest gap
    while len(blocks) > n_blocks:
        # Find the adjacent pair with smallest preference gap
        min_gap = float('inf')
        merge_idx = 0
        
        for k in range(len(blocks) - 1):
            # Gap between last item in block k and first item in block k+1
            last_item_k = blocks[k][-1]
            first_item_kp1 = blocks[k + 1][0]
            
            # Position difference in original borda ranking
            pos_k = borda_rank.index(last_item_k)
            pos_kp1 = borda_rank.index(first_item_kp1)
            gap = pos_kp1 - pos_k  # Always positive since borda_rank is ordered
            
            if gap < min_gap:
                min_gap = gap
                merge_idx = k
        
        # Merge blocks merge_idx and merge_idx+1
        blocks[merge_idx].extend(blocks[merge_idx + 1])
        del blocks[merge_idx + 1]
    
    return blocks


def sample_pitman_yor_K(gamma: float, delta: float, n_items: int, rng: random.Random = None) -> int:
    """
    Sample number of blocks K from a Pitman-Yor process.
    
    The number of clusters follows approximately a Chinese Restaurant Process
    distribution. For small n_items, we use a simplified approach:
    
    P(K) ∝ (gamma + delta*0) * (gamma + delta*1) * ... * (gamma + delta*(K-1)) / K!
    
    Parameters
    ----------
    gamma : float
        Strength parameter (gamma > 0)
    delta : float
        Discount parameter (0 <= delta < 1)
    n_items : int
        Number of items to partition
    rng : random.Random
        Random number generator
    
    Returns
    -------
    K : int
        Sampled number of blocks (1 <= K <= n_items)
    """
    if rng is None:
        rng = random.Random()
    
    # Precompute log probabilities for K = 1, ..., n_items
    log_probs = []
    for K in range(1, n_items + 1):
        # log(gamma) * prod_{k=0}^{K-1} (gamma + delta*k) / K!
        log_prob = K * math.log(gamma)
        for k in range(K):
            log_prob += math.log(gamma + delta * k)
        log_prob -= math.lgamma(K + 1)  # subtract log(K!)
        log_probs.append(log_prob)
    
    # Convert to probabilities (numerically stable)
    log_probs = np.array(log_probs)
    log_probs -= np.max(log_probs)  # shift for stability
    probs = np.exp(log_probs)
    probs /= probs.sum()
    
    # Sample
    K = np.random.choice(range(1, n_items + 1), p=probs)
    return K


def spectral_init_clusters(
    rankings: List[List[int]],
    n_clusters: int,
    gamma: float = 1.0,
    delta: float = 0.5,
    use_kendall: bool = True,
    seed: int = None,
) -> Tuple[List[List[int]], List[List[List[int]]]]:
    """
    Initialize clusters using spectral clustering on ranking agreement.
    
    For each cluster:
    1. Compute Borda ranking (consensus ranking)
    2. Sample K (number of blocks) from Pitman-Yor
    3. Form K blocks using agglomerative clustering on Borda rank
    
    Parameters
    ----------
    rankings : list of lists
        All ranking data
    n_clusters : int
        Number of clusters to form
    gamma, delta : float
        Pitman-Yor hyperparameters for block count
    use_kendall : bool
        If True, use Kendall distance agreement matrix (Spectral_cluster_init specific)
        If False, use RBF kernel (sklearn default)
    seed : int
        Random seed
    
    Returns
    -------
    z : list
        Cluster assignment for each ranking
    cluster_blocks : list of lists
        For each cluster c, list of block structures (as list[list[int]])
        cluster_blocks[c] = [[items in block 0], [items in block 1], ...]
    """
    if seed is not None:
        np.random.seed(seed)
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    n_items = len(rankings[0])
    
    # Step 1: Spectral clustering
    if use_kendall:
        A = agreement_matrix(rankings)
        A_normalized = A / (A.max() + 1e-10)
    else:
        # Use RBF kernel similarity (faster, good for rankings)
        from sklearn.metrics.pairwise import rbf_kernel
        # Compute pairwise Spearman correlation
        rankings_array = np.array(rankings)
        # RBF kernel on rankings (treats rankings as vectors)
        A_normalized = rbf_kernel(rankings_array.astype(float), gamma=1.0/n_items)
    
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=seed or 0,
    )
    z = sc.fit_predict(A_normalized)
    
    # Step 2: For each cluster, build block structure
    cluster_blocks = []
    for c in range(n_clusters):
        cluster_rankings = [rankings[i] for i in range(len(rankings)) if z[i] == c]
        
        if not cluster_rankings:
            # Empty cluster - create single block with all items
            cluster_blocks.append([list(range(n_items))])
            continue
        
        # Borda ranking within cluster
        borda = borda_ranking(cluster_rankings)
        
        # Sample K from Pitman-Yor
        K = sample_pitman_yor_K(gamma, delta, n_items, rng)
        
        # Form blocks by agglomerative clustering
        blocks = agglomerative_block_formation(borda, K)
        cluster_blocks.append(blocks)
    
    return z.tolist(), cluster_blocks 
