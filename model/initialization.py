"""Functions for generating initial cluster configurations."""

import random
import math
from typing import List, Optional, Tuple

import numpy as np
from sklearn.cluster import SpectralClustering

from .utils import invert_perm
from .dataclasses import ClusterParams



class _Fenwick:
    """Fenwick tree (BIT) for prefix sums used in Kendall distance calculation."""
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


def _kendall_inversions(rank_a: List[int], rank_b: List[int]) -> int:
    """
    Count Kendall tau distance (number of inversions) between two rankings.
    
    Maps items to positions in rank_a, then counts inversions in rank_b's
    position sequence using a Fenwick tree (O(n log n)).
    """
    n = len(rank_a)
    pos_in_a = {item: idx for idx, item in enumerate(rank_a)}
    seq = [pos_in_a[item] + 1 for item in rank_b]  # 1-indexed for BIT
    
    bit = _Fenwick(n)
    inv = 0
    seen = 0
    for x in seq:
        inv += seen - bit.sum(x)
        bit.add(x, 1)
        seen += 1
    return inv


def _build_item_agreement_matrix(rankings: List[List[int]]) -> np.ndarray:
    """
    Build an n×n matrix where entry [i,j] = how often item i is preferred to item j.
    
    For each ranking, we check pairwise preferences: if i appears before j,
    that's a vote for i over j.
    
    Returns
    -------
    agreement : ndarray(n, n), float
        agreement[i,j] = fraction of rankings where item i appears before item j
    """
    n_items = len(rankings[0])
    n_rankings = len(rankings)
    
    # Count pairwise preferences
    preference_count = np.zeros((n_items, n_items), dtype=np.float64)
    
    for ranking in rankings:
        pos = {item: idx for idx, item in enumerate(ranking)}
        for i in range(n_items):
            for j in range(n_items):
                if i != j:
                    # If i appears before j in this ranking, increment preference for i over j
                    if pos[i] < pos[j]:
                        preference_count[i, j] += 1.0
    
    # Normalize by number of rankings
    agreement = preference_count / n_rankings
    return agreement


def _build_antisymmetric_preference_matrix(agreement: np.ndarray, 
                                          ranked_items: List[int]) -> np.ndarray:
    """
    Build anti-symmetric preference strength matrix.
    
    For items i,j:
    - If lower Borda position of i < lower Borda position of j (i preferred in final ranking):
        Upper[i,j] = agreement[i,j]
        Lower[j,i] = 1 - agreement[i,j]
    - Preference strength = |difference|, which indicates how likely they should be tied
    
    Low strength (close to 0.5) → items should be tied
    High strength (close to 0 or 1) → items should be separated
    
    Parameters
    ----------
    agreement : ndarray(n, n)
        Pairwise agreement matrix from rankings
    ranked_items : list
        Items in their ranked order (Borda consensus or initial ranking)
    
    Returns
    -------
    preference : ndarray(n, n)
        Antisymmetric preference matrix where lower values suggest items should be tied
    """
    n = len(agreement)
    preference = np.zeros((n, n), dtype=np.float64)
    
    # Map items to final ranking positions
    pos_in_ranking = {ranked_items[idx]: idx for idx in range(len(ranked_items))}
    
    for i in range(n):
        for j in range(i + 1, n):
            # Positions in final ranking
            pos_i = pos_in_ranking.get(i, i)
            pos_j = pos_in_ranking.get(j, j)
            
            if pos_i < pos_j:  # i is ranked higher (preferred) in final ranking
                preference[i, j] = agreement[i, j]
                preference[j, i] = 1.0 - agreement[i, j]
            else:  # j is ranked higher
                preference[i, j] = 1.0 - agreement[j, i]
                preference[j, i] = agreement[j, i]
    
    return preference


def _agglomerative_block_formation_antisymmetric(
    ranked_items: List[int],
    agreement: np.ndarray,
    n_blocks: int
) -> List[List[int]]:
    """
    Form blocks by agglomerative clustering using anti-symmetric agreement matrix.
    
    Merges items that have low preference difference (close to 0.5 agreement),
    indicating weak preference relationship. When merging, computes new preference
    scores as the average over ALL items in the resulting block.
    
    Parameters
    ----------
    ranked_items : list
        Items in ranked order (Borda consensus)
    agreement : ndarray(n, n)
        Pairwise agreement matrix
    n_blocks : int
        Desired number of final blocks
    
    Returns
    -------
    blocks : list of lists
        Items grouped into n_blocks blocks, maintaining Borda order
    """
    n_items = len(ranked_items)
    
    if n_blocks >= n_items:
        return [[item] for item in ranked_items]
    if n_blocks <= 0:
        raise ValueError("n_blocks must be positive")
    
    # Start: each item in its own block
    blocks = [[item] for item in ranked_items]
    block_items = [[item] for item in ranked_items]  # Track which items are in each block
    
    while len(blocks) > n_blocks:
        # Find adjacent blocks with lowest preference difference (closest to tie)
        min_preference_diff = float('inf')
        merge_idx = 0
        
        for k in range(len(blocks) - 1):
            # Items in blocks k and k+1
            items_k = set(block_items[k])
            items_kp1 = set(block_items[k + 1])
            
            # Compute preference strength between block k and block k+1
            # as average of all cross-block pairwise preferences
            preference_sum = 0.0
            count = 0
            
            for i in items_k:
                for j in items_kp1:
                    # Antisymmetric preference: if ordering in Borda is i < j:
                    # preference[i,j] = agreement[i,j], preference[j,i] = 1 - agreement[i,j]
                    pos_i = ranked_items.index(i)
                    pos_j = ranked_items.index(j)
                    
                    if pos_i < pos_j:
                        pref = agreement[i, j]
                    else:
                        pref = 1.0 - agreement[i, j]
                    
                    # Preference strength: how far from 0.5 (ties)
                    # Closer to 0.5 = weaker preference = should tie
                    preference_diff = abs(pref - 0.5)
                    preference_sum += preference_diff
                    count += 1
            
            avg_preference_diff = preference_sum / count if count > 0 else 0.0
            
            if avg_preference_diff < min_preference_diff:
                min_preference_diff = avg_preference_diff
                merge_idx = k
        
        # Merge blocks merge_idx and merge_idx+1
        # Average the preference scores over ALL items in new block
        blocks[merge_idx].extend(blocks[merge_idx + 1])
        block_items[merge_idx].extend(block_items[merge_idx + 1])
        del blocks[merge_idx + 1]
        del block_items[merge_idx + 1]
    
    return blocks


def _sample_pitman_yor_blocks(gamma: float, delta: float, n_items: int, 
                              rng: random.Random) -> int:
    """
    Sample number of blocks K from a Pitman-Yor process.
    
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
    log_probs = []
    for K in range(1, n_items + 1):
        log_prob = K * math.log(gamma)
        for k in range(K):
            log_prob += math.log(gamma + delta * k)
        log_prob -= math.lgamma(K + 1)  # subtract log(K!)
        log_probs.append(log_prob)
    
    # Convert to probabilities (numerically stable)
    log_probs_arr = np.array(log_probs)
    log_probs_arr -= np.max(log_probs_arr)
    probs = np.exp(log_probs_arr)
    probs /= probs.sum()
    
    # Sample using numpy.random (which uses the global seed we set)
    K = np.random.choice(range(1, n_items + 1), p=probs)
    return int(K)


def init_blocks_spectral(
    rankings: List[List[int]],
    n_clusters: int,
    *,
    gamma: float = 1.0,
    delta: float = 0.5,
    theta: float = 1.0,
    py_sampling: bool = True,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[ClusterParams]:
    """
    Generate initial cluster configurations using spectral clustering.
    
    Algorithm:
    1. Build item-level agreement matrix: how often each item i is preferred to j
    2. Use spectral clustering on rankings to assign clusters
    3. For each cluster:
       a. Compute Borda consensus ranking from cluster members
       b. Build anti-symmetric preference matrix from agreement
       c. Sample number of blocks K using either:
          - Pitman-Yor prior (if py_sampling=True): More flexible block count
          - Simple random (if py_sampling=False): Around n/2 blocks, like original
       d. Form K blocks via agglomerative clustering (merging items with weak preferences)
    
    Parameters
    ----------
    rankings : list of lists
        All ranking data (N rankings, each with n items)
    n_clusters : int
        Number of clusters to form
    gamma : float, default=1.0
        Pitman-Yor strength parameter (only used if py_sampling=True)
    delta : float, default=0.5
        Pitman-Yor discount parameter (only used if py_sampling=True)
    theta : float, default=1.0
        Initial theta (concentration parameter for item ordering within blocks)
    py_sampling : bool, default=True
        If True, sample number of blocks from Pitman-Yor distribution.
        If False, use simpler approach: expect ~n/2 blocks with small random variation.
        
        - py_sampling=True: More blocks, Pitman-Yor prior (new)
        - py_sampling=False: Fewer blocks (n/2±2), simpler & original-like approach
    seed : int, optional
        Random seed for reproducibility
    rng : random.Random, optional
        Random number generator
    
    Returns
    -------
    clusters : list of ClusterParams
        List of cluster configurations, ready to pass to MixtureRankingModel via init_clusters
    """
    if rng is None:
        rng = random.Random(seed)
    
    if seed is not None:
        np.random.seed(seed)
    
    n_items = len(rankings[0])
    n_rankings = len(rankings)
    
    # Step 1: Build item-level agreement matrix (how often each item is preferred to each other)
    agreement = _build_item_agreement_matrix(rankings)
    
    # Step 2: Spectral clustering on rankings (using agreement-based affinity)
    # Build ranking-level similarity from item agreement
    ranking_agreement = np.zeros((n_rankings, n_rankings))
    for i in range(n_rankings):
        for j in range(i + 1, n_rankings):
            # Agreement between two rankings: count concordant pairs
            inv = _kendall_inversions(rankings[i], rankings[j])
            total_pairs = n_items * (n_items - 1) // 2
            agree = total_pairs - inv
            ranking_agreement[i, j] = agree
            ranking_agreement[j, i] = agree
    
    # Diagonal is self-agreement
    np.fill_diagonal(ranking_agreement, n_items * (n_items - 1) // 2)
    
    # Normalize for spectral clustering
    ranking_agreement_norm = ranking_agreement / (ranking_agreement.max() + 1e-10)
    
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=seed or 0,
    )
    cluster_assignment = sc.fit_predict(ranking_agreement_norm)
    
    # Step 3: For each cluster, build block structure
    clusters = []
    
    for c in range(n_clusters):
        cluster_indices = [i for i in range(n_rankings) if cluster_assignment[i] == c]
        
        if not cluster_indices:
            # Empty cluster: put all items in one block
            blocks = [list(range(n_items))]
        else:
            # Get cluster-specific rankings
            cluster_data = [rankings[i] for i in cluster_indices]
            
            # Compute Borda consensus ranking for this cluster
            pos_sum = np.zeros(n_items)
            for ranking in cluster_data:
                pos = {item: idx for idx, item in enumerate(ranking)}
                for item in range(n_items):
                    pos_sum[item] += pos[item]
            
            mean_pos = pos_sum / len(cluster_data)
            borda_ranking = sorted(range(n_items), key=lambda i: mean_pos[i])
            
            # Form blocks using selected method
            preference = _build_antisymmetric_preference_matrix(agreement, borda_ranking)
            K = _sample_pitman_yor_blocks(gamma, delta, n_items, rng)
            blocks = _agglomerative_block_formation_antisymmetric(
                borda_ranking, agreement, K
            )

        # Create ClusterParams object
        cluster = ClusterParams(
            blocks=blocks,
            theta=theta,
            gamma=gamma,
            delta=delta,
        )
        clusters.append(cluster)
    
    return clusters


def init_spectral_with_z(
    rankings: List[List[int]],
    n_clusters: int,
    *,
    gamma: float = 1.0,
    delta: float = 0.5,
    theta: float = 1.0,
    py_sampling: bool = True,
    seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[ClusterParams], List[int]]:
    """
    Generate initial cluster configurations using spectral clustering.
    
    Returns BOTH the cluster parameters AND the spectral clustering assignments
    for z, allowing intelligent initialization of cluster assignments and weights.
    
    Parameters
    ----------
    rankings : list of lists
        All ranking data (N rankings, each with n items)
    n_clusters : int
        Number of clusters to form
    gamma : float, default=1.0
        Pitman-Yor strength parameter (only used if py_sampling=True)
    delta : float, default=0.5
        Pitman-Yor discount parameter (only used if py_sampling=True)
    theta : float, default=1.0
        Initial theta (concentration parameter for item ordering within blocks)
    py_sampling : bool, default=True
        If True, sample number of blocks from Pitman-Yor distribution.
        If False, use simpler approach: expect ~n/2 blocks with small random variation.
    seed : int, optional
        Random seed for reproducibility
    rng : random.Random, optional
        Random number generator
    
    Returns
    -------
    clusters : list of ClusterParams
        List of cluster configurations
    z : list of int
        Cluster assignments from spectral clustering (z[i] = cluster for ranking i)
    """
    if rng is None:
        rng = random.Random(seed)
    
    if seed is not None:
        np.random.seed(seed)
    
    n_items = len(rankings[0])
    n_rankings = len(rankings)
    
    # Step 1: Build item-level agreement matrix
    agreement = _build_item_agreement_matrix(rankings)
    
    # Step 2: Spectral clustering on rankings
    ranking_agreement = np.zeros((n_rankings, n_rankings))
    for i in range(n_rankings):
        for j in range(i + 1, n_rankings):
            inv = _kendall_inversions(rankings[i], rankings[j])
            total_pairs = n_items * (n_items - 1) // 2
            agree = total_pairs - inv
            ranking_agreement[i, j] = agree
            ranking_agreement[j, i] = agree
    
    np.fill_diagonal(ranking_agreement, n_items * (n_items - 1) // 2)
    ranking_agreement_norm = ranking_agreement / (ranking_agreement.max() + 1e-10)
    
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='kmeans',
        random_state=seed or 0,
    )
    cluster_assignment = sc.fit_predict(ranking_agreement_norm)
    
    # Step 3: For each cluster, build block structure
    clusters = []
    
    for c in range(n_clusters):
        cluster_indices = [i for i in range(n_rankings) if cluster_assignment[i] == c]
        
        if not cluster_indices:
            blocks = [list(range(n_items))]
        else:
            cluster_data = [rankings[i] for i in cluster_indices]
            
            pos_sum = np.zeros(n_items)
            for ranking in cluster_data:
                pos = {item: idx for idx, item in enumerate(ranking)}
                for item in range(n_items):
                    pos_sum[item] += pos[item]
            
            mean_pos = pos_sum / len(cluster_data)
            borda_ranking = sorted(range(n_items), key=lambda i: mean_pos[i])
            
            preference = _build_antisymmetric_preference_matrix(agreement, borda_ranking)
            K = _sample_pitman_yor_blocks(gamma, delta, n_items, rng)
            blocks = _agglomerative_block_formation_antisymmetric(
                borda_ranking, agreement, K
            )

        cluster = ClusterParams(
            blocks=blocks,
            theta=theta,
            gamma=gamma,
            delta=delta,
        )
        clusters.append(cluster)
    
    return clusters, cluster_assignment.tolist()


def init_clusters_default(
    rankings: List[List[int]],
    n_clusters: int,
    *,
    init_theta: float = 1.0,
    init_gamma: float = 1.0,
    init_delta: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[List[ClusterParams], List[int]]:
    """Trivial initialization: each cluster starts with all items in one block.

    The MCMC Gibbs moves will break blocks apart from this starting point.
    For better-quality initialization use ``init_spectral_with_z()`` instead.

    Returns
    -------
    clusters : list of ClusterParams
    z : list of int
        Round-robin cluster assignments for each ranking (shuffled).
    """
    n = len(rankings[0])
    N = len(rankings)
    rng = random.Random(seed)
    clusters = [
        ClusterParams(
            blocks=[list(range(n))],
            theta=init_theta,
            gamma=init_gamma,
            delta=init_delta,
        )
        for _ in range(n_clusters)
    ]
    z = [i % n_clusters for i in range(N)]
    rng.shuffle(z)
    return clusters, z

