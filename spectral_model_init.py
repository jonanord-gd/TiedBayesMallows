"""
Integration layer: Use spectral clustering initialization with MixtureRankingModel.

This module provides utilities to:
1. Initialize clusters using spectral clustering on ranking agreement
2. Convert spectral results to MixtureRankingModel ClusterParams
3. Provide initialization options (Spectral + sampling vs. traditional Borda)
"""

from typing import List, Tuple, Optional
import random

from model.dataclasses import ClusterParams
from Spectral_cluster_init import spectral_init_clusters, sample_pitman_yor_K


def spectral_initialization(
    rankings: List[List[int]],
    n_clusters: int,
    gamma: float = 1.0,
    delta: float = 0.5,
    init_theta: float = 1.0,
    use_kendall: bool = True,
    seed: int = 123,
) -> List[ClusterParams]:
    """
    Initialize clusters using spectral clustering + Pitman-Yor block sampling.
    
    Parameters
    ----------
    rankings : list of lists
        All ranking data (each ranking is a permutation of 0..n-1)
    n_clusters : int
        Number of clusters to create
    gamma, delta : float
        Pitman-Yor hyperparameters
        - gamma: strength (higher = more clusters expected)
        - delta: discount (0 <= delta < 1)
    init_theta : float
        Initial theta parameter for all clusters
    use_kendall : bool
        If True: Use Kendall distance agreement matrix
        If False: Use RBF kernel (faster, good for rankings)
    seed : int
        Random seed for reproducibility
    
    Returns
    -------
    init_clusters : list of ClusterParams
        Cluster parameters initialized with spectral clustering + block formation
    
    Example
    -------
    >>> from model.core import MixtureRankingModel
    >>> from spectral_model_init import spectral_initialization
    >>>
    >>> init_clusters = spectral_initialization(
    ...     rankings=my_rankings,
    ...     n_clusters=3,
    ...     use_kendall=True,  # Use Kendall distance
    ...     seed=123
    ... )
    >>> model = MixtureRankingModel(my_rankings, init_clusters=init_clusters)
    >>> final_state, samples = model.run_mcmc(n_iter=5000, burn_in=500)
    """
    
    # Run spectral clustering
    z, cluster_blocks = spectral_init_clusters(
        rankings=rankings,
        n_clusters=n_clusters,
        gamma=gamma,
        delta=delta,
        use_kendall=use_kendall,
        seed=seed,
    )
    
    # Convert to ClusterParams
    init_clusters = [
        ClusterParams(
            blocks=cluster_blocks[c],
            theta=init_theta,
            gamma=gamma,
            delta=delta,
        )
        for c in range(n_clusters)
    ]
    
    return init_clusters


def print_spectral_init_info(init_clusters: List[ClusterParams]) -> None:
    """Print summary of spectral initialization."""
    print("\nSpectral Clustering Initialization Summary:")
    print("=" * 70)
    for c, cluster in enumerate(init_clusters):
        K = len(cluster.blocks)
        sizes = [len(b) for b in cluster.blocks]
        print(f"Cluster {c}: {K} blocks, theta={cluster.theta:.3f}, "
              f"gamma={cluster.gamma:.3f}, delta={cluster.delta:.3f}")
        print(f"  Block sizes: {sizes}")
        print(f"  Blocks: {cluster.blocks}")
    print("=" * 70)


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    print("Spectral Clustering Initialization Demo")
    print("=" * 70)
    
    # Generate synthetic data
    np.random.seed(123)
    n_assessors = 50
    n_items = 10
    n_true_clusters = 3
    
    # Create synthetic rankings from clusters
    rankings = []
    true_blocks = [
        [0, 1, 2],           # Cluster 0: items 0-2 prefer together
        [3, 4, 5, 6],        # Cluster 0: items 3-6 prefer together
        [7, 8, 9],           # Cluster 0: items 7-9 prefer together
    ]
    
    # Generate rankings from different clusters
    for _ in range(n_assessors // 2):
        # Cluster 0: strong preference for 0>1>2>3>...
        perm = np.random.permutation(n_items).tolist()
        rankings.append(perm)
    for _ in range(n_assessors // 2):
        # Cluster 1: reverse preference for 9>8>7>...
        perm = np.random.permutation(n_items).tolist()
        rankings.append(perm)
    
    print(f"Input: {n_assessors} rankings, {n_items} items")
    print(f"Creating {n_true_clusters} clusters\n")
    
    # Initialize using spectral clustering
    init_clusters = spectral_initialization(
        rankings=rankings,
        n_clusters=n_true_clusters,
        use_kendall=False,  # Use RBF (faster)
        seed=123,
    )
    
    print_spectral_init_info(init_clusters)
    
    # Now you can use with MixtureRankingModel:
    # model = MixtureRankingModel(rankings, init_clusters=init_clusters)
    # final_state, samples = model.run_mcmc(n_iter=1000, burn_in=100)
