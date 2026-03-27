"""
Data generation for a mixture of K(p)-Mallows models with Pitman-Yor block structure.

This follows the Weak-Mallows Model (WMM) described in the notes:
  - Central ranking rho_m is a WEAK ORDER (blocks of tied items)
  - Observed rankings r are STRICT (no ties)
  - Distance is K(p) (generalised Kemeny), which for strict r vs. weak rho_m
    decomposes as:
        d_{K(p)}(r, rho_m) = D_r(m) + p * T(m)
    where
        D_r(m)  = cross-block inversions  (varies across r)
        T(m)    = sum_s m_s * s*(s-1)/2   (fixed for a given block structure)
        p       = tie penalty in (0, 1/2) for the WMM use-case

  The normalisation constant Z*(rho_m, theta) has the closed form (eq. 15):
        Z*(rho_m, theta) = exp(-theta * p * T(m)) * P(m)
                           * [n]_{e^{-2theta}}! / prod_k [s_k]_{e^{-2theta}}!

  Since T(m) and p only enter as the constant prefactor exp(-theta*p*2*T(m)),
  they do NOT affect the relative sampling weights of strict rankings — only
  cross-block inversions drive the sampler.  The within-block ordering is
  uniform (all permutations give the same distance).

Generative model per cluster c:
  1. pi_c  ~ PY(delta_c, lambda_c)          block partition of items
  2. sigma_c | pi_c ~ Uniform(K_c!)         ordering of blocks
  3. rho_c = (B_{sigma(1)} > ... > B_{sigma(K)})  weak-order central ranking
  4. For each ranker i in cluster c:
       a. Draw within-block permutation (uniform) -> defines relative order
          of items WITHIN each block for the insertion step
       b. Sample strict r_i ~ K(p)-Mallows(rho_c, theta_c, p_c) via
          block-respecting repeated insertion
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    """One mixture component."""
    blocks: list[list[int]]    # unordered partition, e.g. [[0,2],[1],[3,4]]
    block_order: list[int]     # indices into blocks, highest -> lowest rank
    theta: float               # Mallows precision  (theta > 0)
    p: float                   # K(p) tie penalty   (0 < p < 1/2 for WMM)
    weight: float              # mixture weight pi_c

    @property
    def ordered_blocks(self) -> list[list[int]]:
        """Blocks from highest to lowest rank."""
        return [self.blocks[i] for i in self.block_order]

    @property
    def block_sizes(self) -> list[int]:
        return [len(self.blocks[i]) for i in self.block_order]


@dataclass
class Dataset:
    rankings: np.ndarray             # (n_rankers, n_items)  strict, 0-indexed
    cluster_labels: np.ndarray       # (n_rankers,)
    clusters: list[Cluster]
    n_items: int
    n_rankers: int


# ---------------------------------------------------------------------------
# Block-structure helpers
# ---------------------------------------------------------------------------

def T(block_sizes: list[int]) -> int:
    """
    Number of within-block ties T(m) = sum_k s_k*(s_k-1)/2.
    This is the fixed tie-count contribution to the K(p) distance.
    """
    return sum(s * (s - 1) // 2 for s in block_sizes)


def P(block_sizes: list[int]) -> int:
    """
    Number of within-block permutations P(m) = prod_k s_k!
    All of these give the same cross-block distance, so they are
    equally likely under the Mallows distribution.
    """
    result = 1
    for s in block_sizes:
        for j in range(1, s + 1):
            result *= j
    return result


def log_Z_star(block_sizes: list[int], theta: float, p: float) -> float:
    """
    Log normalisation constant (eq. 15 from the notes):

        log Z*(rho_m, theta) = -theta * p * T(m)
                               + log P(m)
                               + log [n]_{e^{-2theta}}!
                               - sum_k log [s_k]_{e^{-2theta}}!

    where  [m]_q! = prod_{j=1}^{m} (1 - q^j) / (1 - q)

    The tie penalty is p * T(m) (not 2*p*T(m)) because K(p) counts unordered
    pairs directly: K(p) = |D| + p*(|R1| + |R2|), and for strict r vs weak
    rho_m, |R2| = T(m) and |R1| = 0.
    """
    import math
    n = sum(block_sizes)
    phi2 = np.exp(-2.0 * theta)

    def log_q_factorial(m: int, q: float) -> float:
        if abs(1.0 - q) < 1e-12:
            return sum(np.log(j) for j in range(1, m + 1))
        return sum(np.log((1.0 - q**j) / (1.0 - q)) for j in range(1, m + 1))

    log_Tm  = -theta * p * T(block_sizes)
    log_Pm  = sum(math.lgamma(s + 1) for s in block_sizes)
    log_num = log_q_factorial(n, phi2)
    log_den = sum(log_q_factorial(s, phi2) for s in block_sizes)

    return log_Tm + log_Pm + log_num - log_den


# ---------------------------------------------------------------------------
# Step 1: Pitman-Yor partition (CRP construction)
# ---------------------------------------------------------------------------

def pitman_yor_partition(
    n_items: int,
    delta: float,
    lam: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    """
    Draw a random partition of n_items via the Pitman-Yor CRP.

    Parameters
    ----------
    delta : float   discount in [0, 1)     (delta=0 -> Dirichlet process)
    lam   : float   strength > -delta
    """
    if not (0 <= delta < 1):
        raise ValueError(f"delta must be in [0, 1), got {delta}")
    if lam <= -delta:
        raise ValueError(f"lam must be > -delta, got lam={lam}, delta={delta}")

    assignments  = []
    block_counts = []

    for i in range(n_items):
        K = len(block_counts)
        weights = [max(c - delta, 0.0) for c in block_counts]
        weights.append(lam + delta * K)
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

        choice = rng.choice(len(weights), p=weights)
        if choice == K:
            block_counts.append(1)
        else:
            block_counts[choice] += 1
        assignments.append(choice)

    blocks = defaultdict(list)
    for item, blk in enumerate(assignments):
        blocks[blk].append(item)
    return [blocks[k] for k in sorted(blocks)]


# ---------------------------------------------------------------------------
# Step 2: Random block ordering (uniform)
# ---------------------------------------------------------------------------

def random_block_order(n_blocks: int, rng: np.random.Generator) -> list[int]:
    return list(rng.permutation(n_blocks))


# ---------------------------------------------------------------------------
# Step 3 & 4: Sample a strict ranking from K(p)-Mallows(rho_m, theta, p)
# ---------------------------------------------------------------------------

def sample_mallows_weak_center(
    ordered_blocks: list[list[int]],
    theta: float,
    p: float,
    rng: np.random.Generator,
) -> list[int]:
    """
    Sample one strict ranking r from K(p)-Mallows(rho_m, theta, p) where
    rho_m is a weak order given by `ordered_blocks`.

    Key insight from eq. (13) in the notes:
        Z*(rho_m, theta) = exp(-theta * p * T(m))
                           * sum_{r strict} exp(-theta * D_r(m))

    The factor exp(-theta * p * T(m)) is CONSTANT across all strict r,
    so it cancels in the normalised probability.  The sampling distribution
    is therefore entirely determined by the cross-block inversion count D_r(m).

    Algorithm — block-respecting zone insertion:
    --------------------------------------------
    Process blocks from highest to lowest rank.  When inserting block b
    (size s_b) into a partial ranking of length L (all from higher-ranked
    blocks), we choose a contiguous insertion zone of width s_b.

    Zone start position j (0 <= j <= L) determines cross-block inversions:
      - The s_b new items will each be inverted with each of the j items
        already placed (all strictly higher-ranked in rho_m).
      - Total new cross-block inversions: s_b * j
      - Weight: phi2^(s_b * j),  phi2 = exp(-2*theta)
        (factor of 2 from the Kemeny matrix symmetry, eq. 13)

    Within the chosen zone, items are placed in a uniformly random order
    (all within-block permutations contribute equally to the distance).
    """
    phi2 = np.exp(-2.0 * theta)

    ranking: list[int] = []

    for block in ordered_blocks:
        s = len(block)
        L = len(ranking)
        n_zones = L + 1

        # Weight of zone starting at j: phi2^(s * j)
        weights = np.array([phi2 ** (s * j) for j in range(n_zones)],
                           dtype=float)
        weights /= weights.sum()

        zone_start = rng.choice(n_zones, p=weights)

        # Uniform within-block order
        within_order = rng.permutation(block).tolist()

        ranking = ranking[:zone_start] + within_order + ranking[zone_start:]

    return ranking


# ---------------------------------------------------------------------------
# Cluster factory
# ---------------------------------------------------------------------------

def make_cluster(
    n_items: int,
    delta: float,
    lam: float,
    theta: float,
    p: float,
    weight: float,
    rng: np.random.Generator,
) -> Cluster:
    blocks      = pitman_yor_partition(n_items, delta, lam, rng)
    block_order = random_block_order(len(blocks), rng)
    return Cluster(blocks=blocks, block_order=block_order,
                   theta=theta, p=p, weight=weight)


# ---------------------------------------------------------------------------
# Full dataset generator
# ---------------------------------------------------------------------------

def generate_dataset(
    n_items: int,
    n_rankers: int,
    clusters: list[Cluster],
    rng: Optional[np.random.Generator] = None,
    seed: Optional[int] = None,
) -> Dataset:
    """
    Generate n_rankers strict rankings from a mixture of K(p)-Mallows models
    with PY block-structured central rankings.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    weights = np.array([c.weight for c in clusters], dtype=float)
    weights /= weights.sum()
    cluster_labels = rng.choice(len(clusters), size=n_rankers, p=weights)

    rankings = np.empty((n_rankers, n_items), dtype=int)

    for i in range(n_rankers):
        c = clusters[cluster_labels[i]]
        r = sample_mallows_weak_center(c.ordered_blocks, c.theta, c.p, rng)
        rankings[i] = r

    return Dataset(
        rankings=rankings,
        cluster_labels=cluster_labels,
        clusters=clusters,
        n_items=n_items,
        n_rankers=n_rankers,
    )


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def generate_mixture_dataset(
    n_items: int,
    n_rankers: int,
    n_clusters: int,
    delta: float = 0.3,
    lam: float = 1.0,
    theta: float | list[float] = 1.0,
    p: float | list[float] = 0.3,
    weights: Optional[list[float]] = None,
    seed: Optional[int] = None,
) -> Dataset:
    """
    High-level helper: draw PY block structures and generate rankings.

    Parameters
    ----------
    delta   : PY discount in [0,1), shared across clusters
    lam     : PY strength > -delta, shared across clusters
    theta   : Mallows precision(s). Scalar = shared; list = per cluster.
    p       : K(p) tie penalty(s). For the WMM use p < 0.5.
              Scalar = shared; list = per cluster.
    weights : mixture weights (uniform if None)
    """
    rng = np.random.default_rng(seed)

    if isinstance(theta, (int, float)):
        theta = [float(theta)] * n_clusters
    if isinstance(p, (int, float)):
        p = [float(p)] * n_clusters
    if weights is None:
        weights = [1.0 / n_clusters] * n_clusters

    clusters = [
        make_cluster(n_items, delta, lam, theta[k], p[k], weights[k], rng)
        for k in range(n_clusters)
    ]
    return generate_dataset(n_items, n_rankers, clusters, rng=rng)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def kemeny_distance_strict_vs_weak(
    r: list[int],
    ordered_blocks: list[list[int]],
    p: float,
) -> float:
    """
    Compute d_{K(p)}(r, rho_m) for strict r and weak-order rho_m.
    Uses eq. (11): d = D_r(m) + p * 2 * T(m).
    """
    pos_r = {item: idx for idx, item in enumerate(r)}

    block_rank = {}
    for b_rank, block in enumerate(ordered_blocks):
        for item in block:
            block_rank[item] = b_rank

    items = [item for block in ordered_blocks for item in block]
    n = len(items)
    Dr = 0
    for a in range(n):
        for b in range(a + 1, n):
            i, j = items[a], items[b]
            if block_rank[i] == block_rank[j]:
                continue   # within-block: tied in rho_m
            rho_i_above_j = block_rank[i] < block_rank[j]
            r_i_above_j   = pos_r[i] < pos_r[j]
            if rho_i_above_j != r_i_above_j:
                Dr += 2    # symmetric Kemeny matrix

    block_sizes = [len(b) for b in ordered_blocks]
    return Dr / 2 + p * T(block_sizes)


def summarise_dataset(ds: Dataset) -> None:
    print(f"Dataset: {ds.n_rankers} rankers, {ds.n_items} items, "
          f"{len(ds.clusters)} clusters\n")
    counts = np.bincount(ds.cluster_labels, minlength=len(ds.clusters))

    for k, c in enumerate(ds.clusters):
        sizes = c.block_sizes
        print(f"  Cluster {k}  (n={counts[k]}, theta={c.theta:.2f}, "
              f"p={c.p:.2f})")
        print(f"    Block sizes (high->low rank): {sizes}")
        print(f"    T(m)={T(sizes)},  log Z*={log_Z_star(sizes, c.theta, c.p):.3f}")
        for rank, block_idx in enumerate(c.block_order):
            print(f"    Rank {rank+1}: items {sorted(c.blocks[block_idx])}")
        print()


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = generate_mixture_dataset(
        n_items=8,
        n_rankers=300,
        n_clusters=3,
        delta=0.3,
        lam=1.0,
        theta=[2.0, 1.0, 0.5],
        p=[0.3, 0.3, 0.3],
        weights=[0.4, 0.35, 0.25],
        seed=42,
    )

    summarise_dataset(ds)

    print("Average K(p) distances per cluster:")
    for k, c in enumerate(ds.clusters):
        idx = np.where(ds.cluster_labels == k)[0]
        dists = [
            kemeny_distance_strict_vs_weak(
                ds.rankings[i].tolist(), c.ordered_blocks, c.p
            )
            for i in idx
        ]
        print(f"  Cluster {k} (theta={c.theta}): "
              f"mean dist={np.mean(dists):.2f}  "
              f"(T(m) contribution={c.p * T(c.block_sizes):.2f})")
