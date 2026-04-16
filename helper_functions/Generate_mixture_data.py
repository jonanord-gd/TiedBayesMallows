"""
Mallows model with tied (block) consensus – samples strict linear orders.

Model
-----
    P(σ | ρ₀, α) ∝ exp(−α · d_K(σ, ρ₀))

where σ is a strict ranking, ρ₀ is a consensus with ties (given as an
ordered list of blocks), α ≥ 0 is the dispersion parameter, and d_K is
the Kendall-tau distance generalized for ties:

  * A pair (i,j) where ρ₀ strictly orders i above j contributes 1 if σ
    reverses them, 0 otherwise.
  * A pair (i,j) tied in ρ₀ contributes 0 regardless of σ.

Sampling Strategy
-----------------
Adjacent-transposition Gibbs sampler with O(1) incremental distance updates.
At each step we pick a random adjacent pair (σ[i], σ[i+1]) and decide whether
to swap using only the delta in Kendall distance (not recomputing full distance):

  * O(1) check: if pair is from same block (tied), swap with prob 0.5
  * O(1) check: if swapping fixes disagreement (Δ = −1), always swap
  * O(1) check: if swapping creates disagreement (Δ = +1), swap with prob exp(−α)

This satisfies detailed balance & is ergodic. Key efficiency: avoiding O(n²)
full distance recomputation at each step.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Sequence
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helper: Build block mapping (pre-compute once, reuse)
# ---------------------------------------------------------------------------

def _build_block_mapping(blocks: list[list[int]], n_items: int | None = None) -> list[int]:
    """Build an item-to-block list once and reuse it throughout sampling."""
    if n_items is None:
        n_items = sum(len(block) for block in blocks)

    block_of = [0] * n_items
    for b_idx, block in enumerate(blocks):
        for item in block:
            block_of[item] = b_idx
    return block_of


# ---------------------------------------------------------------------------
# Kendall distance (diagnostic only, O(n²))
# ---------------------------------------------------------------------------

def kendall_distance_tied(
    sigma: Sequence[int],
    block_of: Sequence[int],
) -> int:
    """
    Kendall-tau distance between strict ranking σ and tied consensus ρ₀.
    
    Note: O(n²) computation. Used for diagnostics/validation only.
    During sampling, we use incremental O(1) updates via _swap_delta instead.
    
    Parameters
    ----------
    sigma : sequence of int
        Strict ranking (permutation).
    block_of : dict[int, int]
        Item-to-block mapping (typically pre-computed once per cluster).
    
    Returns
    -------
    int
        Kendall-tau distance.
    """
    block_idx = list(block_of)
    pos_in_sigma = [0] * len(sigma)
    for pos, item in enumerate(sigma):
        pos_in_sigma[item] = pos

    dist = 0
    n = len(block_idx)
    for i in range(n):
        bi = block_idx[i]
        pos_i = pos_in_sigma[i]
        for j in range(i + 1, n):
            # Tied pairs contribute 0 to distance
            bj = block_idx[j]
            if bi == bj:
                continue
            # Check if discordant
            pos_j = pos_in_sigma[j]
            if bi < bj:
                if pos_i > pos_j:
                    dist += 1
            else:
                if pos_j > pos_i:
                    dist += 1
    return dist


# ---------------------------------------------------------------------------
# Gibbs sampler: O(1) per step via incremental distance
# ---------------------------------------------------------------------------

def _swap_delta(sigma_blocks: list[int], pos: int) -> int:
    """
    Change in Kendall distance from swapping sigma[pos] ↔ sigma[pos+1].
    
    O(1) operation: only check the single adjacent pair, not full distance.
    
    Returns
    -------
    int
        0: tied pair (no distance contribution)
        -1: swap improves (fixes disagreement)
        +1: swap worsens (creates disagreement)
    """
    bi = sigma_blocks[pos]
    bj = sigma_blocks[pos + 1]
    if bi == bj:
        # Tied pair—contributes 0 regardless of order
        return 0
    # Discordant pair: check ordering
    return 1 if bi < bj else -1


def _gibbs_step(
    sigma: list[int],
    sigma_blocks: list[int],
    worsen_prob: float,
    rng: np.random.Generator,
) -> None:
    """
    One Gibbs step: pick random adjacent pair, decide whether to swap (in-place).
    
    O(1) per step using incremental _swap_delta computation.
    """
    n_minus_1 = len(sigma) - 1
    if n_minus_1 < 1:
        return
    pos = int(rng.integers(0, n_minus_1))
    delta = _swap_delta(sigma_blocks, pos)
    if delta <= 0:
        # Tied pair or improves: always swap
        sigma[pos], sigma[pos + 1] = sigma[pos + 1], sigma[pos]
        sigma_blocks[pos], sigma_blocks[pos + 1] = sigma_blocks[pos + 1], sigma_blocks[pos]
    else:
        # Worsens: swap with prob exp(-α)
        if rng.random() < worsen_prob:
            sigma[pos], sigma[pos + 1] = sigma[pos + 1], sigma[pos]
            sigma_blocks[pos], sigma_blocks[pos + 1] = sigma_blocks[pos + 1], sigma_blocks[pos]


def _run_gibbs_steps(
    sigma: list[int],
    sigma_blocks: list[int],
    worsen_prob: float,
    rng: np.random.Generator,
    n_steps: int,
) -> None:
    """Run many adjacent-swap Gibbs steps with cached local references."""
    n_minus_1 = len(sigma) - 1
    if n_minus_1 < 1 or n_steps <= 0:
        return

    batch_size = min(4096, n_steps)
    remaining = n_steps
    while remaining > 0:
        size = batch_size if remaining >= batch_size else remaining
        positions = rng.integers(0, n_minus_1, size=size)
        uniforms = rng.random(size=size)
        for pos, u in zip(positions, uniforms):
            pos = int(pos)
            left_block = sigma_blocks[pos]
            right_block = sigma_blocks[pos + 1]
            if left_block >= right_block or u < worsen_prob:
                sigma[pos], sigma[pos + 1] = sigma[pos + 1], sigma[pos]
                sigma_blocks[pos], sigma_blocks[pos + 1] = sigma_blocks[pos + 1], sigma_blocks[pos]
        remaining -= size


def _initial_sigma(
    blocks: list[list[int]],
    rng: np.random.Generator,
    block_of: Sequence[int],
) -> tuple[list[int], list[int]]:
    """Initialize one chain ordered by blocks and shuffled within each block."""
    sigma: list[int] = []
    sigma_blocks: list[int] = []
    for block in blocks:
        shuffled = list(block)
        rng.shuffle(shuffled)
        sigma.extend(shuffled)
        sigma_blocks.extend(block_of[item] for item in shuffled)
    return sigma, sigma_blocks


def sample_mallows_tied(
    blocks: list[list[int]],
    theta: float,
    rng: np.random.Generator,
    *,
    n_samples: int = 1,
    burn_in: int | None = None,
    thin: int | None = None,
) -> list[int] | list[list[int]]:
    """
    Sample strict ranking(s) from Mallows with tied consensus.
    
    Uses adjacent-transposition Gibbs sampling where each step is O(1)
    via incremental distance computation (avoiding O(n²) recomputation).

    Parameters
    ----------
    blocks : list[list[int]]
        Consensus ρ₀ as ordered blocks (best-first).
    theta : float
        Dispersion parameter (≥ 0).
    rng : numpy.random.Generator
    n_samples : int
        How many rankings to draw.
    burn_in : int or None
        MCMC burn-in steps.  Default: 10·n².
    thin : int or None
        Steps between samples.  Default: 5·n².

    Returns
    -------
    list[int] if n_samples == 1, else list[list[int]]
        Sampled ranking(s).
    """
    if theta < 0:
        raise ValueError("theta must be >= 0")

    n = sum(len(b) for b in blocks)
    if n == 0:
        return [] if n_samples == 1 else [[] for _ in range(n_samples)]

    # Pre-compute block mapping once (reused throughout all steps)
    block_of = _build_block_mapping(blocks, n)

    n2 = n * n
    if burn_in is None:
        burn_in = 10 * n2
    if thin is None:
        thin = 5 * n2
    worsen_prob = math.exp(-theta)

    # Initialize: sort by block, shuffle within each block
    sigma, sigma_blocks = _initial_sigma(blocks, rng, block_of)

    # Burn-in phase: discard these samples
    _run_gibbs_steps(sigma, sigma_blocks, worsen_prob, rng, burn_in)

    # Collect samples
    samples: list[list[int]] = []
    for s in range(n_samples):
        if s > 0:
            # Thin: wait 'thin' steps between samples for independence
            _run_gibbs_steps(sigma, sigma_blocks, worsen_prob, rng, thin)
        samples.append(list(sigma))

    if n_samples == 1:
        return samples[0]
    return samples




# ---------------------------------------------------------------------------
# Batch sampler: efficiently draw many samples from single chain
# ---------------------------------------------------------------------------

def sample_mallows_tied_batch(
    blocks: list[list[int]],
    theta: float,
    rng: np.random.Generator,
    n_samples: int,
    burn_in: int | None = None,
    thin: int | None = None,
) -> list[list[int]]:
    """
    Efficiently draw n_samples rankings from a single Markov chain.
    
    More efficient than calling sample_mallows_tied(n_samples=1) repeatedly,
    as it reuses the chain state and performs burn-in only once.

    Parameters
    ----------
    blocks : list[list[int]]
        Consensus blocks.
    theta : float
        Dispersion parameter.
    rng : numpy.random.Generator
    n_samples : int
        Number of samples to draw.
    burn_in, thin : int or None
        MCMC tuning parameters.

    Returns
    -------
    list[list[int]]
        Always returns list of lists (even if n_samples=1), for consistency.
    """
    result = sample_mallows_tied(
        blocks, theta, rng,
        n_samples=n_samples, burn_in=burn_in, thin=thin,
    )
    if n_samples == 1:
        return [result]
    return result


# ---------------------------------------------------------------------------
# Full data generator: mixture of Mallows with tied consensus
# ---------------------------------------------------------------------------

def generate_mixture_data(
    n_assessors: int = 100,
    n_items: int = 10,
    C: int = 3,
    seed: int = 1,
    theta: float = 3.0,
    tau: np.ndarray | None = None,
    n_blocks: int | None = None,
    n_blocks_range: tuple[int, int] | None = None,
    block_density: float | None = None,
    min_block_size: int = 1,
    burn_in: int | None = None,
    thin: int | None = None,
) -> tuple[list[list[list[int]]], list[float], list[int], list[list[int]]]:
    """
    Generate a mixture of Mallows models with tied consensus rankings.
    
    Efficiency: Uses batch sampling grouped by cluster to share MCMC chains
    within each cluster, avoiding redundant warm-up.

    Each cluster c has a tied consensus ρ₀ᶜ and shared dispersion α.
    Each assessor is assigned to a cluster via τ, then their strict
    ranking is sampled from Mallows(ρ₀ᶜ, α).

    Parameters
    ----------
    n_assessors : int
        Number of assessors.
    n_items : int
        Number of items.
    C : int
        Number of clusters.
    seed : int
        Random seed.
    theta : float
        Mallows dispersion.  Higher → closer to consensus.
    tau : array-like or None
        Mixture weights (length C).  None → uniform.
    n_blocks : int or None
        Fixed number of blocks per cluster. If None, sampled randomly.
    n_blocks_range : tuple[int, int]
        (min, max) number of blocks to sample per cluster (if n_blocks=None).
    min_block_size : int
        Minimum size of each block.
    burn_in, thin : int or None
        MCMC parameters.  None → automatic (10n² and 5n²).

    Returns
    -------
    tuple
        (true_blocks, tau, z_true, rankings)
        - true_blocks[c] = list of blocks defining consensus for cluster c
        - tau = mixture weights (normalized)
        - z_true = cluster assignment (one per assessor)
        - rankings = sampled strict rankings (permutation per assessor)
    """
    rng = np.random.default_rng(seed)

    if n_items < 1:
        raise ValueError("n_items must be >= 1")
    if C < 1:
        raise ValueError("C must be >= 1")
    if min_block_size < 1:
        raise ValueError("min_block_size must be >= 1")

    # ---- Normalize mixture weights ----
    if tau is None:
        tau_arr = np.ones(C) / C
    else:
        tau_arr = np.asarray(tau, dtype=float)
        if tau_arr.shape != (C,):
            raise ValueError(f"tau must have length C={C}")
        if tau_arr.sum() <= 0:
            raise ValueError("tau must sum to a positive value")
        tau_arr = tau_arr / tau_arr.sum()

    if n_blocks is None and n_blocks_range is None:
        _density = block_density if block_density is not None else 0.4
        n_blocks = max(1, min(n_items, round(_density * n_items)))

    # ---- Helper functions for block generation ----
    def choose_k() -> int:
        """Choose number of blocks for this cluster."""
        if n_blocks is not None:
            k = int(n_blocks)
        else:
            lo, hi = n_blocks_range
            if lo < 1 or hi < lo:
                raise ValueError("n_blocks_range must be (low>=1, high>=low)")
            k = int(rng.integers(lo, hi + 1))
        # Cap k so each block can have at least min_block_size items
        k_max = n_items // min_block_size
        if k_max < 1:
            return 1
        return max(1, min(k, k_max))

    def sizes_no_empty(total: int, k: int, min_part: int) -> list[int]:
        """Allocate 'total' items into k blocks, each with >= min_part items."""
        base = np.full(k, min_part, dtype=int)
        remaining = total - k * min_part
        if remaining < 0:
            raise ValueError(
                f"Impossible: total={total}, k={k}, min_part={min_part}"
            )
        # Randomly distribute remaining items
        extras = rng.multinomial(remaining, np.ones(k) / k)
        sizes = base + extras
        rng.shuffle(sizes)
        return sizes.tolist()

    def make_cluster_blocks() -> list[list[int]]:
        """Generate one random block partition for a cluster."""
        k = choose_k()
        sizes = sizes_no_empty(n_items, k, min_block_size)
        items = rng.permutation(n_items).tolist()
        blocks_list: list[list[int]] = []
        idx = 0
        for sz in sizes:
            block = items[idx : idx + sz]
            assert len(block) > 0, "Empty block should be impossible"
            blocks_list.append(block)
            idx += sz
        assert idx == n_items, "Did not consume all items"
        assert len({x for b in blocks_list for x in b}) == n_items, \
            "Items repeated or missing"
        return blocks_list

    # ---- Generate block structures and cluster assignments ----
    true_blocks = [make_cluster_blocks() for _ in range(C)]
    z_true = rng.choice(C, size=n_assessors, p=tau_arr).tolist()

    # ---- Efficient batch sampling: group assessors by cluster ----
    # Within each cluster, sample from a single chain to avoid redundant burn-in
    cluster_indices: dict[int, list[int]] = defaultdict(list)
    for i, z in enumerate(z_true):
        cluster_indices[z].append(i)

    rankings: list[list[int] | None] = [None] * n_assessors
    for c, indices in cluster_indices.items():
        # Sample len(indices) rankings from one chain for cluster c
        sampled = sample_mallows_tied_batch(
            true_blocks[c], theta, rng,
            n_samples=len(indices),
            burn_in=burn_in, thin=thin,
        )
        for idx, s in zip(indices, sampled):
            rankings[idx] = s

    return true_blocks, tau_arr.tolist(), z_true, rankings  # type: ignore
