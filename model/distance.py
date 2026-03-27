"""Efficient distance and inversion-count routines."""

from typing import List, Callable
import time

from .blocks import T_of_sizes
from .utils import logsumexp  # possibly unused here but convenient for testing


class Fenwick:
    __slots__ = ("n", "bit")

    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)

    def add(self, i: int, delta: int) -> None:
        i += 1
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i

    def sum_prefix(self, i: int) -> int:
        i += 1
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s


# numba-accelerated inversion count if available
try:
    from numba import njit
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False


if _USE_NUMBA:
    @njit
    def _cross_block_disagreements_fast_nb(strict_r, block_idx, K):
        # Fenwick tree implemented with local array
        fw = [0] * (K + 1)
        seen = 0
        inv = 0
        for idx in range(len(strict_r)):
            item = strict_r[idx]
            b = block_idx[item]
            # prefix sum - query 0 to b (inclusive)
            s = 0
            i = b + 1
            while i > 0:  # FIX: Descend Fenwick tree by subtracting i & -i
                s += fw[i]
                i -= i & -i
            inv += seen - s
            # add b to Fenwick tree
            i = b + 1
            while i <= K:
                fw[i] += 1
                i += i & -i
            seen += 1
        return inv


def cross_block_disagreements_fast(strict_r: List[int], block_idx: List[int], K: int) -> int:
    """# inversions in block-index sequence along strict_r."""
    if _USE_NUMBA:
        # convert to simple lists for Numba
        return _cross_block_disagreements_fast_nb(strict_r, block_idx, K)
    fw = Fenwick(K)
    seen = 0
    inv = 0
    for item in strict_r:
        b = block_idx[item]
        leq = fw.sum_prefix(b)
        inv += seen - leq
        fw.add(b, 1)
        seen += 1
    return inv


def total_distance_given_block_index(
    rankings: List[List[int]],
    block_index_fn: Callable[[int], int],
    K: int,
    Tm: int,
    tie_penalty: float = 0.5
) -> int:
    """Compute sum_i (disc_i + tie_penalty*Tm) with disc_i via inversion count."""
    total = 0
    weighted_Tm = tie_penalty * Tm
    for r in rankings:
        fw = Fenwick(K)
        seen = 0
        inv = 0
        for item in r:
            b = block_index_fn(item)
            leq = fw.sum_prefix(b)
            inv += seen - leq
            fw.add(b, 1)
            seen += 1
        total += inv + weighted_Tm
    return total


def total_distance_fast(rankings: List[List[int]], blocks: List[List[int]], tie_penalty: float = 0.5) -> int:
    """Sum_i d(r_i, blocks). Uses inversion counting (O(N n log K)).

    A numba-accelerated variant is used when available, which speeds the
    per-ranking loop substantially.  The logic is otherwise identical to
    the original implementation.
    
    Distance computation breakdown:
    - Block index creation: Maps items to block IDs (O(n))
    - Per-ranking inversion counting: For each ranking, count disagreements using Fenwick tree (O(n log K) * N)
    - Tm penalty: Within-block penalty term (O(K)) scaled by tie_penalty
    
    Parameters
    ----------
    rankings : list of lists
        Rankings to compute distance for
    blocks : list of lists  
        Block structure
    tie_penalty : float, default=0.5
        Weight for the within-block penalty term (Tm). The p in K^(p) extended Kendall distance
        (p=0.5 recovers Kemeny distance). Must match the weight used in the likelihood calculation
        for consistency. Distance formula: sum_i (inversions_i + tie_penalty*Tm).
    """
    if not rankings:
        return 0
    
    from .profiling import get_profiler
    profiler = get_profiler()
    
    n = len(rankings[0])
    sizes = [len(b) for b in blocks]
    K = len(sizes)
    from .blocks import blocks_to_block_index

    # Profile block index creation
    if profiler:
        t_start = time.time()
    blk = blocks_to_block_index(blocks, n, validate=False)
    if profiler:
        profiler.record_operation("block_index_creation", time.time() - t_start)
    
    # Profile Tm calculation
    if profiler:
        t_start = time.time()
    Tm = T_of_sizes(sizes)
    weighted_Tm = tie_penalty * Tm  # Apply weight to within-block penalty
    if profiler:
        profiler.record_operation("within_block_penalty_calc", time.time() - t_start)

    if _USE_NUMBA:
        # convert python lists to simple arrays for njit call
        # numba supports typed List; easiest is to use same cross-count function
        if profiler:
            t_start = time.time()
        total = 0
        for r in rankings:
            inv = _cross_block_disagreements_fast_nb(r, blk, K)
            total += inv + weighted_Tm
        if profiler:
            profiler.record_operation("inversion_counting_all_rankings", time.time() - t_start)
        return total

    # Profile inversion counting
    if profiler:
        t_start = time.time()
    total = 0
    for r in rankings:
        disc = cross_block_disagreements_fast(r, blk, K)
        total += disc + weighted_Tm
    if profiler:
        profiler.record_operation("inversion_counting_all_rankings", time.time() - t_start)
    return total
