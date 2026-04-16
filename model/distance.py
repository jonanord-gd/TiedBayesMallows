"""Efficient distance and inversion-count routines."""

from typing import List, Callable
import time

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
) -> int:
    """Compute sum_i disc_i (cross-block disagreements) via inversion count."""
    total = 0
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
        total += inv
    return total


def total_distance_fast(rankings: List[List[int]], blocks: List[List[int]]) -> int:
    """Sum_i d(r_i, blocks). Uses inversion counting (O(N n log K)).

    Returns the total cross-block disagreements across all rankings.
    The tie penalty p·Tm term that appears in the extended Kendall
    distance cancels analytically with the corresponding term in Z*,
    so it is not computed.
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

    if _USE_NUMBA:
        if profiler:
            t_start = time.time()
        total = 0
        for r in rankings:
            inv = _cross_block_disagreements_fast_nb(r, blk, K)
            total += inv
        if profiler:
            profiler.record_operation("inversion_counting_all_rankings", time.time() - t_start)
        return total

    # Profile inversion counting
    if profiler:
        t_start = time.time()
    total = 0
    for r in rankings:
        disc = cross_block_disagreements_fast(r, blk, K)
        total += disc
    if profiler:
        profiler.record_operation("inversion_counting_all_rankings", time.time() - t_start)
    return total
