"""Incremental distance calculation with caching for fast block reassignments.

This module provides delta-based distance updates when only a few items change blocks.
Instead of recalculating inversions for ALL rankings from scratch, we compute only
the difference caused by the changed items.

This is particularly efficient for:
- mh_reassign: Moving ONE item to a different block (O(n) vs O(n log K) per ranking)
- mh_swapshift: Reordering 1-2 blocks (minimal item movement)

The cache stores:
- Previous disagreement count for each ranking
- Current block assignment
- Tracks which items changed
"""

from typing import List, Dict, Optional, Tuple
import time
from .blocks import blocks_to_block_index
from .distance import cross_block_disagreements_fast
from .profiling import get_profiler

try:
    from joblib import Parallel, delayed
    _USE_JOBLIB = True
except ImportError:
    _USE_JOBLIB = False


class AssessmentCache:
    """Per-ranking cache of disagreement counts and block assignments."""
    
    def __init__(self, ranking_id: int):
        self.ranking_id = ranking_id
        self.blocks_tuple: Optional[Tuple] = None  # Immutable representation of blocks
        self.disagreement: float = 0.0
        self.changed_items: set = set()  # Items that changed blocks since last update
    
    def is_valid(self, blocks: List[List[int]]) -> bool:
        """Check if cache is valid for given block structure."""
        return self.blocks_tuple == self._blocks_to_tuple(blocks)
    
    @staticmethod
    def _blocks_to_tuple(blocks: List[List[int]]) -> Tuple:
        """Convert blocks to immutable tuple for comparison."""
        return tuple(tuple(b) for b in blocks)
    
    def invalidate(self):
        """Mark cache as invalid."""
        self.blocks_tuple = None
        self.disagreement = 0.0
        self.changed_items.clear()


class IncrementalDistanceCalculator:
    """Computes incremental distance changes when blocks are modified.
    
    Usage:
        calculator = IncrementalDistanceCalculator(rankings, n_items)
        
        # First call: full computation
        dist = calculator.compute_distance(blocks)
        
        # Subsequent calls with minimal changes: use cached values
        changed_items = {item_id}  # Only item 5 moved blocks
        dist_new = calculator.compute_distance_incremental(blocks_new, changed_items)
    """
    
    def __init__(self, rankings: List[List[int]]):
        """Initialize calculator with rankings."""
        self.rankings = rankings
        self.N = len(rankings)
        self.n = len(rankings[0]) if rankings else 0
        
        # Per-ranking cache
        self.caches: List[AssessmentCache] = [
            AssessmentCache(i) for i in range(self.N)
        ]
        
        # Profiling stats
        self.cache_hits = 0
        self.cache_misses = 0
        self.full_calcs = 0
        self.incremental_calcs = 0
    
    def compute_distance(
        self, 
        blocks: List[List[int]],
        Tm: Optional[int] = None,
        parallel: bool = False,
        n_jobs: int = -1,
        tiePenaltyWeight: float = 1.0
    ) -> int:
        """Compute total distance from scratch (full calculation).
        
        Parameters
        ----------
        blocks : list of lists
            Block structure mapping.
        Tm : int, optional
            Within-block penalty. If None, computed from blocks.
        parallel : bool
            Use parallel computation for rankings (beneficial when N > 100)
        n_jobs : int
            Number of parallel jobs (-1 = all cores)
        tiePenaltyWeight : float
            Multiplier for Tm in distance calculation (default 1.0)
        
        Returns
        -------
        total_distance : int
            Sum of 2*disc_i + tiePenaltyWeight*Tm over all rankings.
        """
        profiler = get_profiler()
        t_start = time.time() if profiler else None
        
        if Tm is None:
            from .blocks import T_of_blocks
            Tm = T_of_blocks(blocks)
        
        weighted_Tm = tiePenaltyWeight * Tm
        
        # Build block index
        block_idx = blocks_to_block_index(blocks, self.n, validate=False)
        K = len(blocks)
        
        if parallel and _USE_JOBLIB and self.N > 50:
            # Parallel computation for large datasets
            def _compute_ranking_distance(ranking, block_idx, K, weighted_Tm):
                disc = cross_block_disagreements_fast(ranking, block_idx, K)
                return 2 * disc + weighted_Tm
            
            distances = Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(_compute_ranking_distance)(ranking, block_idx, K, weighted_Tm)
                for ranking in self.rankings
            )
            total = sum(distances)
            
            # Update caches
            for i, distance in enumerate(distances):
                disc = (distance - weighted_Tm) / 2
                self.caches[i].blocks_tuple = AssessmentCache._blocks_to_tuple(blocks)
                self.caches[i].disagreement = disc
                self.caches[i].changed_items.clear()
        else:
            # Serial computation
            total = 0
            for i, ranking in enumerate(self.rankings):
                disc = cross_block_disagreements_fast(ranking, block_idx, K)
                total += 2 * disc + weighted_Tm
                
                # Update cache
                self.caches[i].blocks_tuple = AssessmentCache._blocks_to_tuple(blocks)
                self.caches[i].disagreement = disc
                self.caches[i].changed_items.clear()
        
        self.full_calcs += 1
        
        if profiler:
            profiler.record_operation(
                "incremental_distance_full_calc", 
                time.time() - t_start
            )
        
        return total
    
    def compute_distance_incremental(
        self,
        blocks_old: List[List[int]],
        blocks_new: List[List[int]],
        changed_items: set,
        Tm_new: int,
        parallel: bool = False,
        n_jobs: int = -1,
        tiePenaltyWeight: float = 1.0
    ) -> int:
        """
        Compute distance using incremental calculation for changed items.
        
        This is more efficient than full recomputation when only a few items
        have moved between blocks.
        
        Parameters
        ----------
        blocks_old : list[list[int]]
            Previous block structure
        blocks_new : list[list[int]]
            New block structure  
        changed_items : set
            Set of item IDs that moved between blocks
        Tm_new : int
            New within-block penalty
        parallel : bool
            Use parallel computation for rankings (beneficial when N > 100)
        n_jobs : int
            Number of parallel jobs (-1 = all cores)
        tiePenaltyWeight : float
            Multiplier for Tm in distance calculation (default 1.0)
        
        Returns
        -------
        total_distance : int
            Updated total distance
        """
        if not changed_items:
            # No items changed, return cached values
            weighted_Tm_new = tiePenaltyWeight * Tm_new
            return sum(
                2 * cache.disagreement + weighted_Tm_new 
                for cache in self.caches
            )
        
        profiler = get_profiler()
        t_start = time.time() if profiler else None
        
        weighted_Tm_new = tiePenaltyWeight * Tm_new
        
        # Build block indices
        block_idx_old = blocks_to_block_index(blocks_old, self.n, validate=False)
        block_idx_new = blocks_to_block_index(blocks_new, self.n, validate=False)
        
        if parallel and _USE_JOBLIB and self.N > 50:
            # Parallel computation for large datasets
            def _compute_ranking_delta(i, ranking):
                old_disc = self.caches[i].disagreement if self.caches[i].is_valid(
                    blocks_old
                ) else self._compute_single_disagreement(ranking, block_idx_old, len(blocks_old))
                
                delta = self._compute_inversion_delta(
                    ranking, block_idx_old, block_idx_new, changed_items
                )
                return old_disc + delta
            
            new_discs = Parallel(n_jobs=n_jobs, backend='threading')(
                delayed(_compute_ranking_delta)(i, ranking)
                for i, ranking in enumerate(self.rankings)
            )
            
            total = sum(2 * d + weighted_Tm_new for d in new_discs)
            
            # Update caches
            for i, new_disc in enumerate(new_discs):
                self.caches[i].blocks_tuple = AssessmentCache._blocks_to_tuple(blocks_new)
                self.caches[i].disagreement = new_disc
                self.caches[i].changed_items.clear()
        else:
            # Serial computation
            total = 0
            
            for i, ranking in enumerate(self.rankings):
                old_disc = self.caches[i].disagreement if self.caches[i].is_valid(
                    blocks_old
                ) else self._compute_single_disagreement(ranking, block_idx_old, len(blocks_old))
                
                # Compute delta from changed items
                delta = self._compute_inversion_delta(
                    ranking, 
                    block_idx_old, 
                    block_idx_new,
                    changed_items
                )
                
                new_disc = old_disc + delta
                total += 2 * new_disc + weighted_Tm_new
                
                # Update cache
                self.caches[i].blocks_tuple = AssessmentCache._blocks_to_tuple(blocks_new)
                self.caches[i].disagreement = new_disc
                self.caches[i].changed_items.clear()
        
        self.incremental_calcs += 1
        
        if profiler:
            profiler.record_operation(
                "incremental_distance_delta_calc",
                time.time() - t_start
            )
        
        return total
    
    def _compute_single_disagreement(
        self,
        ranking: List[int],
        block_idx: List[int],
        K: int
    ) -> int:
        """Compute disagreement for a single ranking."""
        return cross_block_disagreements_fast(ranking, block_idx, K)
    
    def compute_distance_with_heuristic(
        self,
        blocks_old: List[List[int]],
        blocks_new: List[List[int]],
        parallel: bool = False,
        n_jobs: int = -1,
        tiePenaltyWeight: float = 1.0
    ) -> int:
        """
        Intelligently choose between incremental and full calculation.
        
        Decision logic:
        - If blocks_old is None: use full calculation
        - If >30% items changed: use full calculation (overhead not worth it)
        - Otherwise: use incremental calculation
        
        Parameters
        ----------
        blocks_old : list[list[int]]
            Previous block structure
        blocks_new : list[list[int]]
            New block structure
        parallel : bool
            Use parallel computation when beneficial
        n_jobs : int
            Number of parallel jobs
        tiePenaltyWeight : float
            Multiplier for Tm in distance calculation (default 1.0)
        
        Returns
        -------
        total_distance : int
        """
        if blocks_old is None:
            from .blocks import T_of_blocks
            return self.compute_distance(blocks_new, T_of_blocks(blocks_new), tiePenaltyWeight=tiePenaltyWeight)
        
        # Compute changed items
        from .blocks import blocks_to_block_index
        n = self.n
        
        old_idx = blocks_to_block_index(blocks_old, n, validate=False)
        new_idx = blocks_to_block_index(blocks_new, n, validate=False)
        
        changed_items = set()
        for item in range(n):
            if old_idx[item] != new_idx[item]:
                changed_items.add(item)
        
        pct_changed = len(changed_items) / n if n > 0 else 0
        
        if pct_changed > 0.3:
            # Too many changes, do full recompute
            from .blocks import T_of_blocks
            return self.compute_distance(blocks_new, T_of_blocks(blocks_new), parallel=parallel, n_jobs=n_jobs, tiePenaltyWeight=tiePenaltyWeight)
        
        # Use incremental
        from .blocks import T_of_blocks
        return self.compute_distance_incremental(
            blocks_old, blocks_new, changed_items,
            T_of_blocks(blocks_new), parallel=parallel, n_jobs=n_jobs, tiePenaltyWeight=tiePenaltyWeight
        )
    
    def _compute_inversion_delta(
        self,
        ranking: List[int],
        block_idx_old: List[int],
        block_idx_new: List[int],
        changed_items: set
    ) -> int:
        """
        Compute change in inversion count when only certain items change blocks.
        
        For each item that moved blocks, count how many inversions it now has
        with all other items in the ranking.
        
        Time complexity: O(|changed_items| * n)
        vs full inversion count: O(n log K)
        
        Benefit: When |changed_items| << n, this is much faster.
        """
        delta = 0
        
        for i, item_i in enumerate(ranking):
            if item_i not in changed_items:
                continue
            
            old_block = block_idx_old[item_i]
            new_block = block_idx_new[item_i]
            
            # Skip if item stayed in same block
            if old_block == new_block:
                continue
            
            # Count inversions with all other items
            for j, item_j in enumerate(ranking):
                if i >= j or item_j == item_i:
                    continue
                
                old_blocks_differ = block_idx_old[item_i] != block_idx_old[item_j]
                new_blocks_differ = block_idx_new[item_i] != block_idx_new[item_j]
                
                # Change in this pair's inversion status
                if old_blocks_differ and not new_blocks_differ:
                    # Was an inversion, no longer is
                    delta -= 1
                elif not old_blocks_differ and new_blocks_differ:
                    # Wasn't an inversion, now is
                    delta += 1
        
        return delta
    
    def get_stats(self) -> Dict[str, any]:
        """Return caching statistics for profiling."""
        total_calcs = self.full_calcs + self.incremental_calcs
        return {
            "full_calculations": self.full_calcs,
            "incremental_calculations": self.incremental_calcs,
            "total_distance_calcs": total_calcs,
            "pct_incremental": (
                100.0 * self.incremental_calcs / total_calcs
                if total_calcs > 0 else 0.0
            ),
        }
    
    def reset_stats(self):
        """Reset profiling statistics."""
        self.cache_hits = 0
        self.cache_misses = 0
        self.full_calcs = 0
        self.incremental_calcs = 0

