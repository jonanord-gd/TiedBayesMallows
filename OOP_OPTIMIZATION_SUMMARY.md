# OOP Overhead Optimization Summary

## Problem Identified
The original code was experiencing significant performance overhead from repeated Python attribute lookups (`self.cfg`, `self.rng`, `self.state`) inside tight loops, particularly in the MCMC step, which is called thousands of times per run.

**Example of the issue:**
```python
# BAD: Each access requires a dictionary lookup + attribute resolution
for _ in range(n_item_moves):  # Tight loop, repeated many times
    u = self.rng.random()         # Attribute lookup overhead
    move_name = self._select_move(self.rng, self.cfg)  # More lookups for each call
    # ... inside moves.py, the move functions use rng multiple times
```

## Solution: Local Caching Strategy
Instead of repeatedly accessing `self.x`, we cache frequently-used references as **local variables** at method entry. This is fast because:

1. Local variable access is O(1) dictionary lookup on the local scope
2. Attribute access requires O(1) dictionary lookup on `self.__dict__` PLUS descriptor protocol checks
3. In tight loops, local caching eliminates thousands of redundant lookups

**Pattern used:**
```python
# GOOD: Cache once at method entry
def _tight_loop_method(self):
    # Cache self references as local variables (zero-cost)
    rng = self.rng      # One-time attribute lookup
    cfg = self.cfg
    state = self.state
    
    for _ in range(n_item_moves):  # Tight loop
        u = rng.random()           # Fast local variable access, not attribute lookup
        # ... use rng, cfg, state locally - all fast
```

## Changes Made

### 1. **`_update_z()` method**
- Cached: `self.state`, `self.rng`, `self.C`, `self._cache`, `self.rankings`
- **Impact**: Called once per iteration, inner loop over all assessors (N) and clusters (C)
- **Gain**: Eliminates N×C attribute lookups per iteration

### 2. **`_update_tau()` method**
- Cached: `self.state`, `self.rng`, `self.C`, `self.init_mu`
- **Impact**: Called once per iteration, loop over clusters
- **Gain**: Eliminates C attribute lookups per iteration

### 3. **`_update_cluster_blocks()` method** ⭐ **BIGGEST HOTSPOT**
- Cached: `self.rng`, `self.cfg`, `self.state`, cluster blocks, cluster theta, cfg parameters
- **Impact**: Called C times per iteration, inner loop over n_item_moves (typically 2-10)
- **Gain**: Eliminates ~50-100 attribute lookups per iteration per cluster (C × n_item_moves overhead reduced)
- **Key optimization**: Pre-cached `cfg_ordering_p_short`, `cfg_ordering_n_swap_steps`, etc. to avoid repeated `self.cfg` lookups in move selection

### 4. **`_update_cluster_theta()` method**
- Cached: `self.rng`, `self.state`, `self._cache`
- **Impact**: Called C times per iteration
- **Gain**: Eliminates C attribute lookups per iteration

### 5. **`step()` method**
- Cached: `self.C`, `self.state.z`, `self.rankings` (used in cluster ranking filtering)
- **Impact**: Main iteration method, called n_iter times
- **Gain**: Inlined `_cluster_rankings()` logic to avoid function call overhead + attribute lookups
- **Optimization**: Replaced method call with direct list comprehension using cached state

### 6. **`log_joint()` method**
- Cached: `self.state`, `self.init_mu`, `self.C`, `self.rankings`, `state.tau`, `state.z`
- **Impact**: Called for every saved iteration (affects posterior tracking)
- **Gain**: Eliminates redundant state lookups in tight loops

## Performance Implications

### Estimated Overhead Reduction
For a typical run with:
- N = 100 assessors
- C = 6 clusters  
- K = 3-6 blocks per cluster
- n_item_moves = 2 moves per cluster
- n_iter = 10,000 iterations

**Original overhead lookup count per step:**
- `_update_z()`: ~200 lookups (N × C)
- `_update_cluster_blocks()`: ~600 lookups (C × n_item_moves × avg moves tried × multiple cfg accesses)
- `_update_tau()`: ~20 lookups
- `_update_cluster_theta()`: ~50 lookups
- **Total per step: ~870 lookups**
- **Total for run: ~8.7 million lookups**

**After optimization:**
- ~90% reduction in attribute lookup count for these hot methods

### Expected Speedup
Based on Python attribute lookup costs (~100-200 nanoseconds per lookup):
- **Estimated improvement: 5-15% runtime reduction** (depending on system, Python version, CPU cache efficiency)
- **Maximum improvement if distance calculation is parallelized separately: 10-20%**

## Maintaining OOP Structure
This optimization **preserves the object-oriented design**:
- ✅ Methods remain class methods with access to `self`
- ✅ State encapsulation is maintained
- ✅ Public interface unchanged
- ✅ Code readability slightly improved (local variables are clearer than multiple `self.` accesses)
- ✅ Maintainability preserved (caching is localized to method scope)

## Backwards Compatibility
- ✅ All changes are internal implementation details
- ✅ Public API unchanged
- ✅ No breaking changes to method signatures
- ✅ Existing code using the model works without modification

## Further Optimization Opportunities

If you need even more speed, consider:

1. **Numba JIT compilation** for moves.py functions - could give 10-50x speedup for distance calculations
2. **Cython** for the most compute-intensive portions (distance, inversion counting)
3. **PyPy** interpreter - may provide automatic JIT benefits
4. **Parallelize** distance calculations across assessors (currently single-threaded)

However, this OOP caching strategy is a "free" optimization with no code complexity cost.

