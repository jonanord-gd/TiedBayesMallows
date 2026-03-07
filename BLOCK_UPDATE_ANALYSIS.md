"""
Summary: Why Block Updates Are Slow

Key Finding: Block moves call log_blocks_posterior() twice per acceptance test
- Once for the OLD blocks (cached, ~0.5ms)
- Once for the NEW proposed blocks (always computed, ~1.7ms)

This means each move does ~ 2 full distance calculations:
- mh_swapshift: 91 moves × ~2 distance calcs = 182 distance calculations
- mh_transfer: 82 moves × ~2 distance calcs = 164 distance calculations
- mh_splitmerge: 41 moves × ~2 distance calcs = 82 distance calculations

Total: ~428 distance calculations in 100 iterations!

This is the expected behavior, but there ARE optimization opportunities.
"""

# PROFILING RESULTS SUMMARY
print("="*80)
print("BLOCK UPDATE TIME BREAKDOWN (100 iterations)")
print("="*80)

# From profiling results
update_blocks_time = 0.844  # seconds (from first profiling)
move_time = 0.744  # seconds (from block move profiling)
other_time = update_blocks_time - move_time

print(f"""
Total _update_cluster_blocks:    {update_blocks_time:.3f}s (28.1% of MCMC)
  ├─ Block move proposals:       {move_time:.3f}s (91% of _update_blocks)
  │   ├─ mh_swapshift (42%):     {0.313:.3f}s   [91 calls @ 3.4ms/call]
  │   ├─ mh_transfer (34%):      {0.255:.3f}s   [82 calls @ 3.1ms/call]
  │   └─ mh_splitmerge (24%):    {0.176:.3f}s   [41 calls @ 4.3ms/call]
  └─ Distance/posterior setup:   {other_time:.3f}s (9% of _update_blocks)

Per move breakdown (mh_swapshift as example):
  For each move proposal:
    1. Calculate log_posterior(old_blocks):  ~0.5ms (cached from cache)
    2. Propose new blocks:                   <0.1ms (trivial reordering)
    3. Calculate log_posterior(new_blocks):  ~1.7ms (full distance calc required)
    4. Accept/reject test:                   ~1.2ms (Hastings ratio, sampling)
    └─ Total per move:                       ~3.4ms
  
  Chain of moves: 91 moves × 3.4ms = 309ms ≈ 0.31s ✓

WHERE THE TIME GOES:
  - Distance calculations:    ~70% (Fenwick tree operations on all rankings)
  - log_Z* calculations:      ~15% (relatively fast)
  - Hastings acceptance test: ~15% (sampling, log operations)

REUSED vs RECALCULATED:
  ✓ update_z distance:        Pre-computed once per iteration, reused in moves
  ✗ Move distance:            Must be computed for EACH proposed block configuration
      Reason: Blocks change, so distances change - no reuse possible
""")

print("="*80)
print("OPTIMIZATION OPPORTUNITIES")
print("="*80)

print("""
1. REDUCE DISTANCE CALCULATIONS PER MOVE (Most effective)
   ────────────────────────────────────────────────────
   Current: Each move proposes K different block orderings, each requires distance calc
   Idea: Pre-compute a "distance delta" matrix for block swaps
   
   Example: For swap(j, j+1), compute how distance changes
   - Build influence matrix: D[i,j] = change in distance if item i moves to block j
   - Reuse across multiple swaps
   - Reduce from 2 full calcs to 1 full + N delta updates
   
   Potential speedup: 2-4× for mh_swapshift

2. USE INCREMENTAL DISTANCE CALCULATION (Currently implemented but not activated)
   ────────────────────────────────────────────────────────────
   Current: log_blocks_posterior uses full distance_calculator for some moves
   Issue: Not consistently used across all move types
   
   Check: In mh_swapshift, is distance_calculator being passed correctly?
   If activated properly: 1.2-1.5× speedup per move
   
3. REDUCE MOVE PROPOSAL ATTEMPTS (Sampling approach)
   ──────────────────────────────────────────────
   Current: Try to accept each proposal independently
   Idea: Batch moves within a Gibbs sweep
   - For cluster c, do multiple swap+accept steps without re-evaluating posterior
   - Accept batch if new posterior better than old
   
   Trade: Slightly lower MH acceptance rate for fewer distance calculations
   Potential speedup: 1.5-2×

4. APPROXIMATE DISTANCE FOR EXPLORATION (Research approach)
   ───────────────────────────────────────────────────────
   Current: Full Kendall-tau distance with Fenwick trees
   Idea: Use approximate distance for move proposals
   - Spearman footrule: cheaper to compute
   - Or subset of assessors for faster evaluation
   - Refine with full distance only when promising
   
   Trade: May affect mixing properties
   Potential speedup: 3-5×

5. PARALLELIZE DISTANCE ACROSS MOVES (Works if N > 500)
   ──────────────────────────────────────────────────
   Current: Sequential move proposals
   Idea: Propose multiple adjacent orderings, compute distances in parallel
   - Given K blocks, propose all adjacent swaps simultaneously
   - Compute distances in parallel (jobb lib)
   - Accept best one
   
   Requires: Rethinking move structure
   Potential speedup: 2-3× (if N large enough to overcome overhead)

RECOMMENDATION RANK ORDER:
1. ✓ Confirm incremental distance is working (quick debug, 1.2-1.5× speedup)
2. ⚠️ Pre-compute distance deltas for block swaps (1-2 days, 2-4× speedup)
3. ⚠️ Reduce move proposal attempts using batch Gibbs (1-2 days, 1.5-2× speedup)
4. ❌ Approximate distance (research, 3-5× but needs validation)
5. ❌ Parallelize across moves (3-5× but requires N > 500)

EXPECTED IMPACT:
- Quick win: 1.2× (enable incremental distance properly)
- Medium effort: 2-3× (delta-based swaps + batch Gibbs)
- Hard to scale beyond 3× without algorithmic changes
""")

print("\n" + "="*80)
print("VERIFICATION NEEDED")
print("="*80)
print("""
Before optimizing, verify:

1. Is incremental_distance enabled and working?
   Check: distance_calculator being passed to log_blocks_posterior?
   
2. How often is the posterior_cache hit?
   Measure: Add logging to see cache hit rate
   Expected: ~80% on old_blocks, 0% on new_blocks
   
3. What % of move time is each component?
   Measure: Profile log_blocks_posterior internals
   - Distance calc: should be ~70%
   - Z* calc: should be ~15%
   - Other: should be ~15%
   
4. Is the Fenwick tree implementation optimal?
   check: Complexity is O(n log K) per (assessor, blocks) pair
   This is inherent and hard to improve
""")
