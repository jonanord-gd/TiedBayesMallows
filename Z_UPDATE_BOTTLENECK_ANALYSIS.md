# Cluster Reassignment (_update_z) Bottleneck Analysis

## Quick Summary

**_update_z consumes 33-42% of total MCMC time because it has O(N×C×n) computational complexity with unavoidable overhead.**

It's not a bug—it's fundamental to the algorithm. Here's why and what can be done about it.

---

## The Cost Breakdown

### What Happens in _update_z?

For each assessor (N times):
```
for i in 0..N-1:
    for c in 0..C-1:
        # This is the expensive part:
        disagreements = cross_block_disagreements_fast(ranking[i], blocks[c])
        logw[c] = log_tau[c] - theta * disagreements - logZ[c]
    
    # Sample assignement from logweights (fast, O(C))
    z[i] = sample_categorical(logw)
```

### Measured Costs (10 items, 30 assessors, 3 clusters)

```
_update_z per iteration:     1.75ms
Distance calculations:       90 (30 assessors × 3 clusters)
Fenwick tree calls:          900 (90 distances × 10 items)
Per distance call:           17.65μs
```

### Computation Complexity

| Component | Complexity | Per Call |
|-----------|-----------|----------|
| cross_block_disagreements_fast | O(n log K) | 17.65μs |
| Loop over assessors × clusters | O(N × C) | 90 calls per iteration |
| **Total _update_z** | **O(N×C×n×log K)** | **1.75ms per call** |

---

## Why It Scales Badly

### Empirical Scaling Data

```
Problem Size        Total Ops    Time/Iteration   Complexity
────────────────────────────────────────────────────────────
5 assessors, 10 items, 3 clusters     150         0.30ms    ✓
10 assessors, 20 items, 3 clusters    600         1.00ms    ✓
20 assessors, 50 items, 5 clusters   5,000        7.55ms    ✓
30 assessors, 100 items, 5 clusters  15,000       22.57ms   ✗ (Approaching bottleneck)
50 assessors, 50 items, 5 clusters   12,500       19.24ms   ✗
```

**Linear scaling with N, n, and C:** Doubling any parameter doubles the time.

---

## The 3 Reasons It's Expensive

### 1. **Exhaustive Cluster Evaluation** (Unavoidable)

Must evaluate **every assessor against every cluster**:
- 30 assessors × 3 clusters = 90 evaluations per iteration
- With 1,000 iterations = 90,000 evaluations
- Each evaluation is a full O(n log K) distance calculation

```
Why can't we skip this?
- We need accurate cluster probabilities for Gibbs sampling
- Skipping would bias the posterior inference
- No principled way to approximate without losing correctness
```

### 2. **Fenwick Tree Overhead** (Inherent)

Each distance calculation uses a Fenwick tree:
```python
for item in ranking:
    b = block_idx[item]        # O(1) lookup
    leq = fw.sum_prefix(b)     # O(log K) 
    inv += seen - leq          # O(1) arithmetic
    fw.add(b, 1)               # O(log K)
```

- Per ranking: O(n) items × O(log K) tree ops = **O(n log K)**
- This is optimal for this problem, but still expensive at scale
- Measurement: **17.65μs per call** = 1.11μs per Fenwick operation (already fast)

### 3. **Sequential Sampling Method** (Algorithmic Choice)

The Gibbs sampler requires:
1. Compute logweights for all clusters
2. Sample one cluster from the distribution
3. Update assignment

This sequential dependency prevents:
- ❌ Caching (theta changes each iteration)
- ❌ Parallelization (sequential sampling)
- ❌ Batching (single assessor per iteration)

---

## Cannot Cache (Why the Cache Optimization Helps So Little)

For posterior caching to help, you need the same (blocks, theta, gamma, delta) tuple repeated.

In _update_z:
- ✅ blocks stay the same during an iteration ✓
- ❌ **theta changes every iteration** (or every theta_jump iterations)
- ❌ Different for each cluster c

**Result:** Cache hit rate is ~0% because theta is the dominant factor and it changes constantly.

This is why `theta_jump=10` helps blocks_update (54.5% speedup) but NOT z_update (no speedup).

---

## Potential Optimizations (with Trade-offs)

### 1. **Vectorized Distance Calculation** (5-10% speedup)
```python
# Current: Pure Python loops
for item in ranking:
    b = block_idx[item]
    # ...

# Potential: NumPy vectorization
block_assignments = block_idx[ranking]
# ...vectorized Fenwick operations
```
**Limitation:** Fenwick tree is hard to vectorize; probably 5-10% at best

### 2. **Parallelization Over Assessors** (1.2-1.5× speedup)
```python
# Current: sequential
for i, r_i in enumerate(rankings):
    evaluate_in_all_clusters(r_i)

# Potential: parallel via joblib
results = Parallel(n_jobs=-1)(
    delayed(evaluate_in_all_clusters)(r_i) for r_i in rankings
)
```
**Limitation:** Python GIL limits true parallelism; probably 1.2-1.5× with multiprocessing

### 3. **Approximate Cluster Evaluation** (2-5× speedup, loses accuracy)
```python
# Only evaluate top K clusters per assessor
top_clusters = find_likely_clusters(r_i, max_clusters=K)
for c in top_clusters:
    evaluate(c)
```
**Limitation:** Biases posterior; breaks Gibbs sampling guarantees

### 4. **Alternative Inference Methods** (research-level, 5-10× potential)
- Variational inference instead of MCMC
- Approximate Bayesian computation
- Hierarchical clustering first, then refinement

**Limitation:** Complete algorithmic change; tons of research & testing needed

---

## Comparison: Why Blocks Update is Faster

Block moves (54.5% of time) are expensive but **much cheaper per evaluation** than cluster reassignment:

```
Block Moves:
- O(K²) distance calculations (K = blocks per cluster)
- But K << n (usually 2-5 items per block)
- Result: Fast

Cluster Reassignment:
- O(N×C×n) distance calculations
- Full ranking distances (n items)
- Result: Slow
```

Example with 10 items, 3 clusters:
- **Block moves:** ~5 block reorderings × O(25) ops = O(125) per cluster
- **Cluster reassign:** 30 assessors × 3 clusters × O(100+) Fenwick ops = O(9000+) per iteration

---

## Practical Recommendations

### For Small Datasets (n < 20, N < 50)
Keep default settings—baseline is fast enough:
```python
model.run_mcmc(
    n_iter=2000,
    theta_jump=1,
    use_annealing=False,
)
# Expected: 200-300 iters/sec
```

### For Medium Datasets (n = 20-100, N = 50-200)
Use theta_jump trick to speed up block moves:
```python
model.run_mcmc(
    n_iter=5000,
    theta_jump=5,  # z_update unaffected, but blocks get 1.16× speedup
    use_annealing=False,
)
# Expected: 250 iters/sec (limited by z_update)
```

### For Large Datasets (n > 100, N > 200)
Consider parallelization:
```python
# Enable Fenwick tree parallelization if implemented
# Or use algorithmic alternatives (e.g., Variational Inference)
```

---

## Key Takeaway

**_update_z is slow because cluster assignment in Bayesian clustering is inherently expensive:**

- Must evaluate every assessor against every cluster
- Each evaluation is O(n log K) distance computation
- Total = unavoidable O(N×C×n×log K) per iteration
- No practical optimization path to 10× speedup without changing the algorithm

It's not a bug—it's the price of accuracy in Gibbs sampling for mixture models. The good news is that block moves (the other major component) CAN be optimized with theta_jump, giving you 1.16-1.68× overall speedup.

---

## How to Use the Diagnostics

Run this to see your specific bottleneck costs:

```bash
python z_update_bottleneck_analysis.py
```

The output shows:
1. Actual _update_z timing on your problem size
2. Distance calculation costs
3. Scaling behavior
4. Fraction of total MCMC time

Use this data to decide if optimization is worth the effort for your application.
