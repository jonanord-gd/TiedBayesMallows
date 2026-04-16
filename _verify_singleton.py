"""Check whether singletons always have higher posterior when p >= 0.5."""
import numpy as np
import math
from itertools import combinations
from model.priors import log_Z_star_from_sizes
from model.distance import total_distance_fast

n = 6
N = 50
R = [list(np.random.RandomState(42 + i).permutation(n)) for i in range(N)]

def log_posterior(blocks, theta, p):
    """Log posterior without PY prior or order prior."""
    S = total_distance_fast(R, blocks, tie_penalty=p)
    sizes = [len(b) for b in blocks]
    logZ = log_Z_star_from_sizes(sizes, theta)
    return -theta * S - N * logZ

singleton = [[i] for i in range(n)]

print("=" * 70)
print("  Singleton vs all single-pair merges (no PY, no order prior)")
print("=" * 70)

for theta in [0.1, 0.3, 1.0, 3.0]:
    print(f"\n  theta = {theta}")
    for p in [0.3, 0.5, 0.7, 1.0]:
        lp_sing = log_posterior(singleton, theta, p)
        worst_diff = float('inf')
        best_merge = None
        for a, b in combinations(range(n), 2):
            blocks = [[i] for i in range(n) if i != a and i != b] + [[a, b]]
            lp_merge = log_posterior(blocks, theta, p)
            diff = lp_sing - lp_merge
            if diff < worst_diff:
                worst_diff = diff
                best_merge = (a, b)
        status = "singletons ALWAYS better" if worst_diff > 0 else "MERGE can be better"
        print(f"    p={p:.1f}: gap={worst_diff:+.4f}  closest_merge={best_merge}  -> {status}")

# Also decompose the gap into distance vs Z* contributions
print("\n" + "=" * 70)
print("  Decomposition: distance vs Z* for merge (0,1) at theta=0.3")
print("=" * 70)
theta = 0.3
merge01 = [[i] for i in range(2, n)] + [[0, 1]]
for p in [0.3, 0.5, 0.7, 1.0]:
    S_sing = total_distance_fast(R, singleton, tie_penalty=p)
    S_merge = total_distance_fast(R, merge01, tie_penalty=p)
    logZ_sing = log_Z_star_from_sizes([1]*n, theta)
    logZ_merge = log_Z_star_from_sizes([1]*(n-2) + [2], theta)
    dist_diff = -theta * (S_sing - S_merge)      # negative = merge has lower distance
    Z_diff = -N * (logZ_sing - logZ_merge)        # negative = singletons have lower Z
    total = dist_diff + Z_diff
    print(f"  p={p:.1f}: dist_contribution={dist_diff:+.4f}  Z_contribution={Z_diff:+.4f}  total={total:+.4f}")
