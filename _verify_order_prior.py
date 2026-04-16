"""
Verify hypothesis: without PY prior and without order prior,
p >= 0.5 should yield all-singleton blocks (K = n).

With p < 0.5 we should see ties (K < n).
"""
import numpy as np
from model.core import MixtureRankingModel

n_items, n_assessors = 6, 50
R = [list(np.random.RandomState(42 + i).permutation(n_items)) for i in range(n_assessors)]

print("=" * 70)
print("  No PY prior, no order prior: does p >= 0.5 => all singletons?")
print("=" * 70)

for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    m = MixtureRankingModel(R, n_clusters=1, seed=42)
    m.run_mcmc(
        n_iter=3000, burn_in=1000,
        tie_penalty=p,
        use_py_prior=False,
        include_order_prior=False,
    )
    # Extract K from saved blocks (post burn-in)
    Ks = [len(b[0]) for b in m.samples.blocks_samples]
    mean_K = np.mean(Ks)
    min_K = np.min(Ks)
    max_K = np.max(Ks)
    all_singleton = all(k == n_items for k in Ks)
    print(f"  p={p:.1f}: mean K = {mean_K:.2f}, range [{min_K}, {max_K}], "
          f"all_singleton={all_singleton}  {'<-- EXPECTED' if (p >= 0.5 and all_singleton) or (p < 0.5 and not all_singleton) else '<-- UNEXPECTED' if p >= 0.5 else ''}")

print()
print("=" * 70)
print("  Same but WITH order prior: p >= 0.5 should still have ties")
print("=" * 70)

for p in [0.5, 0.9]:
    m = MixtureRankingModel(R, n_clusters=1, seed=42)
    m.run_mcmc(
        n_iter=3000, burn_in=1000,
        tie_penalty=p,
        use_py_prior=False,
        include_order_prior=True,
    )
    Ks = [len(b[0]) for b in m.samples.blocks_samples]
    mean_K = np.mean(Ks)
    print(f"  p={p:.1f}: mean K = {mean_K:.2f} (with order prior, expect K < {n_items})")
