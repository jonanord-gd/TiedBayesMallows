"""Benchmark: incremental D updates vs full matmul.

Runs a short MCMC at various n values and reports per-iteration timing.
The incremental D path is the default after the code change.  To compare
against the old full-matmul path, set FORCE_FULL_MATMUL = True below.

Usage:
    python bench_incremental_D.py
"""

import time
import numpy as np
import random as _random
from math import comb

from model import MixtureRankingModel, ClusterParams
from model.moves import compute_U_all, build_cluster_pair_masks
from model.blocks import blocks_to_block_index
from helper_functions.Generate_mixture_data import sample_mallows_tied


# ── Configuration ─────────────────────────────────────────────────────────────
N_VALUES = [50, 200, 500]       # number of assessors
N_ITEMS_VALUES = [25, 50, 100, 200, 500]  # number of items to benchmark
C = 5                            # clusters
N_ITER = 20                      # MCMC iterations per run
SEED = 42
FORCE_FULL_MATMUL = False        # set True to benchmark old path


def make_synthetic(n_items: int, n_assessors: int, n_clusters: int, seed: int):
    """Generate synthetic rankings from a Mallows mixture."""
    rng = _random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Create n_clusters consensuses (random block structures)
    clusters = []
    for _ in range(n_clusters):
        perm = np_rng.permutation(n_items).tolist()
        # Random blocks: 2-4 items per block
        blocks = []
        i = 0
        while i < n_items:
            bsize = min(rng.randint(2, 4), n_items - i)
            blocks.append(perm[i:i + bsize])
            i += bsize
        clusters.append(blocks)

    # Sample rankings from each cluster
    rankings = []
    z_true = []
    per_cluster = n_assessors // n_clusters
    np_rng = np.random.default_rng(seed)
    for c_idx, blocks in enumerate(clusters):
        count = per_cluster if c_idx < n_clusters - 1 else n_assessors - len(rankings)
        samples = sample_mallows_tied(blocks, alpha=1.5, rng=np_rng, n_samples=count)
        if count == 1:
            samples = [samples]
        rankings.extend(samples)
        z_true.extend([c_idx] * count)

    # Build init clusters
    perm0 = list(range(n_items))
    init_blocks = [[perm0[i], perm0[i + 1]] for i in range(0, n_items - 1, 2)]
    if n_items % 2 == 1:
        init_blocks.append([perm0[-1]])

    init_clusters = [
        ClusterParams(blocks=[b[:] for b in init_blocks], theta=1.0, gamma=1.0, delta=0.5)
        for _ in range(n_clusters)
    ]

    init_z = [i % n_clusters for i in range(n_assessors)]
    rng.shuffle(init_z)

    return rankings, init_clusters, init_z


def bench_mcmc(n_items: int, n_assessors: int, n_clusters: int, n_iter: int):
    """Run short MCMC and return per-iteration time."""
    rankings, init_clusters, init_z = make_synthetic(n_items, n_assessors, n_clusters, SEED)

    model = MixtureRankingModel(
        rankings=rankings,
        n_items=n_items,
        init_clusters=init_clusters,
        init_z=init_z,
        init_mu=[1.0 / n_clusters] * n_clusters,
        seed=SEED,
        verbose=False,
    )

    if FORCE_FULL_MATMUL:
        # Monkey-patch to disable incremental path
        model._D_cache = None
        _orig_apply = model._apply_incremental_D_update
        def _noop(*a, **kw):
            pass
        model._apply_incremental_D_update = _noop
        # Also ensure _rebuild_cluster_cache marks dirty so full matmul runs
        _orig_rebuild = model._rebuild_cluster_cache
        def _rebuild_and_dirty(c):
            _orig_rebuild(c)
            model._M_dirty[c] = True
            model._D_cache = None
        model._rebuild_cluster_cache = _rebuild_and_dirty

    # Warm-up iteration (first one always does full matmul)
    model.step(iteration=0, theta_jump=1, n_item_moves_per_cluster=1)

    # Timed iterations
    t0 = time.perf_counter()
    for it in range(1, n_iter + 1):
        model.step(iteration=it, theta_jump=10, n_item_moves_per_cluster=1)
    elapsed = time.perf_counter() - t0

    return elapsed / n_iter


def bench_raw_operations(n_items: int, n_assessors: int, n_clusters: int):
    """Directly time the full matmul vs a single incremental update."""
    rankings, init_clusters, init_z = make_synthetic(n_items, n_assessors, n_clusters, SEED)
    n_pairs = comb(n_items, 2)

    # Build U_all and M
    U_all = compute_U_all([r for r in rankings], n_items).astype(np.float32)
    block_idx_list = [
        blocks_to_block_index(cl.blocks, n_items, validate=False)
        for cl in init_clusters
    ]
    M, offsets = build_cluster_pair_masks(block_idx_list, n_items)
    M_f32_T = np.ascontiguousarray(M.T, dtype=np.float32)

    # Time full matmul
    n_reps = max(1, min(50, int(5e8 / (n_assessors * n_clusters * n_pairs + 1))))
    t0 = time.perf_counter()
    for _ in range(n_reps):
        D = U_all @ M_f32_T + offsets[np.newaxis, :]
    t_full = (time.perf_counter() - t0) / n_reps

    # Time a single incremental update (simulate moving one item in one cluster)
    D_cache = D.copy()
    c = 0
    old_bi = np.asarray(block_idx_list[c], dtype=np.intp)
    # Simulate moving item 0 from block 0 to block 1
    new_bi = old_bi.copy()
    new_bi[0] = max(old_bi) if max(old_bi) != old_bi[0] else 0

    xs = np.arange(n_items)
    xs = xs[xs != 0]
    a_arr = np.minimum(0, xs)
    b_arr = np.maximum(0, xs)
    pidx_arr = (a_arr * (2 * n_items - a_arr - 1) // 2 + (b_arr - a_arr - 1)).astype(np.intp)

    n_reps_inc = max(1, min(500, int(5e8 / (n_assessors * n_items + 1))))
    t0 = time.perf_counter()
    for _ in range(n_reps_inc):
        old_signs = M[c, pidx_arr].copy()
        new_signs = np.sign(new_bi[b_arr] - new_bi[a_arr]).astype(np.float64)
        changed = old_signs != new_signs
        if changed.any():
            cp = pidx_arr[changed]
            delta = (new_signs[changed] - old_signs[changed]).astype(np.float32)
            D_cache[:, c] += (U_all[:, cp] @ delta) + 0.0
    t_inc = (time.perf_counter() - t0) / n_reps_inc

    return t_full, t_inc


if __name__ == "__main__":
    print("=" * 72)
    print("Incremental D update benchmark")
    print("=" * 72)

    # Raw operation comparison
    print(f"\n{'n':>6} {'N':>6} {'C':>3} | {'Full matmul':>14} {'Incremental':>14} {'Speedup':>10}")
    print("-" * 72)
    for n_items in N_ITEMS_VALUES:
        for N in N_VALUES:
            if N < C * 2:
                continue
            t_full, t_inc = bench_raw_operations(n_items, N, C)
            speedup = t_full / t_inc if t_inc > 0 else float('inf')
            print(f"{n_items:>6} {N:>6} {C:>3} | {t_full*1000:>11.3f} ms {t_inc*1000:>11.3f} ms {speedup:>9.1f}x")

    # End-to-end MCMC
    print(f"\n{'n':>6} {'N':>6} {'C':>3} | {'ms/iter':>10}   mode")
    print("-" * 72)
    for n_items in N_ITEMS_VALUES:
        for N in N_VALUES:
            if N < C * 2:
                continue
            t_per_iter = bench_mcmc(n_items, N, C, N_ITER)
            mode = "FULL MATMUL" if FORCE_FULL_MATMUL else "incremental"
            print(f"{n_items:>6} {N:>6} {C:>3} | {t_per_iter*1000:>8.1f} ms   {mode}")
