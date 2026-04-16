"""Benchmark: augmentation cost with varying missingness and n.

Compares:
  1. Incremental delta  — O(|p2 - p1|) per proposal
  2. Full Fenwick       — O(n log K) per proposal
  3. Full MCMC step     — measures total overhead of augmentation
"""

import random
import time
import warnings
from typing import List

import numpy as np

warnings.filterwarnings("ignore")

from model import MixtureRankingModel, ClusterParams
from model.augmentation import (
    PartialRankingInfo,
    augmentation_mh_step,
    complete_rankings,
    detect_missing,
)


def make_partial_rankings(
    n: int, N: int, frac_partial: float, frac_missing_per_assessor: float, seed: int = 0
) -> List[List[int]]:
    """Generate N rankings of n items with controlled missingness.

    Parameters
    ----------
    frac_partial : float
        Fraction of assessors that have missing items (0.0 to 1.0).
    frac_missing_per_assessor : float
        For each partial assessor, fraction of items that are missing.
    """
    rng = random.Random(seed)
    rankings = []
    for i in range(N):
        r = list(range(n))
        rng.shuffle(r)
        if rng.random() < frac_partial:
            n_miss = max(1, int(frac_missing_per_assessor * n))
            r = r[: n - n_miss]  # keep only observed prefix
        rankings.append(r)
    return rankings


def make_blocks(n: int, K: int, seed: int = 0) -> List[List[int]]:
    rng = random.Random(seed)
    items = list(range(n))
    rng.shuffle(items)
    blocks = []
    per = n // K
    for k in range(K):
        s = k * per
        e = s + per if k < K - 1 else n
        blocks.append(items[s:e])
    return blocks


def bench_augmentation_methods(n, N, K, frac_partial, frac_missing, n_sweeps=20, seed=42):
    """Time incremental vs fenwick augmentation over n_sweeps MH sweeps."""
    rankings_raw = make_partial_rankings(n, N, frac_partial, frac_missing, seed)
    blocks = make_blocks(n, K, seed)
    info = detect_missing(rankings_raw, n)
    rng = random.Random(seed)

    # Create cluster/cache-like objects
    from model.blocks import blocks_to_block_index
    block_idx = blocks_to_block_index(blocks, n)
    sizes = [len(b) for b in blocks]

    class FakeCluster:
        def __init__(self):
            self.theta = 1.0
            self.blocks = blocks

    class FakeCache:
        def __init__(self):
            self.block_idx = block_idx
            self.K = K
            self.sizes = sizes

    clusters = [FakeCluster()]
    cache = [FakeCache()]
    z = np.zeros(N, dtype=np.intp)

    results = {}
    for method in ["incremental", "fenwick"]:
        # Fresh copy each time
        rankings_copy = complete_rankings(
            rankings_raw, info, n, random.Random(seed), partial_mode="top_k"
        )
        rng_m = random.Random(seed + 1)
        t0 = time.perf_counter()
        total_prop = 0
        total_acc = 0
        for _ in range(n_sweeps):
            p, a = augmentation_mh_step(
                rankings_copy,
                info,
                z,
                clusters,
                cache,
                rng_m,
                n,
                method=method,
                partial_mode="top_k",
            )
            total_prop += p
            total_acc += a
        elapsed = time.perf_counter() - t0
        results[method] = {
            "time": elapsed,
            "per_sweep": elapsed / n_sweeps * 1000,  # ms
            "proposals": total_prop,
            "accepts": total_acc,
        }

    return results


def bench_full_mcmc(n, N, K, frac_partial, frac_missing, n_iter=200, seed=42):
    """Time full MCMC with and without partial rankings."""
    blocks = make_blocks(n, K, seed)
    clusters_init = [ClusterParams(blocks=blocks, theta=1.0, gamma=1.0, delta=0.5)]

    # Complete data baseline
    complete = make_partial_rankings(n, N, 0.0, 0.0, seed)
    model_c = MixtureRankingModel(complete, init_clusters=[
        ClusterParams(blocks=[b[:] for b in blocks], theta=1.0, gamma=1.0, delta=0.5)
    ], seed=seed)
    t0 = time.perf_counter()
    model_c.run_mcmc(n_iter=n_iter, save_logp=False)
    time_complete = time.perf_counter() - t0

    # Partial data
    partial = make_partial_rankings(n, N, frac_partial, frac_missing, seed)
    model_p = MixtureRankingModel(partial, n_items=n, init_clusters=[
        ClusterParams(blocks=[b[:] for b in blocks], theta=1.0, gamma=1.0, delta=0.5)
    ], seed=seed, partial_mode="top_k")

    # ranking_jump=1
    t0 = time.perf_counter()
    model_p.run_mcmc(n_iter=n_iter, ranking_jump=1, save_logp=False)
    time_partial_1 = time.perf_counter() - t0

    # ranking_jump=5
    model_p2 = MixtureRankingModel(partial, n_items=n, init_clusters=[
        ClusterParams(blocks=[b[:] for b in blocks], theta=1.0, gamma=1.0, delta=0.5)
    ], seed=seed, partial_mode="top_k")
    t0 = time.perf_counter()
    model_p2.run_mcmc(n_iter=n_iter, ranking_jump=5, save_logp=False)
    time_partial_5 = time.perf_counter() - t0

    return {
        "complete": time_complete,
        "partial_rj1": time_partial_1,
        "partial_rj5": time_partial_5,
        "overhead_rj1": time_partial_1 / time_complete,
        "overhead_rj5": time_partial_5 / time_complete,
    }


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("BENCHMARK 1: Incremental vs Fenwick augmentation (per MH sweep)")
    print("=" * 72)

    configs = [
        # (n, N, K, frac_partial, frac_missing)
        (20,  100, 5,  0.5, 0.3),
        (50,  100, 10, 0.5, 0.3),
        (100, 100, 15, 0.5, 0.3),
        (200, 100, 20, 0.5, 0.3),
        (50,  100, 10, 0.5, 0.1),   # low missingness
        (50,  100, 10, 0.5, 0.5),   # high missingness
        (50,  100, 10, 0.8, 0.3),   # many partial assessors
        (50,  200, 10, 0.5, 0.3),   # more assessors
    ]

    print(f"\n{'n':>5} {'N':>5} {'K':>4} {'%part':>6} {'%miss':>6} | "
          f"{'Incr (ms)':>10} {'Fenw (ms)':>10} {'Speedup':>8}")
    print("-" * 72)

    for n, N, K, fp, fm in configs:
        res = bench_augmentation_methods(n, N, K, fp, fm, n_sweeps=50)
        incr = res["incremental"]["per_sweep"]
        fenw = res["fenwick"]["per_sweep"]
        speedup = fenw / incr if incr > 0 else float("inf")
        print(f"{n:>5} {N:>5} {K:>4} {fp:>6.0%} {fm:>6.0%} | "
              f"{incr:>10.3f} {fenw:>10.3f} {speedup:>7.1f}x")

    print("\n" + "=" * 72)
    print("BENCHMARK 2: Full MCMC overhead from augmentation")
    print("=" * 72)

    mcmc_configs = [
        # (n, N, K, frac_partial, frac_missing)
        (20,  50,  5,  0.5, 0.3),
        (50,  100, 10, 0.5, 0.3),
        (50,  100, 10, 0.8, 0.5),   # heavy missingness
    ]

    print(f"\n{'n':>5} {'N':>5} {'K':>4} {'%part':>6} {'%miss':>6} | "
          f"{'Complete':>10} {'Part rj=1':>10} {'Part rj=5':>10} "
          f"{'OH rj=1':>8} {'OH rj=5':>8}")
    print("-" * 80)

    for n, N, K, fp, fm in mcmc_configs:
        res = bench_full_mcmc(n, N, K, fp, fm, n_iter=200)
        print(f"{n:>5} {N:>5} {K:>4} {fp:>6.0%} {fm:>6.0%} | "
              f"{res['complete']:>9.3f}s {res['partial_rj1']:>9.3f}s "
              f"{res['partial_rj5']:>9.3f}s "
              f"{res['overhead_rj1']:>7.2f}x {res['overhead_rj5']:>7.2f}x")


    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("BENCHMARK 3: Same data, complete vs partial (full MCMC)")
    print("=" * 72)
    print()
    print("Generate complete rankings, then strip items to create partial")
    print("versions.  Run identical MCMC on both.  Measures the true cost")
    print("of handling missingness on the *same underlying data*.")
    print()

    scenarios = [
        # label, n, N, C, K, n_iter, frac_partial, frac_missing, ranking_jump
        ("Small (n=20, N=50)",     20,   50, 1,  5, 500, 0.3, 0.2, 1),
        ("Small (n=20, N=50)",     20,   50, 1,  5, 500, 0.3, 0.2, 5),
        ("Small (n=20, N=50)",     20,   50, 1,  5, 500, 0.5, 0.3, 1),
        ("Small (n=20, N=50)",     20,   50, 1,  5, 500, 0.5, 0.3, 5),
        ("Medium (n=50, N=100)",   50,  100, 2, 10, 500, 0.3, 0.2, 1),
        ("Medium (n=50, N=100)",   50,  100, 2, 10, 500, 0.3, 0.2, 5),
        ("Medium (n=50, N=100)",   50,  100, 2, 10, 500, 0.5, 0.3, 1),
        ("Medium (n=50, N=100)",   50,  100, 2, 10, 500, 0.5, 0.3, 5),
        ("Medium (n=50, N=100)",   50,  100, 2, 10, 500, 0.8, 0.5, 1),
        ("Medium (n=50, N=100)",   50,  100, 2, 10, 500, 0.8, 0.5, 5),
        ("Large (n=100, N=200)",  100,  200, 2, 15, 300, 0.5, 0.3, 1),
        ("Large (n=100, N=200)",  100,  200, 2, 15, 300, 0.5, 0.3, 5),
        ("Large (n=100, N=200)",  100,  200, 2, 15, 300, 0.8, 0.5, 1),
        ("Large (n=100, N=200)",  100,  200, 2, 15, 300, 0.8, 0.5, 5),
    ]

    header = (f"{'Scenario':<26} {'%part':>6} {'%miss':>6} {'rj':>3} "
              f"{'n_iter':>6} | {'Complete':>10} {'Partial':>10} "
              f"{'Overhead':>9} {'ms/iter(C)':>11} {'ms/iter(P)':>11}")
    print(header)
    print("-" * len(header))

    # Cache complete-data timings to avoid re-running for same (n,N,C,K,n_iter)
    complete_cache = {}

    for label, n, N, C, K, n_iter, fp, fm, rj in scenarios:
        seed = 42
        rng_data = random.Random(seed)
        blocks_list = []
        for c_idx in range(C):
            blocks_list.append(make_blocks(n, K, seed=seed + c_idx))

        # Generate complete rankings
        complete_rankings_data = []
        for i in range(N):
            r = list(range(n))
            rng_data.shuffle(r)
            complete_rankings_data.append(r)

        # Strip items to create partial version
        rng_strip = random.Random(seed + 999)
        partial_rankings_data = []
        for r in complete_rankings_data:
            if rng_strip.random() < fp:
                n_miss = max(1, int(fm * n))
                partial_rankings_data.append(r[:n - n_miss])
            else:
                partial_rankings_data.append(list(r))

        n_partial = sum(1 for r in partial_rankings_data if len(r) < n)

        # Run complete
        cache_key = (n, N, C, K, n_iter)
        if cache_key not in complete_cache:
            clusters_c = [
                ClusterParams(blocks=[b[:] for b in blocks_list[c_idx]],
                              theta=1.0, gamma=1.0, delta=0.5)
                for c_idx in range(C)
            ]
            model_c = MixtureRankingModel(
                complete_rankings_data, init_clusters=clusters_c, seed=seed
            )
            t0 = time.perf_counter()
            model_c.run_mcmc(n_iter=n_iter, save_logp=False)
            t_complete = time.perf_counter() - t0
            complete_cache[cache_key] = t_complete
        else:
            t_complete = complete_cache[cache_key]

        # Run partial
        clusters_p = [
            ClusterParams(blocks=[b[:] for b in blocks_list[c_idx]],
                          theta=1.0, gamma=1.0, delta=0.5)
            for c_idx in range(C)
        ]
        model_p = MixtureRankingModel(
            partial_rankings_data,
            n_items=n,
            init_clusters=clusters_p,
            seed=seed,
            partial_mode="top_k",
        )
        t0 = time.perf_counter()
        model_p.run_mcmc(n_iter=n_iter, ranking_jump=rj, save_logp=False)
        t_partial = time.perf_counter() - t0

        overhead = t_partial / t_complete
        ms_c = t_complete / n_iter * 1000
        ms_p = t_partial / n_iter * 1000

        print(f"{label:<26} {fp:>6.0%} {fm:>6.0%} {rj:>3} "
              f"{n_iter:>6} | {t_complete:>9.3f}s {t_partial:>9.3f}s "
              f"{overhead:>8.2f}x {ms_c:>10.2f}ms {ms_p:>10.2f}ms")
