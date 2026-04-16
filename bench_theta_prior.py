"""
Sensitivity analysis: Theta prior parameters (a_theta, b_theta).

Sweep the Gamma(a_theta, b_theta) prior on the per-cluster concentration
parameter theta.  A stronger prior (large a, small b) concentrates theta
near a/b, while a weaker prior (small a) gives the data more influence.

The prior mean is a_theta / b_theta and variance is a_theta / b_theta^2.
Varying these controls how tightly theta is constrained, which in turn
affects cluster sharpness and, indirectly, the number of active clusters.

Results are saved to  f1_data/runs/theta_prior_<timestamp>/.
"""

import json
import pickle
import random as _random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from model import (
    ClusterParams,
    MixtureRankingModel,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("f1_data")
RANKINGS_DIR = DATA_DIR / "rankings"
RUNS_DIR     = DATA_DIR / "runs"


# ── Load data (same pipeline as Formula_1.ipynb) ─────────────────────────────

def to_partial_rank_lists(rankings: pd.DataFrame):
    first_app = (
        rankings.sort_values(["season", "round", "rank"])
        .drop_duplicates("driver_id")
        .set_index("driver_id")[["season", "round"]]
    )
    drivers = first_app.sort_values(["season", "round"]).index.tolist()
    driver_to_idx = {d: i for i, d in enumerate(drivers)}

    rankings_list, race_labels = [], []
    for (season, rnd), grp in (
        rankings.sort_values(["season", "round", "rank"])
        .groupby(["season", "round"], sort=False)
    ):
        meta = grp.iloc[0]
        r = [driver_to_idx[d] for d in grp.sort_values("rank")["driver_id"]]
        rankings_list.append(r)
        race_labels.append(f"{season} R{str(rnd).zfill(2)} {meta['race_name']}")
    return rankings_list, race_labels, drivers


rankings_full = pd.read_csv(RANKINGS_DIR / "f1_rankings.csv")
partial_rankings, race_labels, all_drivers = to_partial_rank_lists(rankings_full)
n_items = len(all_drivers)
N = len(partial_rankings)

print(f"Dataset: {N} races, {n_items} drivers")


# ── Shared MCMC settings ─────────────────────────────────────────────────────

N_ITER   = 30_000
BURN_IN  = 20_000
THIN     = 2
C        = 40            # fewer initial clusters (closer to expected active count)
SEED     = 42

# Common kwargs (a_theta / b_theta are overridden per config)
COMMON_MCMC = dict(
    n_iter=N_ITER,
    burn_in=BURN_IN,
    thin=THIN,
    save_samples=True,
    save_tau=True,
    save_theta=True,
    save_logp=True,
    n_item_moves_per_cluster=1,
    theta_jump=10,
    ranking_jump=5,
    use_annealing=True,
    temp_min=0.1,
    temp_max=1.0,
    use_py_prior=True,
    gamma=1.0,
    delta=0.5,
)


# ── Helper: build initial clusters + z ──────────────────────────────────────

def make_init(gamma: float = 1.0, delta: float = 0.5):
    """Return (clusters, z, mu) with paired-block init."""
    blocks = [[i, i + 1] for i in range(0, n_items - 1, 2)]
    if n_items % 2 == 1:
        blocks.append([n_items - 1])

    clusters = [
        ClusterParams(
            blocks=[b[:] for b in blocks],
            theta=1.0,
            gamma=gamma,
            delta=delta,
        )
        for _ in range(C)
    ]
    rng = _random.Random(SEED)
    z = [i % C for i in range(N)]
    rng.shuffle(z)
    mu = [1 / C] * C
    return clusters, z, mu


# ── Helper: run one configuration and collect results ─────────────────────────

def run_config(label: str, a_theta: float, b_theta: float) -> dict:
    """Run MCMC for one (a_theta, b_theta) configuration."""
    prior_mean = a_theta / b_theta
    prior_var  = a_theta / b_theta**2
    print(f"\n{'='*60}")
    print(f"  Config: {label}")
    print(f"  a_theta={a_theta}, b_theta={b_theta}  "
          f"(prior mean={prior_mean:.2f}, var={prior_var:.2f})")
    print(f"{'='*60}")

    clusters, z, mu = make_init()

    model = MixtureRankingModel(
        rankings=partial_rankings,
        n_items=n_items,
        init_clusters=clusters,
        init_z=z,
        init_mu=mu,
        seed=SEED,
        verbose=True,
        init_theta=1.0,
    )

    t0 = time.time()
    final_state, samples = model.run_mcmc(
        **COMMON_MCMC,
        a_theta=a_theta,
        b_theta=b_theta,
    )
    elapsed = time.time() - t0

    model.print_acceptance_summary()

    # Compute summary statistics
    active = [c for c in range(model.C) if c not in model._dead_clusters
              and sum(1 for zi in final_state.z if zi == c) > 0]
    n_active = len(active)

    logp_post = samples.logp[len(samples.logp)//2:] if samples.logp else []
    mean_logp = float(np.mean(logp_post)) if logp_post else float("nan")

    block_counts = {c: len(final_state.clusters[c].blocks) for c in active}

    summary = dict(
        label=label,
        a_theta=a_theta,
        b_theta=b_theta,
        prior_mean=prior_mean,
        prior_var=prior_var,
        n_active_clusters=n_active,
        mean_logp_post_burnin=mean_logp,
        block_counts=block_counts,
        elapsed_s=round(elapsed, 1),
    )

    print(f"\n  → Active clusters: {n_active}, mean log-p: {mean_logp:.1f}, "
          f"time: {elapsed:.0f}s")

    return dict(summary=summary, final_state=final_state,
                samples=samples, model=model)


# ── Output directory ──────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = RUNS_DIR / f"theta_prior_{timestamp}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Theta prior sweep:  vary (a_theta, b_theta)
#
#   Grid designed to cover:
#     - Different prior means (a/b):  0.5, 1, 2, 4, 8
#     - Different prior strengths:   weak (small a) vs strong (large a)
#
#   (a, b) pairs:
#     Prior mean ≈ 1:   (1,1), (2,2), (4,4)        — weak → strong
#     Prior mean ≈ 2:   (1,0.5), (2,1), (4,2)       — weak → strong
#     Prior mean ≈ 4:   (2,0.5), (4,1), (8,2)       — weak → strong
#     Prior mean ≈ 0.5: (0.5,1), (1,2), (2,4)       — weak → strong
# ══════════════════════════════════════════════════════════════════════════════

THETA_PRIOR_GRID = [
    # (a_theta, b_theta)  — prior mean = a/b
    # --- Prior mean ≈ 0.5 (weak theta → diffuse clusters) ---
    (0.5, 1.0),
    (1.0, 2.0),
    (2.0, 4.0),
    # --- Prior mean ≈ 1 ---
    (1.0, 1.0),
    (2.0, 2.0),
    (4.0, 4.0),
    # --- Prior mean ≈ 2 ---
    (1.0, 0.5),
    (2.0, 1.0),      # SamplerConfig default
    (4.0, 2.0),
    # --- Prior mean ≈ 4 (strong theta → sharp clusters) ---
    (2.0, 0.5),
    (4.0, 1.0),      # used in bench_py_tiepen.py
    (8.0, 2.0),
]

print("\n" + "=" * 60)
print("  Theta prior sweep: Gamma(a_theta, b_theta)")
print(f"  {len(THETA_PRIOR_GRID)} configurations, C={C} initial clusters")
print("=" * 60)

results = []
for a, b in THETA_PRIOR_GRID:
    label = f"theta_a{a}_b{b}"
    result = run_config(label, a_theta=a, b_theta=b)
    results.append(result)

    run_subdir = OUT_DIR / label
    run_subdir.mkdir(exist_ok=True)
    with open(run_subdir / "summary.json", "w") as f:
        json.dump(result["summary"], f, indent=2, default=str)
    with open(run_subdir / "samples.pkl", "wb") as f:
        pickle.dump(result["samples"], f)
    with open(run_subdir / "final_state.pkl", "wb") as f:
        pickle.dump(result["final_state"], f)


# ══════════════════════════════════════════════════════════════════════════════
# Summary table
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 80)
print("  SUMMARY — Theta prior sweep: Gamma(a, b)")
print("=" * 80)
print(f"{'a':>6} {'b':>6} {'mean':>6} {'var':>8} {'#active':>8} {'mean logp':>12} {'time(s)':>8}")
for r in results:
    s = r["summary"]
    print(f"{s['a_theta']:>6.1f} {s['b_theta']:>6.1f} {s['prior_mean']:>6.1f} "
          f"{s['prior_var']:>8.2f} {s['n_active_clusters']:>8d} "
          f"{s['mean_logp_post_burnin']:>12.1f} {s['elapsed_s']:>8.1f}")

# Save combined summary
all_summaries = [r["summary"] for r in results]
with open(OUT_DIR / "all_summaries.json", "w") as f:
    json.dump(all_summaries, f, indent=2, default=str)

summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv(OUT_DIR / "summary_table.csv", index=False)

print(f"\n\nAll results saved to: {OUT_DIR}")
