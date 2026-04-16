"""
Sensitivity analysis: Pitman-Yor prior parameters & tie-penalty weight.

Two experiments on the F1 partial-ranking dataset:

  Experiment 1 — Pitman-Yor sweep (fixed tie_penalty=0.5)
    Vary (gamma, delta) across a grid while keeping p = 0.5.

  Experiment 2 — Tie-penalty sweep (no PY prior)
    Disable the PY prior (use_py_prior=False) and sweep p in {0.1, 0.2, ..., 0.9}.

Each configuration is run for a moderate number of MCMC iterations.
Results (log-posterior, number of active clusters, MAP block counts,
theta traces, acceptance rates) are saved to a timestamped folder
under  f1_data/runs/sensitivity_<timestamp>/.
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

N_ITER   = 30_000        # total iterations per run (adjust for time budget)
BURN_IN  = 20_000        # burn-in
THIN     = 2
C        = 100           # initial clusters (model will collapse unused ones)
SEED     = 42

# Common kwargs passed to every run_mcmc call
COMMON_MCMC = dict(
    n_iter=N_ITER,
    burn_in=BURN_IN,
    thin=THIN,
    save_samples=True,
    save_tau=True,
    save_theta=True,
    save_logp=True,
    n_item_moves_per_cluster=1,
    a_theta=4,
    b_theta=1,
    theta_jump=10,
    ranking_jump=5,
    use_annealing=True,
    temp_min=0.1,
    temp_max=1.0,
)


# ── Helper: build initial clusters + z ──────────────────────────────────────

def make_init(gamma: float, delta: float):
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

def run_config(label: str, gamma: float, delta: float,
               use_py_prior: bool) -> dict:
    """Run MCMC for one parameter configuration and return summary dict."""
    print(f"\n{'='*60}")
    print(f"  Config: {label}")
    print(f"  gamma={gamma}, delta={delta}, "
          f"use_py_prior={use_py_prior}")
    print(f"{'='*60}")

    clusters, z, mu = make_init(gamma, delta)

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
        gamma=gamma,
        delta=delta,
        use_py_prior=use_py_prior,
    )
    elapsed = time.time() - t0

    model.print_acceptance_summary()

    # Compute summary statistics
    active = [c for c in range(model.C) if c not in model._dead_clusters
              and sum(1 for zi in final_state.z if zi == c) > 0]
    n_active = len(active)

    logp_post = samples.logp[len(samples.logp)//2:] if samples.logp else []
    mean_logp = float(np.mean(logp_post)) if logp_post else float("nan")

    # Block counts per active cluster at final state
    block_counts = {c: len(final_state.clusters[c].blocks) for c in active}

    # Theta traces (post burn-in)
    theta_post = samples.theta_samples if samples.theta_samples else []

    summary = dict(
        label=label,
        gamma=gamma,
        delta=delta,
        use_py_prior=use_py_prior,
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
OUT_DIR = RUNS_DIR / f"sensitivity_{timestamp}"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 1:  Pitman-Yor parameter sweep  (tie_penalty = 0.5)
# ══════════════════════════════════════════════════════════════════════════════

PY_GRID = [
    # (gamma, delta) — gamma = concentration, delta = discount
    (0.5, 0.0),     # delta=0 → Dirichlet process limit
    (1.0, 0.0),
    (2.0, 0.0),
    (0.5, 0.25),
    (1.0, 0.25),
    (2.0, 0.25),
    (0.5, 0.5),
    (1.0, 0.5),     # baseline used in notebook
    (2.0, 0.5),
    (0.5, 0.75),
    (1.0, 0.75),
    (2.0, 0.75),
]

print("\n" + "=" * 60)
print("  EXPERIMENT 1: Pitman-Yor parameter sweep (p = 0.5)")
print("=" * 60)

exp1_results = []
for gamma, delta in PY_GRID:
    label = f"PY_g{gamma}_d{delta}_p0.5"
    result = run_config(label, gamma=gamma, delta=delta,
                        use_py_prior=True)
    exp1_results.append(result)

    # Save individual run
    run_subdir = OUT_DIR / label
    run_subdir.mkdir(exist_ok=True)
    with open(run_subdir / "summary.json", "w") as f:
        json.dump(result["summary"], f, indent=2, default=str)
    with open(run_subdir / "samples.pkl", "wb") as f:
        pickle.dump(result["samples"], f)
    with open(run_subdir / "final_state.pkl", "wb") as f:
        pickle.dump(result["final_state"], f)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment 2:  Tie-penalty sweep  (no PY prior)
# ══════════════════════════════════════════════════════════════════════════════

P_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

print("\n" + "=" * 60)
print("  EXPERIMENT 2: Tie-penalty sweep (no PY prior)")
print("=" * 60)

exp2_results = []
for p in P_VALUES:
    label = f"noPY_p{p:.1f}"
    # gamma/delta are still set on clusters but use_py_prior=False disables the EPPF term
    result = run_config(label, gamma=1.0, delta=0.5,
                        use_py_prior=False)
    exp2_results.append(result)

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
print("  SUMMARY — Experiment 1: Pitman-Yor sweep (tie_penalty = 0.5)")
print("=" * 80)
print(f"{'gamma':>6} {'delta':>6} {'#active':>8} {'mean logp':>12} {'time(s)':>8}")
for r in exp1_results:
    s = r["summary"]
    print(f"{s['gamma']:>6.2f} {s['delta']:>6.2f} {s['n_active_clusters']:>8d} "
          f"{s['mean_logp_post_burnin']:>12.1f} {s['elapsed_s']:>8.1f}")

print("\n" + "=" * 80)
print("  SUMMARY — Experiment 2: Tie-penalty sweep (no PY prior)")
print("=" * 80)
print(f"{'label':>15} {'#active':>8} {'mean logp':>12} {'time(s)':>8}")
for r in exp2_results:
    s = r["summary"]
    print(f"{s['label']:>15} {s['n_active_clusters']:>8d} "
          f"{s['mean_logp_post_burnin']:>12.1f} {s['elapsed_s']:>8.1f}")

# Save combined summary
all_summaries = [r["summary"] for r in exp1_results + exp2_results]
with open(OUT_DIR / "all_summaries.json", "w") as f:
    json.dump(all_summaries, f, indent=2, default=str)

summary_df = pd.DataFrame(all_summaries)
summary_df.to_csv(OUT_DIR / "summary_table.csv", index=False)

print(f"\n\nAll results saved to: {OUT_DIR}")
