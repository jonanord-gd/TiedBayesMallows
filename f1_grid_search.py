"""
F1 Hyperparameter Grid Search
==============================
Sweeps over (n_iter, C, seed) and evaluates how well race cluster assignments
capture the true F1 seasons.

Evaluation metrics (all computed against z_map, the MAP cluster assignment):
  - ARI  : Adjusted Rand Index between z_map and season label
  - NMI  : Normalized Mutual Information
  - homo : Homogeneity  (each cluster ≈ one season)
  - comp : Completeness (each season ≈ one cluster)
  - n_active_clusters : number of clusters with at least one race

The main outputs are:
  - results/f1_grid_results.csv        — one row per run
  - results/f1_grid_z_map.npy          — z_map arrays, shape (n_runs, N_races)

Usage:
  python f1_grid_search.py              # full grid
  python f1_grid_search.py --dry-run    # print grid without running
"""

import argparse
import json
import random
import time
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
)

from model import MixtureRankingModel, ClusterParams
from model.initialization import init_spectral_with_z

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("f1_data")
RANKINGS_DIR  = DATA_DIR / "rankings"
RESULTS_DIR   = Path("results") / "f1_grid"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Fixed MCMC settings (held constant across all runs) ───────────────────────
FIXED = dict(
    burn_in                  = None,   # set to n_iter // 2 per run (see below)
    thin                     = 1,
    use_py_prior             = False,
    include_order_prior      = False,
    n_item_moves_per_cluster = 1,
    a_theta                  = 4,
    b_theta                  = 1,
    theta_jump               = 10,
    ranking_jump             = 5,
    use_annealing            = True,
    temp_min                 = 0.1,
    temp_max                 = 1.0,
    gamma                    = 1.0,
    delta                    = 0.5,
    init_theta               = 1.0,
)

# ── Grid ──────────────────────────────────────────────────────────────────────
GRID = dict(
    n_iter = [20_000, 50_000, 80_000],   # total iterations (burn_in = n_iter // 2)
    C      = [10, 15, 20, 25, 30],       # number of initial clusters
    seed   = [42, 123, 777],             # MCMC seed
)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers (reused from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def to_partial_rank_lists(rankings: pd.DataFrame):
    """Mirror of the notebook helper — converts long-format rankings to partial lists."""
    first_app = (
        rankings.sort_values(["season", "round", "rank"])
        .drop_duplicates("driver_id")
        .set_index("driver_id")[["season", "round"]]
    )
    drivers = first_app.sort_values(["season", "round"]).index.tolist()
    driver_to_idx = {d: i for i, d in enumerate(drivers)}

    rankings_list: list[list[int]] = []
    race_labels:   list[str]       = []

    for (season, rnd), grp in (
        rankings
        .sort_values(["season", "round", "rank"])
        .groupby(["season", "round"], sort=False)
    ):
        meta = grp.iloc[0]
        r = [driver_to_idx[d] for d in grp.sort_values("rank")["driver_id"]]
        rankings_list.append(r)
        race_labels.append(f"{season} R{str(rnd).zfill(2)} {meta['race_name']}")

    return rankings_list, race_labels, drivers


def build_spectral_full_rankings(partial_rankings, n_items_partial):
    """Extend each partial ranking to a full permutation for spectral init."""
    pos_sum = np.zeros(n_items_partial, dtype=float)
    obs_cnt = np.zeros(n_items_partial, dtype=int)
    for r in partial_rankings:
        for pos, item in enumerate(r):
            pos_sum[item] += pos
            obs_cnt[item] += 1
    fallback = n_items_partial + np.arange(n_items_partial) / max(n_items_partial, 1)
    global_tail_order = list(
        np.argsort(np.where(obs_cnt > 0, pos_sum / obs_cnt, fallback))
    )
    full = []
    for r in partial_rankings:
        seen = set(r)
        missing = [item for item in global_tail_order if item not in seen]
        full.append(list(r) + missing)
    return full


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def season_labels_from_race_labels(race_labels: list[str]) -> np.ndarray:
    """Extract integer season (year) from race label strings."""
    return np.array([int(lbl.split()[0]) for lbl in race_labels])


def evaluate_clustering(z_map: list[int], season_labels: np.ndarray) -> dict:
    """Return a dict of clustering quality metrics vs true season labels."""
    z = np.array(z_map, dtype=int)
    return {
        "ari":              adjusted_rand_score(season_labels, z),
        "nmi":              normalized_mutual_info_score(season_labels, z, average_method="arithmetic"),
        "homogeneity":      homogeneity_score(season_labels, z),
        "completeness":     completeness_score(season_labels, z),
        "n_active_clusters": int(np.unique(z).size),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(
    partial_rankings:          list[list[int]],
    partial_rankings_spectral: list[list[int]],
    n_items_partial:           int,
    race_labels:               list[str],
    season_labels:             np.ndarray,
    *,
    n_iter: int,
    C:      int,
    seed:   int,
) -> dict:
    """Run one MCMC experiment and return a result dict."""
    burn_in = n_iter // 2

    # ── Spectral init ─────────────────────────────────────────────────────────
    gamma = FIXED["gamma"]
    delta = FIXED["delta"]

    clusters_init, z_init = init_spectral_with_z(
        rankings    = partial_rankings_spectral,
        n_clusters  = C,
        py_sampling = False,
        gamma       = gamma,
        delta       = delta,
        seed        = seed,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MixtureRankingModel(
        rankings       = partial_rankings,
        n_items        = n_items_partial,
        init_clusters  = clusters_init,
        init_z         = z_init,
        init_mu        = [1 / C],
        seed           = seed,
        verbose        = False,
        init_theta     = FIXED["init_theta"],
        init_gamma     = gamma,
        init_delta     = delta,
        partial_mode   = "top_k",
    )

    t0 = time.perf_counter()
    final_state, samples = model.run_mcmc(
        n_iter                   = n_iter,
        burn_in                  = burn_in,
        thin                     = FIXED["thin"],
        use_py_prior             = FIXED["use_py_prior"],
        include_order_prior      = FIXED["include_order_prior"],
        save_samples             = True,
        save_tau                 = False,
        save_theta               = False,
        n_item_moves_per_cluster = FIXED["n_item_moves_per_cluster"],
        gamma                    = gamma,
        delta                    = delta,
        a_theta                  = FIXED["a_theta"],
        b_theta                  = FIXED["b_theta"],
        theta_jump               = FIXED["theta_jump"],
        ranking_jump             = FIXED["ranking_jump"],
        use_annealing            = FIXED["use_annealing"],
        temp_min                 = FIXED["temp_min"],
        temp_max                 = FIXED["temp_max"],
    )
    elapsed = time.perf_counter() - t0

    # ── MAP assignment ────────────────────────────────────────────────────────
    map_result = model.find_map(samples, refine=True, verbose=False)
    z_map      = map_result["z"]

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = evaluate_clustering(z_map, season_labels)

    return {
        "n_iter":          n_iter,
        "burn_in":         burn_in,
        "C":               C,
        "seed":            seed,
        "elapsed_s":       round(elapsed, 1),
        "logp_map":        float(map_result.get("logp_refined", map_result["logp_chain"])),
        **metrics,
        "z_map":           z_map,   # excluded from CSV, saved separately
    }


# ─────────────────────────────────────────────────────────────────────────────
# Grid driver
# ─────────────────────────────────────────────────────────────────────────────

def main(dry_run: bool = False):
    print("Loading data …")
    rankings_full   = pd.read_csv(RANKINGS_DIR / "f1_rankings.csv")
    partial_rankings, race_labels, all_drivers = to_partial_rank_lists(rankings_full)
    n_items_partial = len(all_drivers)
    season_labels   = season_labels_from_race_labels(race_labels)

    print(f"  {n_items_partial} drivers, {len(partial_rankings)} races, "
          f"{len(np.unique(season_labels))} seasons")

    print("Precomputing spectral full-rankings …")
    partial_rankings_spectral = build_spectral_full_rankings(partial_rankings, n_items_partial)

    # ── Build full grid ───────────────────────────────────────────────────────
    grid_keys   = list(GRID.keys())
    grid_values = list(GRID.values())
    combos      = list(product(*grid_values))

    total = len(combos)
    print(f"\nGrid: {GRID}")
    print(f"Total runs: {total}\n")

    if dry_run:
        print("Dry-run — printing grid only:")
        for i, combo in enumerate(combos):
            kv = dict(zip(grid_keys, combo))
            print(f"  {i+1:>3}/{total}  {kv}")
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    records:  list[dict]       = []
    z_maps:   list[list[int]]  = []
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = RESULTS_DIR / f"f1_grid_results_{run_ts}.csv"
    npy_path = RESULTS_DIR / f"f1_grid_z_maps_{run_ts}.npy"

    for run_idx, combo in enumerate(combos):
        kv = dict(zip(grid_keys, combo))
        print(f"[{run_idx+1:>3}/{total}]  {kv}  … ", end="", flush=True)

        result = run_one(
            partial_rankings          = partial_rankings,
            partial_rankings_spectral = partial_rankings_spectral,
            n_items_partial           = n_items_partial,
            race_labels               = race_labels,
            season_labels             = season_labels,
            **kv,
        )

        z_maps.append(result.pop("z_map"))
        records.append(result)

        print(
            f"ARI={result['ari']:.3f}  NMI={result['nmi']:.3f}  "
            f"homo={result['homogeneity']:.3f}  comp={result['completeness']:.3f}  "
            f"k={result['n_active_clusters']}  {result['elapsed_s']:.0f}s"
        )

        # ── Incremental save after every run ──────────────────────────────────
        pd.DataFrame(records).to_csv(csv_path, index=False)
        np.save(npy_path, np.array(z_maps, dtype=np.int32))

    # ── Final save (also writes latest copies) ────────────────────────────────
    df = pd.DataFrame(records)
    print(f"\nSaved results: {csv_path}")

    z_arr = np.array(z_maps, dtype=np.int32)
    print(f"Saved z_maps:  {npy_path}  shape={z_arr.shape}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n── Top 10 runs by ARI ──────────────────────────────────────────────")
    top = df.sort_values("ari", ascending=False).head(10)
    print(top[["n_iter", "C", "seed", "ari", "nmi", "homogeneity",
               "completeness", "n_active_clusters", "elapsed_s"]].to_string(index=False))

    print("\n── Mean ARI by n_iter ───────────────────────────────────────────────")
    print(df.groupby("n_iter")["ari"].describe().round(3))

    print("\n── Mean ARI by C ────────────────────────────────────────────────────")
    print(df.groupby("C")["ari"].describe().round(3))

    # Save a fixed-name "latest" symlink-style copy for easy reloading
    latest_csv = RESULTS_DIR / "f1_grid_results_latest.csv"
    df.to_csv(latest_csv, index=False)
    latest_npy = RESULTS_DIR / "f1_grid_z_maps_latest.npy"
    np.save(latest_npy, z_arr)
    print(f"\nLatest copies: {latest_csv}, {latest_npy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 hyperparameter grid search")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the grid without running any MCMC",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
