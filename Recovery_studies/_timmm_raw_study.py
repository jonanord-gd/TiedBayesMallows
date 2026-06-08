"""
TiMMM Raw Recovery Study
=========================
Simulation recovery study that saves the **raw model outputs** alongside
computed metrics — enough data to re-derive any metric offline without
re-running MCMC.

What is saved per scenario
--------------------------
  scenario        : data-generation parameters (C, N, n, bd, θ, seed)
  settings        : MCMC / model hyper-parameters
  ground_truth    : true cluster vector z, per-cluster block structure and θ
  prediction      : predicted z, per-cluster blocks, theta, logp_refined
  metrics         : all six metrics (cluster_ari, cluster_nmi, order_distance,
                    block_ari, block_nmi, theta_rmse) — pre-computed for convenience
  runtime_seconds : wall-clock time for the full scenario

Saved values allow you to compute, e.g.:
  - Any clustering index (ARI, NMI, AMI, V-measure, …) from z_true / z_pred
  - Any block-structure metric from true_blocks / pred_blocks
  - Posterior variability from repeated seeds (30 seeds × 1 restart)

OAT design
----------
  n_clusters   : 2, 5, 10, 15, 20         (default  5)
  N_assessors  : 20, 50, 100, 200, 500, 1000  (default 200)
  n_items      : 5, 10, 20, 50, 100           (default  20)
  block_density: 0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0  (default 0.4)
  theta        : 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0 (default 1.0)
  seeds        : 30 seeds (0–29)  →  27 base scenarios × 30 seeds = 810 runs
  restarts     : 1 per seed (variance comes from seed diversity, not restarts)
  init         : spectral (default) or random (--random flag)

Output layout
-------------
  <out_dir>/
    run_metadata.json          — human-readable run summary
    <scenario_name>.json       — one file per (base_scenario, seed)
    all_results.json           — concatenation of all scenario files

Usage
-----
  python _timmm_raw_study.py                          # full 780-scenario run (sequential)
  python _timmm_raw_study.py --workers 16             # parallel with 16 processes
  python _timmm_raw_study.py --random                 # random initialisation
  python _timmm_raw_study.py --limit 5                # quick smoke-test
  python _timmm_raw_study.py --resume-dir <dir>       # continue interrupted run
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score as _sklearn_nmi

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helper_functions.Generate_mixture_data import generate_mixture_data
from model.TiedMallowsModel import MixtureRankingModel
from model.initialization import init_spectral_with_z


# ─────────────────────────────────────────────────────────────────────────────
# OAT design constants
# ─────────────────────────────────────────────────────────────────────────────

# 30 distinct seeds — deterministic so results are fully reproducible.
SEEDS =  tuple(range(30))   # 0, 1, 2, …, 29

C_VALUES  = (50,) #(5, 20, 50) #(2, 5, 10, 20, 30)
N_VALUES  = (500,) #(20, 100, 500) #(20, 50, 100, 200, 500, 1000)
N_ITEMS   = (200,) #(10, 20, 50, 200) #(5, 10, 20, 50, 100)
BD_VALUES = (0.5,)#(0.1, 0.25, 0.5, 1.0) #(0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
TH_VALUES = (1.0,) #(0.1, 0.25, 0.5, 1.0, 3.0) #(0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)

C_DEF, N_DEF, NI_DEF, BD_DEF, TH_DEF =  50, 500, 200, 0.5, 1.0 #20, 200, 20, 0.5, 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Scenario dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Scenario:
    name: str
    n_clusters: int
    n_assessors: int
    n_items: int
    theta: float
    block_density: float
    seed: int


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def _block_index(blocks: list[list[int]], n_items: int) -> list[int]:
    out = [-1] * n_items
    for block_id, block in enumerate(blocks):
        for item in block:
            out[item] = block_id
    return out


def _normalized_kemeny(
    true_blocks: list[list[int]], pred_blocks: list[list[int]], n_items: int
) -> float:
    max_dist = n_items * (n_items - 1) / 2
    if max_dist == 0:
        return 0.0
    true_idx = _block_index(true_blocks, n_items)
    pred_idx = _block_index(pred_blocks, n_items)
    dist = 0.0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            tr = (true_idx[i] > true_idx[j]) - (true_idx[i] < true_idx[j])
            pr = (pred_idx[i] > pred_idx[j]) - (pred_idx[i] < pred_idx[j])
            if tr == pr:
                continue
            dist += 0.5 if (tr == 0 or pr == 0) else 1.0
    return dist / max_dist


def _adjusted_rand_index(a: list[int], b: list[int]) -> float:
    def _comb2(x: int) -> int:
        return x * (x - 1) // 2

    n = len(a)
    true_ids = sorted(set(a))
    pred_ids = sorted(set(b))
    t2i = {v: i for i, v in enumerate(true_ids)}
    p2i = {v: i for i, v in enumerate(pred_ids)}
    cont = [[0] * len(pred_ids) for _ in range(len(true_ids))]
    for ta, pb in zip(a, b):
        cont[t2i[ta]][p2i[pb]] += 1
    row_sums = [sum(row) for row in cont]
    col_sums = [sum(cont[r][c] for r in range(len(true_ids))) for c in range(len(pred_ids))]
    s_cell = sum(_comb2(cell) for row in cont for cell in row)
    s_row  = sum(_comb2(v) for v in row_sums)
    s_col  = sum(_comb2(v) for v in col_sums)
    total  = _comb2(n)
    if total == 0:
        return 1.0
    expected = (s_row * s_col) / total
    denom    = 0.5 * (s_row + s_col) - expected
    # denom == 0 when both clusterings are trivially degenerate → perfect agreement
    return 1.0 if denom == 0 else (s_cell - expected) / denom


def _align_clusters(
    z_true: list[int], z_pred: list[int], n_true: int, n_fit: int
) -> dict[int, int]:
    active = sorted(set(z_pred))
    active_idx = {lab: j for j, lab in enumerate(active)}
    n_active = len(active)

    # Build overlap matrix: cost[ti, j] = # assessors with true cluster ti
    # assigned to predicted cluster active[j]
    cost = np.zeros((n_true, n_active), dtype=float)
    for zt, zp in zip(z_true, z_pred):
        j = active_idx.get(zp)
        if j is not None:
            cost[zt, j] += 1

    # Pad to square so every true cluster gets a unique assignment
    n_dim = max(n_true, n_active)
    padded = np.zeros((n_true, n_dim), dtype=float)
    padded[:, :n_active] = cost

    row_ind, col_ind = linear_sum_assignment(-padded)  # negate to maximise overlap

    pred_to_true: dict[int, int] = {}
    for ti, ci in zip(row_ind, col_ind):
        if ci < n_active:
            pred_to_true[active[ci]] = ti

    return pred_to_true


def compute_metrics(
    true_blocks: list[list[list[int]]],
    z_true: list[int],
    map_result: dict[str, Any],
    n_true_clusters: int,
    n_items: int,
    true_theta: float,
) -> dict[str, float]:
    z_pred = map_result["z"]
    pred_blocks_all: list[list[list[int]]] = [
        cluster["blocks"] for cluster in map_result["clusters"]
    ]
    n_fit = len(pred_blocks_all)
    n_pred_active = len(set(z_pred))

    # Cluster ARI + NMI
    cluster_ari = _adjusted_rand_index(z_true, z_pred)
    if n_true_clusters == 1 and n_pred_active == 1:
        cluster_ari = 1.0
    cluster_nmi = (
        1.0 if (n_true_clusters == 1 and n_pred_active == 1)
        else float(_sklearn_nmi(z_true, z_pred, average_method="arithmetic"))
    )

    pred_to_true = _align_clusters(z_true, z_pred, n_true_clusters, n_fit)
    true_to_pred = {v: k for k, v in pred_to_true.items()}

    cluster_sizes = [sum(1 for z in z_true if z == c) for c in range(n_true_clusters)]
    total_n = sum(cluster_sizes)
    max_pair_dist = n_items * (n_items - 1) / 2

    w_kemeny    = 0.0
    w_block_ari = 0.0
    w_block_nmi = 0.0
    theta_sq    = 0.0
    theta_wsum  = 0.0

    for tc in range(n_true_clusters):
        pc = true_to_pred.get(tc)
        true_cons = true_blocks[tc]
        w = cluster_sizes[tc] / total_n if total_n else 0.0

        if pc is None:
            w_kemeny += w * (1.0 if max_pair_dist > 0 else 0.0)
        else:
            pred_cons = pred_blocks_all[pc]
            w_kemeny += w * _normalized_kemeny(true_cons, pred_cons, n_items)

            true_bix = _block_index(true_cons, n_items)
            pred_bix = _block_index(pred_cons, n_items)
            w_block_ari += w * _adjusted_rand_index(true_bix, pred_bix)
            w_block_nmi += w * (
                1.0 if len(set(true_bix)) == 1 and len(set(pred_bix)) == 1
                else float(_sklearn_nmi(true_bix, pred_bix, average_method="arithmetic"))
            )

            pred_theta = float(map_result["clusters"][pc]["theta"])
            if not math.isnan(pred_theta):
                theta_sq   += w * (true_theta - pred_theta) ** 2
                theta_wsum += w

    theta_rmse = math.sqrt(theta_sq / theta_wsum) if theta_wsum > 0 else float("nan")

    return {
        "cluster_ari":    cluster_ari,
        "cluster_nmi":    cluster_nmi,
        "order_distance": w_kemeny,
        "block_ari":      w_block_ari,
        "block_nmi":      w_block_nmi,
        "theta_rmse":     theta_rmse,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main scenario runner
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_record(
    scenario: Scenario,
    *,
    n_iter: int,
    burn_in: int,
    thin: int,
    use_spectral: bool,
) -> dict[str, Any]:
    """
    Generate data, fit one TiMMM chain, return full raw record.

    The record contains:
      ground_truth  — z_true (list[int]), true_blocks (list of cluster block lists),
                      true_theta (float applied uniformly across clusters)
      prediction    — z_pred (list[int]), pred_clusters (list of dicts with
                      'cluster_id', 'blocks', 'theta', 'size'), logp_refined
      metrics       — pre-computed convenience metrics
    """
    t0 = time.time()

    true_blocks, _tau_true, z_true, rankings = generate_mixture_data(
        n_assessors=scenario.n_assessors,
        n_items=scenario.n_items,
        C=scenario.n_clusters,
        seed=scenario.seed,
        theta=scenario.theta,
        block_density=scenario.block_density,
    )

    n_fit = 2 * scenario.n_clusters
    init_mu = [0.001]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_spectral:
            init_clusters, init_z = init_spectral_with_z(
                rankings, n_fit, seed=scenario.seed, py_sampling=False
            )
            model = MixtureRankingModel(
                rankings,
                init_clusters=init_clusters,
                init_z=init_z,
                init_mu=init_mu,
                seed=scenario.seed,
                verbose=False,
            )
        else:
            model = MixtureRankingModel(
                rankings,
                n_clusters=n_fit,
                init_mu=init_mu,
                seed=scenario.seed,
                verbose=False,
            )

    _, samples = model.run_mcmc(
        n_iter=n_iter,
        burn_in=burn_in,
        thin=thin,
        save_samples=True,
        save_tau=True,
        save_theta=True,
        save_logp=True,
        use_py_prior=False,
        include_order_prior=False,
        use_annealing=True,
        annealing_schedule_type="linear",
        annealing_plateau_frac=0.5,
        temp_min=0.1,
        temp_max=1.0,
    )
    result = model.find_map(samples, refine=True, verbose=False)

    # ── Ground truth ──────────────────────────────────────────────────────────
    # true_blocks[c] = ordered list of blocks for cluster c;
    # each block is a list of item indices.
    ground_truth = {
        "z":          z_true,                # list[int], length N_assessors
        "clusters": [
            {
                "cluster_id": c,
                "blocks":     true_blocks[c],   # list[list[int]]
                "theta":      scenario.theta,
                "size":       int(sum(1 for z in z_true if z == c)),
            }
            for c in range(scenario.n_clusters)
        ],
    }

    # ── Prediction ────────────────────────────────────────────────────────────
    # Include only non-empty predicted clusters for compactness.
    z_pred = result["z"]
    pred_cluster_counts = {}
    for z in z_pred:
        pred_cluster_counts[z] = pred_cluster_counts.get(z, 0) + 1

    prediction = {
        "z":            z_pred,              # list[int], length N_assessors
        "logp_refined": result["logp_refined"],
        "clusters": [
            {
                "cluster_id": cid,
                "blocks":     result["clusters"][cid]["blocks"],
                "theta":      float(result["clusters"][cid]["theta"]),
                "size":       pred_cluster_counts.get(cid, 0),
            }
            for cid in sorted(pred_cluster_counts.keys())
        ],
    }

    metrics = compute_metrics(
        true_blocks=true_blocks,
        z_true=z_true,
        map_result=result,
        n_true_clusters=scenario.n_clusters,
        n_items=scenario.n_items,
        true_theta=scenario.theta,
    )

    return {
        "scenario":        asdict(scenario),
        "settings": {
            "n_iter":          n_iter,
            "burn_in":         burn_in,
            "thin":            thin,
            "n_restarts":      1,
            "fit_n_clusters":  n_fit,
            "init_strategy":   "spectral" if use_spectral else "random",
        },
        "ground_truth":    ground_truth,
        "prediction":      prediction,
        "metrics":         metrics,
        "runtime_seconds": time.time() - t0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OAT scenario design  (same grid, 30 seeds)
# ─────────────────────────────────────────────────────────────────────────────

def oat_scenarios() -> list[Scenario]:
    def _name(c: int, n: int, ni: int, bd: float, th: float) -> str:
        bd_str = f"{bd:.2f}".replace(".", "p")
        th_str = str(th).replace(".", "p")
        return f"c{c}_n{n}_ni{ni}_bd{bd_str}_theta{th_str}"

    seen: set[tuple] = set()
    base: list[tuple] = []

    def _add(c: int, n: int, ni: int, bd: float, th: float) -> None:
        key = (c, n, ni, bd, th)
        if key not in seen:
            seen.add(key)
            base.append((_name(c, n, ni, bd, th), c, n, ni, bd, th))

    for c  in C_VALUES:  _add(c,     N_DEF, NI_DEF, BD_DEF, TH_DEF)
    for n  in N_VALUES:  _add(C_DEF, n,     NI_DEF, BD_DEF, TH_DEF)
    for ni in N_ITEMS:   _add(C_DEF, N_DEF, ni,     BD_DEF, TH_DEF)
    for bd in BD_VALUES: _add(C_DEF, N_DEF, NI_DEF, bd,     TH_DEF)
    for th in TH_VALUES: _add(C_DEF, N_DEF, NI_DEF, BD_DEF, th)

    scenarios: list[Scenario] = []
    for name, c, n, ni, bd, th in base:
        for seed in SEEDS:
            scenarios.append(Scenario(
                name=f"{name}_seed{seed:02d}",
                n_clusters=c,
                n_assessors=n,
                n_items=ni,
                theta=th,
                block_density=bd,
                seed=seed,
            ))
    return scenarios


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker (must be at module scope to be picklable by multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args_tuple: tuple) -> dict[str, Any] | None:
    scenario, n_iter, burn_in, thin, use_spectral = args_tuple
    try:
        return fit_and_record(
            scenario,
            n_iter=n_iter,
            burn_in=burn_in,
            thin=thin,
            use_spectral=use_spectral,
        )
    except Exception as exc:
        print(f"ERROR {scenario.name}: {exc}", flush=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TiMMM OAT study — saves raw cluster assignments, block structures, "
            "and theta values alongside metrics. 30 seeds × 27 base scenarios = 810 runs."
        )
    )
    parser.add_argument("--n-iter",    type=int,  default=10000, help="MCMC iterations.")
    parser.add_argument("--burn-in",   type=int,  default=5000,  help="Burn-in iterations.")
    parser.add_argument("--thin",      type=int,  default=1,    help="Thinning interval.")
    parser.add_argument("--random",    action="store_true",      help="Use random initialisation (default is spectral).")
    parser.add_argument("--limit",     type=int,  default=None,  help="Cap number of scenarios (smoke-test).")
    parser.add_argument("--workers",   type=int,  default=1,     help="Number of parallel worker processes (default: 1 = sequential).")
    parser.add_argument("--out-dir",   type=str,  default=None,  help="Override output directory.")
    parser.add_argument(
        "--resume-dir", type=str, default=None,
        help="Resume from an existing output directory (skips already-completed scenarios).",
    )
    args = parser.parse_args()

    scenarios = oat_scenarios()
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    init_tag = "random" if args.random else "spectral"

    # ── Resume or fresh start ─────────────────────────────────────────────────
    if args.resume_dir:
        out_dir = Path(args.resume_dir)
        if not out_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {out_dir}")
        done = {
            p.stem for p in out_dir.glob("*.json")
            if p.name not in {"run_metadata.json", "all_results.json"}
        }
        completed: list[dict] = [
            json.loads((out_dir / f"{s.name}.json").read_text(encoding="utf-8"))
            for s in scenarios if s.name in done
        ]
        scenarios = [s for s in scenarios if s.name not in done]
        print(f"Resuming : {out_dir}")
        print(f"  Done   : {len(completed)},  remaining: {len(scenarios)}")
    else:
        if args.out_dir:
            out_dir = Path(args.out_dir)
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_dir = Path("simulation_recovery_runs") / f"recovery_timmm_raw_{init_tag}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        completed = []

        # Write human-readable metadata immediately so the directory is self-describing
        metadata = {
            "study":           "TiMMM OAT — raw outputs (z, blocks, theta)",
            "description": (
                "Each scenario JSON contains the full ground-truth cluster assignment "
                "vector and block structure, plus the MAP prediction (z, blocks, theta) "
                "from a single MCMC chain.  Metrics are pre-computed for convenience "
                "but all inputs are present to re-derive any index offline."
            ),
            "oat_design": {
                "n_clusters":    list(C_VALUES),
                "n_assessors":   list(N_VALUES),
                "n_items":       list(N_ITEMS),
                "block_density": list(BD_VALUES),
                "theta":         list(TH_VALUES),
                "defaults": {
                    "n_clusters": C_DEF, "n_assessors": N_DEF,
                    "n_items": NI_DEF, "block_density": BD_DEF, "theta": TH_DEF,
                },
            },
            "seeds":           list(SEEDS),
            "n_seeds":         len(SEEDS),
            "n_base_scenarios": len(set(s.name.rsplit("_seed", 1)[0] for s in oat_scenarios())),
            "total_scenarios": len(oat_scenarios()),
            "mcmc": {
                "n_iter":         args.n_iter,
                "burn_in":        args.burn_in,
                "thin":           args.thin,
                "n_restarts":     1,
                "init_strategy":  init_tag,
            },
            "metrics_stored": [
                "cluster_ari", "cluster_nmi", "order_distance",
                "block_ari", "block_nmi", "theta_rmse",
            ],
            "raw_fields_stored": [
                "ground_truth.z",
                "ground_truth.clusters[*].blocks",
                "ground_truth.clusters[*].theta",
                "prediction.z",
                "prediction.clusters[*].blocks",
                "prediction.clusters[*].theta",
                "prediction.logp_refined",
            ],
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (out_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"Output  : {out_dir}")
        print(
            f"  {len(scenarios)} scenarios  "
            f"(n_iter={args.n_iter}, burn_in={args.burn_in}, "
            f"thin={args.thin}, init={init_tag})"
        )

    # ── Run scenarios ─────────────────────────────────────────────────────────
    n_total = len(scenarios) + len(completed)
    results = list(completed)
    n_workers = args.workers
    use_spectral = not args.random
    worker_args = [
        (s, args.n_iter, args.burn_in, args.thin, use_spectral)
        for s in scenarios
    ]

    def _handle_record(record: dict[str, Any] | None, scenario: Scenario, i: int) -> None:
        if record is None:
            return
        sc = scenario
        m = record["metrics"]
        print(
            f"[{i:>4}/{n_total}]  C={sc.n_clusters:>2}  N={sc.n_assessors:>4}"
            f"  n={sc.n_items:>3}  bd={sc.block_density:.2f}"
            f"  θ={sc.theta:>5}  seed={sc.seed:>2}  …  "
            f"ARI={m['cluster_ari']:.3f}  NMI={m['cluster_nmi']:.3f}"
            f"  bARI={m['block_ari']:.3f}  θRMSE={m['theta_rmse']:.3f}"
            f"  ({record['runtime_seconds']:.1f}s)",
            flush=True,
        )
        (out_dir / f"{scenario.name}.json").write_text(
            json.dumps(_to_jsonable(record), indent=2), encoding="utf-8"
        )
        results.append(record)

    if n_workers > 1:
        print(f"  Running with {n_workers} parallel workers.")
        with mp.Pool(processes=n_workers) as pool:
            for i, (record, scenario) in enumerate(
                zip(pool.imap(_worker, worker_args), scenarios),
                start=len(completed) + 1,
            ):
                _handle_record(record, scenario, i)
    else:
        for i, (worker_arg, scenario) in enumerate(
            zip(worker_args, scenarios), start=len(completed) + 1
        ):
            record = _worker(worker_arg)
            _handle_record(record, scenario, i)

    # ── Save aggregate ────────────────────────────────────────────────────────
    (out_dir / "all_results.json").write_text(
        json.dumps(_to_jsonable(results), indent=2), encoding="utf-8"
    )

    if not results:
        print("No results to summarise.")
        return

    n = len(results)
    metrics_all = [r["metrics"] for r in results]
    print(f"\n{'─' * 70}")
    print(f"  Scenarios completed : {n}/{n_total}")
    for key, label in [
        ("cluster_ari",    "Cluster ARI   "),
        ("cluster_nmi",    "Cluster NMI   "),
        ("order_distance", "Order distance"),
        ("block_ari",      "Block ARI     "),
        ("block_nmi",      "Block NMI     "),
        ("theta_rmse",     "θ RMSE        "),
    ]:
        vals = [m[key] for m in metrics_all if not math.isnan(m[key])]
        if vals:
            print(f"  {label}: {sum(vals) / len(vals):.3f}")
    print(f"{'─' * 70}")
    print(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()