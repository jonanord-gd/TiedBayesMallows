"""TiMMM convergence study — slow convergence vs local minima diagnosis.

For scenarios where spectral init outperforms random init in the OAT study,
this script runs BOTH initialisations with a **long** MCMC chain and evaluates
recovery metrics at multiple **checkpoints** along the *same* chain.

Interpretation
--------------
  * If random init improves steadily toward spectral init performance as the
    chain grows → **slow convergence** (not stuck, just needs more samples).
  * If random init plateaus at an inferior value even with a long chain →
    **local minima** (random init is stuck, spectral init escapes).

Each checkpoint evaluates the MAP using only the first k saved samples (a
prefix of the full chain), so both short-chain and long-chain assessments come
from the *same single run* — eliminating run-to-run variance as a confounder.

Usage
-----
  python _timmm_convergence_study.py \\
      --rand-dir  simulation_recovery_runs/recovery_timmm_oat_... \\
      --spec-dir  simulation_recovery_runs/recovery_timmm_spectral_... \\
      [--n-hard 20]             # number of hard base scenarios to study
      [--n-iter 30000]          # total MCMC iterations (long chain)
      [--burn-in 5000]          # warm-up iterations
      [--thin 20]               # thinning factor
      [--n-restarts 1]          # restarts per run (1 = single long run)
      [--checkpoints 0.25 0.5 0.75 1.0]  # chain-fraction checkpoints
      [--seeds 42 123 999]      # which seeds of each base scenario to run
      [--limit N]               # cap total runs (useful for smoke-testing)
      [--resume-dir DIR]        # continue an interrupted run
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import normalized_mutual_info_score as _sklearn_nmi

from helper_functions.Generate_mixture_data import generate_mixture_data
from model.TiedMallowsModel import MixtureRankingModel
from model.dataclasses import MCMCSamples
from model.initialization import init_spectral_with_z


# ─────────────────────────────────────────────────────────────────────────────
# Scenario dataclass  (same as _timmm_oat_study.py)
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
# Metric helpers  (reproduced here to keep this script self-contained)
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
    return 1.0 if denom == 0 else (s_cell - expected) / denom


def _align_clusters(
    z_true: list[int], z_pred: list[int], n_true: int, n_fit: int
) -> dict[int, int]:
    conf = [[0] * n_fit for _ in range(n_true)]
    for zt, zp in zip(z_true, z_pred):
        if 0 <= zp < n_fit:
            conf[zt][zp] += 1
    active = sorted(set(z_pred))
    dummies = [-(i + 1) for i in range(max(0, n_true - len(active)))]
    cands = active + dummies

    @lru_cache(maxsize=None)
    def _best(ti: int, used: int) -> int:
        if ti == n_true:
            return 0
        best_val = -1
        for pos, lab in enumerate(cands):
            if used & (1 << pos):
                continue
            gain = conf[ti][lab] if lab >= 0 else 0
            total = gain + _best(ti + 1, used | (1 << pos))
            if total > best_val:
                best_val = total
        return best_val

    pred_to_true: dict[int, int] = {}
    used = 0
    for ti in range(n_true):
        best_pos = best_lab = best_val = -1
        for pos, lab in enumerate(cands):
            if used & (1 << pos):
                continue
            gain = conf[ti][lab] if lab >= 0 else 0
            total = gain + _best(ti + 1, used | (1 << pos))
            if total > best_val:
                best_val = total
                best_pos = pos
                best_lab = lab
        used |= 1 << best_pos
        if best_lab >= 0:
            pred_to_true[best_lab] = ti
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

    cluster_ari = _adjusted_rand_index(z_true, z_pred)
    if n_true_clusters == 1 and n_pred_active == 1:
        cluster_ari = 1.0

    pred_to_true = _align_clusters(z_true, z_pred, n_true_clusters, n_fit)
    true_to_pred = {v: k for k, v in pred_to_true.items()}

    cluster_sizes = [sum(1 for z in z_true if z == c) for c in range(n_true_clusters)]
    total_n = sum(cluster_sizes)
    max_pair_dist = n_items * (n_items - 1) / 2

    w_kemeny    = 0.0
    w_block_ari = 0.0
    theta_sq    = 0.0
    theta_wsum  = 0.0

    for tc in range(n_true_clusters):
        pc = true_to_pred.get(tc)
        true_cons = true_blocks[tc]
        w = cluster_sizes[tc] / total_n if total_n else 0.0

        if pc is None:
            w_kemeny    += w * (1.0 if max_pair_dist > 0 else 0.0)
            w_block_ari += w * 0.0
        else:
            pred_cons = pred_blocks_all[pc]
            w_kemeny += w * _normalized_kemeny(true_cons, pred_cons, n_items)
            true_bix = _block_index(true_cons, n_items)
            pred_bix = _block_index(pred_cons, n_items)
            w_block_ari += w * _adjusted_rand_index(true_bix, pred_bix)
            pred_theta = float(map_result["clusters"][pc]["theta"])
            if not math.isnan(pred_theta):
                theta_sq   += w * (true_theta - pred_theta) ** 2
                theta_wsum += w

    theta_rmse = math.sqrt(theta_sq / theta_wsum) if theta_wsum > 0 else float("nan")

    return {
        "cluster_ari":    cluster_ari,
        "order_distance": w_kemeny,
        "block_ari":      w_block_ari,
        "theta_rmse":     theta_rmse,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Samples slicing helper
# ─────────────────────────────────────────────────────────────────────────────

def slice_samples(samples: MCMCSamples, n: int) -> MCMCSamples:
    """Return a new MCMCSamples containing only the first *n* saved draws."""
    return MCMCSamples(
        z_samples      = samples.z_samples[:n],
        blocks_samples = samples.blocks_samples[:n],
        tau_samples    = samples.tau_samples[:n]    if samples.tau_samples    else None,
        theta_samples  = samples.theta_samples[:n]  if samples.theta_samples  else None,
        logp           = samples.logp[:n]            if samples.logp           else None,
        K              = samples.K[:n]               if samples.K              else None,
        saved_iterations = (
            samples.saved_iterations[:n] if samples.saved_iterations else None
        ),
        theta_jump     = samples.theta_jump,
        theta_accepts  = samples.theta_accepts[:n]  if samples.theta_accepts  else None,
        block_accepts  = samples.block_accepts[:n]  if samples.block_accepts  else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hard-scenario identification
# ─────────────────────────────────────────────────────────────────────────────

def _strip_seed(name: str) -> str:
    return name.rsplit("_seed", 1)[0] if "_seed" in name else name


def _load_oat_results(run_dir: Path) -> dict[str, dict[str, float]]:
    """Load OAT results into {scenario_name: metrics_dict}."""
    all_path = run_dir / "all_results.json"
    if all_path.exists():
        data = json.loads(all_path.read_text(encoding="utf-8"))
    else:
        files = sorted([
            p for p in run_dir.glob("*.json")
            if p.name not in {"run_metadata.json", "all_results.json"}
        ])
        data = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    return {r["scenario"]["name"]: r for r in data}


def identify_hard_scenarios(
    rand_dir: Path,
    spec_dir: Path,
    *,
    n_hard: int = 20,
    gap_metric: str = "cluster_ari",   # metric to rank by
    min_gap: float = 0.0,              # only consider scenarios with gap > this
) -> list[tuple[str, dict[str, Any], float]]:
    """Return [(base_scenario_name, scenario_dict, signed_gap), ...].

    Scenarios are ranked by *signed* improvement (spectral − random), using the
    chosen gap_metric.  For cluster_ari and block_ari, higher = better so
    gap = spec − rand > 0 means spectral wins.  For order_distance, lower =
    better so gap = rand − spec > 0 means spectral wins.  In both cases,
    positive gap_value means spectral is better.
    """
    higher_better = gap_metric in {"cluster_ari", "block_ari"}

    rand_results = _load_oat_results(rand_dir)
    spec_results = _load_oat_results(spec_dir)

    common_names = set(rand_results.keys()) & set(spec_results.keys())

    # Aggregate per base scenario (average over seeds)
    base_gaps: dict[str, list[float]] = {}
    base_scenario_info: dict[str, dict[str, Any]] = {}
    for name in sorted(common_names):
        base = _strip_seed(name)
        rm  = rand_results[name]["metrics"][gap_metric]
        sm  = spec_results[name]["metrics"][gap_metric]
        gap = (sm - rm) if higher_better else (rm - sm)
        base_gaps.setdefault(base, []).append(gap)
        # Store scenario metadata from random run (same parameters either way)
        base_scenario_info[base] = rand_results[name]["scenario"]

    # Rank base scenarios by mean gap (descending)
    ranked = sorted(
        [(base, base_scenario_info[base], float(np.mean(gaps)))
         for base, gaps in base_gaps.items()],
        key=lambda t: t[2],
        reverse=True,
    )
    # Keep only those with gap > min_gap
    ranked = [(b, s, g) for b, s, g in ranked if g > min_gap]
    return ranked[:n_hard]


# ─────────────────────────────────────────────────────────────────────────────
# Scenario reconstruction from metadata
# ─────────────────────────────────────────────────────────────────────────────

def _scenario_from_dict(d: dict[str, Any]) -> Scenario:
    return Scenario(
        name          = d["name"],
        n_clusters    = int(d["n_clusters"]),
        n_assessors   = int(d["n_assessors"]),
        n_items       = int(d["n_items"]),
        theta         = float(d["theta"]),
        block_density = float(d["block_density"]),
        seed          = int(d["seed"]),
    )


def _build_convergence_scenarios(
    hard_base_scenarios: list[tuple[str, dict[str, Any], float]],
    seeds: list[int],
) -> list[tuple[Scenario, float]]:
    """Expand hard base scenarios over the requested seeds, paired with their gap."""
    out: list[tuple[Scenario, float]] = []
    for base_name, base_info, gap in hard_base_scenarios:
        for seed in seeds:
            sc = Scenario(
                name          = f"{base_name}_seed{seed}",
                n_clusters    = int(base_info["n_clusters"]),
                n_assessors   = int(base_info["n_assessors"]),
                n_items       = int(base_info["n_items"]),
                theta         = float(base_info["theta"]),
                block_density = float(base_info["block_density"]),
                seed          = seed,
            )
            out.append((sc, gap))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model fitting with checkpoint evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _fitted_k(n_true: int) -> int:
    return 2 * n_true


def fit_with_checkpoints(
    scenario: Scenario,
    *,
    n_iter: int,
    burn_in: int,
    thin: int,
    n_restarts: int,
    checkpoints: list[float],   # fractions of saved samples, e.g. [0.25, 0.5, 0.75, 1.0]
    use_spectral: bool = False,
    true_blocks: list[list[list[int]]],
    z_true: list[int],
    rankings: list[Any],
) -> dict[str, Any]:
    """Fit TiMMM with a long chain and evaluate MAP at multiple chain-prefix checkpoints.

    Returns a dict with 'checkpoints' list, each entry containing:
      - fraction      : chain fraction
      - n_samples     : number of saved samples used
      - metrics       : {cluster_ari, order_distance, block_ari, theta_rmse}
      - best_logp     : log-joint of the MAP sample at this checkpoint
    """
    n_fit = _fitted_k(scenario.n_clusters)
    init_mu = [0.001]

    best_model: MixtureRankingModel | None = None
    best_samples: MCMCSamples | None = None
    best_logp_val: float = float("-inf")

    for restart in range(n_restarts):
        rseed = scenario.seed + 1000 * restart
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if use_spectral:
                init_clusters, init_z = init_spectral_with_z(
                    rankings, n_fit, seed=rseed, py_sampling=False
                )
                model = MixtureRankingModel(
                    rankings,
                    init_clusters=init_clusters,
                    init_z=init_z,
                    init_mu=init_mu,
                    seed=rseed,
                    verbose=False,
                )
            else:
                model = MixtureRankingModel(
                    rankings,
                    n_clusters=n_fit,
                    init_mu=init_mu,
                    seed=rseed,
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
        assert samples is not None

        # Track the restart with the best MAP log-joint at the end of the chain
        if samples.logp:
            run_best_logp = max(samples.logp)
            if run_best_logp > best_logp_val:
                best_logp_val = run_best_logp
                best_model = model
                best_samples = samples

    assert best_model is not None and best_samples is not None

    total_saved = len(best_samples.logp) if best_samples.logp else 0

    checkpoint_results: list[dict[str, Any]] = []
    for frac in sorted(set(checkpoints)):
        # Ensure at least 1 sample at any checkpoint
        n_keep = max(1, int(round(frac * total_saved)))
        prefix = slice_samples(best_samples, n_keep)

        map_result = best_model.find_map(prefix, refine=True, verbose=False)
        metrics = compute_metrics(
            true_blocks=true_blocks,
            z_true=z_true,
            map_result=map_result,
            n_true_clusters=scenario.n_clusters,
            n_items=scenario.n_items,
            true_theta=scenario.theta,
        )
        best_logp_at_cp = float(max(prefix.logp)) if prefix.logp else float("nan")
        checkpoint_results.append({
            "fraction":   frac,
            "n_samples":  n_keep,
            "metrics":    metrics,
            "best_logp":  best_logp_at_cp,
            "logp_refined": map_result.get("logp_refined", float("nan")),
        })

    return {
        "total_saved_samples": total_saved,
        "checkpoints": checkpoint_results,
        "n_iter": n_iter,
        "burn_in": burn_in,
        "thin": thin,
        "n_restarts": n_restarts,
    }


def run_convergence_scenario(
    scenario: Scenario,
    *,
    n_iter: int,
    burn_in: int,
    thin: int,
    n_restarts: int,
    checkpoints: list[float],
    use_spectral: bool = False,
) -> dict[str, Any]:
    """Generate data once, then run fit_with_checkpoints."""
    true_blocks, tau_true, z_true, rankings = generate_mixture_data(
        n_assessors   = scenario.n_assessors,
        n_items       = scenario.n_items,
        C             = scenario.n_clusters,
        seed          = scenario.seed,
        theta         = scenario.theta,
        block_density = scenario.block_density,
    )

    result = fit_with_checkpoints(
        scenario,
        n_iter=n_iter,
        burn_in=burn_in,
        thin=thin,
        n_restarts=n_restarts,
        checkpoints=checkpoints,
        use_spectral=use_spectral,
        true_blocks=true_blocks,
        z_true=z_true,
        rankings=rankings,
    )
    return result


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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TiMMM convergence study — slow convergence vs local minima."
    )
    parser.add_argument(
        "--rand-dir", type=str, required=True,
        help="Path to the random-init OAT run directory.",
    )
    parser.add_argument(
        "--spec-dir", type=str, required=True,
        help="Path to the spectral-init OAT run directory.",
    )
    parser.add_argument("--n-hard",     type=int, default=20,
                        help="Number of hard base scenarios to study.")
    parser.add_argument("--gap-metric", type=str, default="cluster_ari",
                        choices=["cluster_ari", "order_distance", "block_ari"],
                        help="Metric used to rank scenarios by spectral-vs-random gap.")
    parser.add_argument("--min-gap",    type=float, default=0.0,
                        help="Minimum signed gap to include a scenario.")
    parser.add_argument("--n-iter",     type=int, default=30000,
                        help="Total MCMC iterations (long chain).")
    parser.add_argument("--burn-in",    type=int, default=5000,
                        help="Warm-up iterations.")
    parser.add_argument("--thin",       type=int, default=20,
                        help="Thinning factor.")
    parser.add_argument("--n-restarts", type=int, default=1,
                        help="Number of restarts per (scenario, init_strategy) pair. "
                             "Best MAP across restarts is used for checkpoint evaluation.")
    parser.add_argument(
        "--checkpoints", type=float, nargs="+",
        default=[0.1, 0.25, 0.5, 0.75, 1.0],
        help="Chain-fraction checkpoints at which to evaluate MAP.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+",
        default=[42, 123, 999],
        help="Which data seeds of each base scenario to run.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap total (scenario, init) pairs (useful for smoke-testing).",
    )
    parser.add_argument(
        "--resume-dir", type=str, default=None,
        help="Resume an interrupted run from this directory.",
    )
    args = parser.parse_args()

    rand_dir = Path(args.rand_dir)
    spec_dir = Path(args.spec_dir)
    checkpoints = sorted(set(args.checkpoints))

    # ── Identify hard scenarios ────────────────────────────────────────────────
    print("Identifying hard scenarios …")
    hard = identify_hard_scenarios(
        rand_dir, spec_dir,
        n_hard=args.n_hard,
        gap_metric=args.gap_metric,
        min_gap=args.min_gap,
    )
    print(f"  Found {len(hard)} hard base scenarios (gap_metric={args.gap_metric}):")
    for rank, (base, sc_info, gap) in enumerate(hard, 1):
        print(f"    {rank:>2}. {base}  gap={gap:+.4f}")

    # ── Build full scenario list ───────────────────────────────────────────────
    scenario_pairs = _build_convergence_scenarios(hard, args.seeds)
    # Each (scenario, gap) pair will be run for BOTH init strategies
    init_strategies = [("random", False), ("spectral", True)]

    # Build full work list: (scenario, gap, init_label, use_spectral)
    work_list = [
        (sc, gap, init_label, use_spec)
        for sc, gap in scenario_pairs
        for init_label, use_spec in init_strategies
    ]
    if args.limit is not None:
        work_list = work_list[: args.limit]

    # ── Resume mode ────────────────────────────────────────────────────────────
    if args.resume_dir:
        out_dir = Path(args.resume_dir)
        if not out_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {out_dir}")
        done_keys: set[str] = {
            p.stem for p in out_dir.glob("*.json")
            if p.name not in {"run_metadata.json", "all_results.json"}
        }
        completed: list[dict[str, Any]] = [
            json.loads((out_dir / f"{k}.json").read_text(encoding="utf-8"))
            for k in done_keys
        ]
        work_list = [
            w for w in work_list
            if f"{w[0].name}__{w[2]}" not in done_keys
        ]
        print(f"\nResuming: {out_dir}")
        print(f"  Already done: {len(completed)},  remaining: {len(work_list)}")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("simulation_recovery_runs") / f"recovery_timmm_convergence_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        completed = []
        metadata = {
            "mode": "timmm_convergence",
            "rand_dir": str(rand_dir),
            "spec_dir": str(spec_dir),
            "gap_metric": args.gap_metric,
            "n_hard": args.n_hard,
            "n_iter": args.n_iter,
            "burn_in": args.burn_in,
            "thin": args.thin,
            "n_restarts": args.n_restarts,
            "checkpoints": checkpoints,
            "seeds": args.seeds,
            "hard_scenarios": [
                {"base": base, "gap": float(gap), "scenario": sc_info}
                for base, sc_info, gap in hard
            ],
            "total_work_items": len(work_list),
        }
        (out_dir / "run_metadata.json").write_text(
            json.dumps(_to_jsonable(metadata), indent=2), encoding="utf-8"
        )
        print(f"\nOutput: {out_dir}")
        print(f"  {len(work_list)} (scenario × init) pairs to run")
        print(f"  Checkpoints: {checkpoints}")
        print(f"  n_iter={args.n_iter}, burn_in={args.burn_in}, thin={args.thin}")

    # ── Run ────────────────────────────────────────────────────────────────────
    n_total = len(work_list) + len(completed)
    results = list(completed)

    for i, (scenario, gap, init_label, use_spectral) in enumerate(work_list, start=len(completed) + 1):
        sc = scenario
        print(
            f"[{i:>3}/{n_total}]  C={sc.n_clusters:>2}  N={sc.n_assessors:>4}"
            f"  n={sc.n_items:>3}  bd={sc.block_density:.2f}"
            f"  θ={sc.theta:>5}  seed={sc.seed}"
            f"  init={init_label}  gap={gap:+.3f}",
            end="  …  ",
            flush=True,
        )
        t0 = time.time()
        try:
            cp_result = run_convergence_scenario(
                scenario,
                n_iter=args.n_iter,
                burn_in=args.burn_in,
                thin=args.thin,
                n_restarts=args.n_restarts,
                checkpoints=checkpoints,
                use_spectral=use_spectral,
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        runtime = time.time() - t0
        # Print metrics at shortest and longest checkpoint
        cps = cp_result["checkpoints"]
        if cps:
            short = cps[0]["metrics"]
            full  = cps[-1]["metrics"]
            print(
                f"ARI: {short['cluster_ari']:.3f}→{full['cluster_ari']:.3f}"
                f"  ord: {short['order_distance']:.3f}→{full['order_distance']:.3f}"
                f"  ({runtime:.1f}s)"
            )

        run_record = {
            "scenario":      asdict(sc),
            "init_strategy": init_label,
            "oat_gap":       float(gap),
            "gap_metric":    args.gap_metric,
            "convergence":   cp_result,
            "runtime_seconds": runtime,
        }
        file_key = f"{sc.name}__{init_label}"
        (out_dir / f"{file_key}.json").write_text(
            json.dumps(_to_jsonable(run_record), indent=2), encoding="utf-8"
        )
        results.append(run_record)

    # ── Aggregate ──────────────────────────────────────────────────────────────
    (out_dir / "all_results.json").write_text(
        json.dumps(_to_jsonable(results), indent=2), encoding="utf-8"
    )

    if not results:
        print("No results to summarise.")
        return

    print(f"\n{'─' * 60}")
    print(f"  Pairs completed : {len(results)}/{n_total}")
    # Summarise mean ARI improvement (shortest→longest checkpoint)
    for init_label in ["random", "spectral"]:
        subset = [r for r in results if r["init_strategy"] == init_label]
        if not subset:
            continue
        ari_gains: list[float] = []
        for r in subset:
            cps = r["convergence"]["checkpoints"]
            if len(cps) >= 2:
                ari_gains.append(cps[-1]["metrics"]["cluster_ari"] - cps[0]["metrics"]["cluster_ari"])
        if ari_gains:
            print(
                f"  {init_label:>8} init — mean ARI gain "
                f"(short→long): {np.mean(ari_gains):+.4f}"
            )
    print(f"  Results saved to: {out_dir}")
    print(f"{'─' * 60}")


if __name__ == "__main__":
    main()
