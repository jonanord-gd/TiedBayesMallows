"""TiMMM OAT recovery study — clean, focused metrics.

Metrics (four only)
-------------------
  cluster_ari     : Adjusted Rand Index between true and predicted cluster
                    assignments.  Measures cluster recovery quality.
  order_distance  : Weighted normalised Kemeny distance between true and
                    predicted weak-order consensuses (∈ [0, 1], lower is
                    better).  Captures both ordering error and block membership
                    simultaneously.  The complement 1 − order_distance is the
                    "order score".
  block_ari       : Cluster-size-weighted ARI between true and predicted block
                    memberships within each matched cluster pair.  Measures
                    block structure recovery *ignoring* the order of blocks.
  theta_rmse      : sqrt(Σ_c w_c · (θ_true − θ̂_c)²) where the sum is over
                    matched (true cluster, predicted cluster) pairs and w_c is
                    the fraction of assessors in true cluster c.  Measures
                    precision-parameter recovery.

Model assumptions
-----------------
  - No spectral pre-processing; no consensus clustering.
  - Fitted clusters : 2 × C_true (deliberate over-saturation).
  - Initialisation  : MixtureRankingModel default (random start built-in).
  - No PY prior, no block-order prior.
  - Annealing       : linear, temp_min=0.1 → temp_max=1.0 over burn-in.
  - Best MAP across n_restarts independent restarts is kept.

OAT design
----------
  n_clusters   : 1, 2, 5, 10, 15, 20         (default  5)
  N_assessors  : 20, 50, 100, 200, 500, 1000  (default 200)
  n_items      : 5, 10, 20, 50, 100           (default  20)
  block_density: 0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0  (default 0.4)
  theta        : 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0 (default 1.0)
  seeds        : 42, 123, 999, 7, 13, 17, 31, 53, 97, 137  →  27 base scenarios × 10 seeds = 270 runs

Usage
-----
  python _timmm_oat_study.py                     # full 81-scenario run
  python _timmm_oat_study.py --limit 5           # quick smoke-test
  python _timmm_oat_study.py --resume-dir <dir>  # continue interrupted run
"""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import normalized_mutual_info_score as _sklearn_nmi

from helper_functions.Generate_mixture_data import generate_mixture_data
from model.TiedMallowsModel import MixtureRankingModel
from model.initialization import init_spectral_with_z


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
    """Map each item to its block index (0-based in block order)."""
    out = [-1] * n_items
    for block_id, block in enumerate(blocks):
        for item in block:
            out[item] = block_id
    return out


def _normalized_kemeny(
    true_blocks: list[list[int]], pred_blocks: list[list[int]], n_items: int
) -> float:
    """p=½ Kemeny distance between two weak orders, normalised to [0, 1]."""
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
    """Adjusted Rand Index between two flat label vectors."""
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
    # denom == 0 when clustering is trivially degenerate (all-in-one or all-
    # singletons); in that case numerator is also 0 and agreement is perfect.
    return 1.0 if denom == 0 else (s_cell - expected) / denom


def _align_clusters(
    z_true: list[int], z_pred: list[int], n_true: int, n_fit: int
) -> dict[int, int]:
    """Return {pred_label → true_label} optimal one-to-one alignment via DP.

    Uses exact DP over subsets, matching the logic in the original script but
    scoped to this file so the new script has no hidden dependency on the old one.
    """
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
    """Compute the four targeted recovery metrics."""
    z_pred = map_result["z"]
    pred_blocks_all: list[list[list[int]]] = [
        cluster["blocks"] for cluster in map_result["clusters"]
    ]
    n_fit = len(pred_blocks_all)
    n_pred_active = len(set(z_pred))

    # ── 1. Cluster ARI + NMI ─────────────────────────────────────────────────
    cluster_ari = _adjusted_rand_index(z_true, z_pred)
    # C=1 edge case: ARI is 0/0 when all labels are identical → treat as 1.0
    if n_true_clusters == 1 and n_pred_active == 1:
        cluster_ari = 1.0
    cluster_nmi = (
        1.0 if (n_true_clusters == 1 and n_pred_active == 1)
        else float(_sklearn_nmi(z_true, z_pred, average_method='arithmetic'))
    )

    # ── 2–4. Per-cluster metrics (order_distance, block_ari, theta_rmse) ──────
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
            # Unmatched true cluster: worst-case order distance, zero block ARI.
            w_kemeny    += w * (1.0 if max_pair_dist > 0 else 0.0)
            w_block_ari += w * 0.0
            # Theta: skip unmatched cluster (theta_wsum stays smaller → RMSE
            # is computed over matched clusters only and re-normalised).
        else:
            pred_cons = pred_blocks_all[pc]

            # Order distance: normalised Kemeny (0 = perfect, 1 = worst)
            w_kemeny += w * _normalized_kemeny(true_cons, pred_cons, n_items)

            # Block ARI / NMI: compare item → block-id memberships, ignoring order
            true_bix = _block_index(true_cons, n_items)
            pred_bix = _block_index(pred_cons, n_items)
            w_block_ari += w * _adjusted_rand_index(true_bix, pred_bix)
            w_block_nmi += w * (
                1.0 if len(set(true_bix)) == 1 and len(set(pred_bix)) == 1
                else float(_sklearn_nmi(true_bix, pred_bix, average_method='arithmetic'))
            )

            # Theta RMSE
            pred_theta = float(map_result["clusters"][pc]["theta"])
            if not math.isnan(pred_theta):
                theta_sq   += w * (true_theta - pred_theta) ** 2
                theta_wsum += w

    theta_rmse = (
        math.sqrt(theta_sq / theta_wsum) if theta_wsum > 0 else float("nan")
    )

    return {
        "cluster_ari":    cluster_ari,
        "cluster_nmi":    cluster_nmi,
        "order_distance": w_kemeny,    # 0 = perfect, 1 = worst
        "block_ari":      w_block_ari,
        "block_nmi":      w_block_nmi,
        "theta_rmse":     theta_rmse,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model fitting
# ─────────────────────────────────────────────────────────────────────────────

def _fitted_k(n_true: int) -> int:
    """Over-saturated fitted cluster count: 2 × C_true."""
    return 2 * n_true


def fit_and_evaluate(
    scenario: Scenario,
    *,
    n_iter: int,
    burn_in: int,
    thin: int,
    n_restarts: int,
    use_spectral: bool = False,
) -> dict[str, Any]:
    """Generate data, fit TiMMM (multiple restarts), return scenario result."""
    t0 = time.time()

    # Generate synthetic data from the Mallows mixture generative model
    true_blocks, tau_true, z_true, rankings = generate_mixture_data(
        n_assessors=scenario.n_assessors,
        n_items=scenario.n_items,
        C=scenario.n_clusters,
        seed=scenario.seed,
        theta=scenario.theta,
        block_density=scenario.block_density,
    )

    n_fit = _fitted_k(scenario.n_clusters)
    init_mu = [0.001]

    # Multiple independent restarts — keep the MAP with the highest log-posterior
    best: dict[str, Any] | None = None
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
        result = model.find_map(samples, refine=True, verbose=False)
        if best is None or result["logp_refined"] > best["logp_refined"]:
            best = result

    assert best is not None

    metrics = compute_metrics(
        true_blocks=true_blocks,
        z_true=z_true,
        map_result=best,
        n_true_clusters=scenario.n_clusters,
        n_items=scenario.n_items,
        true_theta=scenario.theta,
    )

    return {
        "scenario": asdict(scenario),
        "settings": {
            "n_iter": n_iter,
            "burn_in": burn_in,
            "thin": thin,
            "n_restarts": n_restarts,
            "fit_n_clusters": n_fit,
            "init_strategy": "spectral" if use_spectral else "random",
        },
        "metrics": metrics,
        "runtime_seconds": time.time() - t0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OAT scenario design
# ─────────────────────────────────────────────────────────────────────────────

def oat_scenarios() -> list[Scenario]:
    """OAT design: same parameter grid as the large_oat baseline run.

    One axis is swept at a time; all other axes are held at their central
    defaults.  Results are therefore directly comparable to the large_oat run
    (which uses the same grid and the same seeds) except that this script
    computes only the four targeted metrics and uses no spectral pre-processing.
    """
    C_VALUES  = (2, 5, 10, 15, 20)
    N_VALUES  = (20, 50, 100, 200, 500, 1000)
    N_ITEMS   = (5, 10, 20, 50, 100)
    BD_VALUES = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
    TH_VALUES = (0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    SEEDS     = (42, 123, 999, 7, 13, 17, 31, 53, 97, 137)   # 10 data seeds

    # OAT defaults (same as large_oat)
    C_DEF, N_DEF, NI_DEF, BD_DEF, TH_DEF = 5, 200, 20, 0.4, 1.0

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

    # Vary one axis at a time
    for c  in C_VALUES:  _add(c,     N_DEF, NI_DEF, BD_DEF, TH_DEF)
    for n  in N_VALUES:  _add(C_DEF, n,     NI_DEF, BD_DEF, TH_DEF)
    for ni in N_ITEMS:   _add(C_DEF, N_DEF, ni,     BD_DEF, TH_DEF)
    for bd in BD_VALUES: _add(C_DEF, N_DEF, NI_DEF, bd,     TH_DEF)
    for th in TH_VALUES: _add(C_DEF, N_DEF, NI_DEF, BD_DEF, th)

    scenarios: list[Scenario] = []
    for name, c, n, ni, bd, th in base:
        for seed in SEEDS:
            scenarios.append(Scenario(
                name=f"{name}_seed{seed}",
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

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TiMMM OAT recovery study — clean four-metric evaluation."
    )
    parser.add_argument("--n-iter",     type=int, default=10000)
    parser.add_argument("--burn-in",    type=int, default=5000)
    parser.add_argument("--thin",       type=int, default=20)
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap on scenarios (useful for quick smoke-tests).",
    )
    parser.add_argument(
        "--resume-dir", type=str, default=None,
        help="Resume an interrupted run from this directory.",
    )
    parser.add_argument(
        "--spectral", action="store_true",
        help="Use spectral-clustering warm-start instead of random initialisation.",
    )
    args = parser.parse_args()

    scenarios = oat_scenarios()
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    # ── Resume mode ────────────────────────────────────────────────────────────
    if args.resume_dir:
        out_dir = Path(args.resume_dir)
        if not out_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {out_dir}")
        done = {
            p.stem for p in out_dir.glob("*.json")
            if p.name not in {"run_metadata.json", "all_results.json"}
        }
        completed: list[dict[str, Any]] = [
            json.loads((out_dir / f"{s.name}.json").read_text(encoding="utf-8"))
            for s in scenarios if s.name in done
        ]
        scenarios = [s for s in scenarios if s.name not in done]
        print(f"Resuming: {out_dir}")
        print(f"  Already done: {len(completed)},  remaining: {len(scenarios)}")
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "spectral" if args.spectral else "oat"
        out_dir = Path("simulation_recovery_runs") / f"recovery_timmm_{suffix}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        completed = []
        metadata = {
            "mode": "timmm_spectral" if args.spectral else "timmm_oat",
            "init_strategy": "spectral" if args.spectral else "random",
            "n_iter": args.n_iter,
            "burn_in": args.burn_in,
            "thin": args.thin,
            "n_restarts": args.n_restarts,
            "total_scenarios": len(scenarios),
            "metrics": ["cluster_ari", "order_distance", "block_ari", "theta_rmse"],
        }
        (out_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"Output: {out_dir}")
        print(f"  {len(scenarios)} scenarios to run "
              f"(n_iter={args.n_iter}, burn_in={args.burn_in}, "
              f"thin={args.thin}, n_restarts={args.n_restarts})")

    # ── Run scenarios ──────────────────────────────────────────────────────────
    n_total = len(scenarios) + len(completed)
    results = list(completed)

    for i, scenario in enumerate(scenarios, start=len(completed) + 1):
        sc = scenario
        print(
            f"[{i:>3}/{n_total}]  C={sc.n_clusters:>2}  N={sc.n_assessors:>4}"
            f"  n={sc.n_items:>3}  bd={sc.block_density:.2f}"
            f"  θ={sc.theta:>5}  seed={sc.seed}",
            end="  …  ",
            flush=True,
        )
        try:
            result = fit_and_evaluate(
                scenario,
                n_iter=args.n_iter,
                burn_in=args.burn_in,
                thin=args.thin,
                n_restarts=args.n_restarts,
                use_spectral=args.spectral,
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        m = result["metrics"]
        theta_str = "nan" if math.isnan(m["theta_rmse"]) else f"{m['theta_rmse']:.3f}"
        print(
            f"ARI={m['cluster_ari']:.3f}"
            f"  ord={m['order_distance']:.3f}"
            f"  blkARI={m['block_ari']:.3f}"
            f"  θRMSE={theta_str}"
            f"  ({result['runtime_seconds']:.1f}s)"
        )

        (out_dir / f"{scenario.name}.json").write_text(
            json.dumps(_to_jsonable(result), indent=2), encoding="utf-8"
        )
        results.append(result)

    # ── Aggregate all results into a single file ───────────────────────────────
    (out_dir / "all_results.json").write_text(
        json.dumps(_to_jsonable(results), indent=2), encoding="utf-8"
    )

    # ── Final summary ──────────────────────────────────────────────────────────
    if not results:
        print("No results to summarise.")
        return

    metrics_all = [r["metrics"] for r in results]
    n = len(metrics_all)
    valid_rmse = [m["theta_rmse"] for m in metrics_all if not math.isnan(m["theta_rmse"])]

    print(f"\n{'─' * 60}")
    print(f"  Scenarios completed : {n}/{n_total}")
    print(f"  Mean cluster ARI    : {sum(m['cluster_ari']    for m in metrics_all) / n:.3f}")
    print(f"  Mean order distance : {sum(m['order_distance'] for m in metrics_all) / n:.3f}")
    print(f"  Mean block ARI      : {sum(m['block_ari']      for m in metrics_all) / n:.3f}")
    if valid_rmse:
        print(f"  Mean theta RMSE     : {sum(valid_rmse) / len(valid_rmse):.3f}")
    else:
        print( "  Mean theta RMSE     : n/a")
    print(f"{'─' * 60}")
    print(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
