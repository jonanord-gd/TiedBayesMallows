"""
TiMMM Items × Clusters Grid Study
====================================
Targeted study exploring how **clustering recovery** varies with the number
of items (n_items) for hard scenarios with many clusters (n_clusters ≥ 30).

Hypothesis
----------
On hard datasets (many clusters), recovery improves with more items
because larger item spaces provide richer pairwise information.

Design
------
  n_clusters  :  30, 40, 50          (three hard levels)
  n_items     :  20, 50, 100, 200    (increasing item space)
  n_assessors :  200                 (baseline)
  block_density: 0.4                 (baseline)
  theta        : 1.0                 (baseline)
  seeds        : 0 – 29             (30 seeds per base scenario)

  Total = 3 × 4 × 30 = 360 runs.

All other MCMC settings match the TiMMM raw OAT study defaults:
  n_iter=10000, burn_in=5000, thin=1, init=spectral.

Usage
-----
  python _timmm_items_cluster_grid.py                       # full 360-run study
  python _timmm_items_cluster_grid.py --limit 10            # smoke-test
  python _timmm_items_cluster_grid.py --resume-dir <dir>    # continue interrupted run
  python _timmm_items_cluster_grid.py --n-iter 5000 --burn-in 2500  # custom MCMC
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Re-use helpers from the raw OAT study (all defined at module level).
from _timmm_raw_study import (
    Scenario,
    fit_and_record,
)


# ─────────────────────────────────────────────────────────────────────────────
# Grid design constants
# ─────────────────────────────────────────────────────────────────────────────

SEEDS = tuple(range(30))          # 30 reproducible seeds

C_GRID  = (30, 40, 50)            # hard cluster counts
NI_GRID = (20, 50, 100, 200)      # item counts to sweep

# Fixed baseline values for all other axes
N_ASSESSORS_BASE  = 200
BLOCK_DENSITY_BASE = 0.4
THETA_BASE         = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Scenario construction
# ─────────────────────────────────────────────────────────────────────────────

def grid_scenarios() -> list[Scenario]:
    """Return all (n_clusters, n_items, seed) combinations."""
    def _name(c: int, ni: int) -> str:
        return f"c{c}_n{N_ASSESSORS_BASE}_ni{ni}_bd0p40_theta1p0"

    scenarios: list[Scenario] = []
    for c in C_GRID:
        for ni in NI_GRID:
            base_name = _name(c, ni)
            for seed in SEEDS:
                scenarios.append(Scenario(
                    name=f"{base_name}_seed{seed:02d}",
                    n_clusters=c,
                    n_assessors=N_ASSESSORS_BASE,
                    n_items=ni,
                    theta=THETA_BASE,
                    block_density=BLOCK_DENSITY_BASE,
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
        description=(
            "TiMMM items × clusters grid study.  "
            "Sweeps n_items ∈ {20,50,100,200} × n_clusters ∈ {30,40,50} "
            "with all other parameters at baseline.  360 runs total."
        )
    )
    parser.add_argument("--n-iter",    type=int, default=10_000, help="MCMC iterations.")
    parser.add_argument("--burn-in",   type=int, default=5_000,  help="Burn-in iterations.")
    parser.add_argument("--thin",      type=int, default=1,      help="Thinning interval.")
    parser.add_argument("--limit",     type=int, default=None,   help="Cap scenarios (smoke-test).")
    parser.add_argument("--out-dir",   type=str, default=None,   help="Override output directory.")
    parser.add_argument(
        "--resume-dir", type=str, default=None,
        help="Resume from an existing directory (skips already-completed scenarios).",
    )
    args = parser.parse_args()

    scenarios = grid_scenarios()
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

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
            out_dir = (
                Path("simulation_recovery_runs")
                / f"recovery_items_cluster_grid_{timestamp}"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        completed = []

        metadata = {
            "study":        "TiMMM items × clusters grid study",
            "description": (
                "Sweeps n_items and n_clusters for hard scenarios "
                "to test whether more items improve clustering recovery "
                "when the cluster count is large (≥ 30)."
            ),
            "grid_design": {
                "n_clusters":    list(C_GRID),
                "n_items":       list(NI_GRID),
                "n_assessors":   N_ASSESSORS_BASE,
                "block_density": BLOCK_DENSITY_BASE,
                "theta":         THETA_BASE,
            },
            "seeds":            list(SEEDS),
            "n_seeds":          len(SEEDS),
            "n_base_scenarios": len(C_GRID) * len(NI_GRID),
            "total_scenarios":  len(C_GRID) * len(NI_GRID) * len(SEEDS),
            "mcmc": {
                "n_iter":        args.n_iter,
                "burn_in":       args.burn_in,
                "thin":          args.thin,
                "n_restarts":    1,
                "init_strategy": "spectral",
            },
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        (out_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"Output  : {out_dir}")
        print(
            f"  {len(scenarios)} scenarios  "
            f"(n_iter={args.n_iter}, burn_in={args.burn_in}, "
            f"thin={args.thin}, init=spectral)"
        )

    # ── Run scenarios ─────────────────────────────────────────────────────────
    n_total = len(scenarios) + len(completed)
    results = list(completed)

    for i, scenario in enumerate(scenarios, start=len(completed) + 1):
        sc = scenario
        print(
            f"[{i:>4}/{n_total}]  C={sc.n_clusters:>2}  N={sc.n_assessors:>4}"
            f"  n={sc.n_items:>3}  bd={sc.block_density:.2f}"
            f"  θ={sc.theta:>5}  seed={sc.seed:>2}",
            end="  …  ",
            flush=True,
        )
        try:
            record = fit_and_record(
                scenario,
                n_iter=args.n_iter,
                burn_in=args.burn_in,
                thin=args.thin,
                use_spectral=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}")
            continue

        results.append(record)

        # Write individual scenario file immediately (crash-safe)
        scenario_path = out_dir / f"{scenario.name}.json"
        scenario_path.write_text(
            json.dumps(_to_jsonable(record), indent=2), encoding="utf-8"
        )

        m = record["metrics"]
        print(
            f"NMI={m.get('cluster_nmi', float('nan')):.3f}  "
            f"OD={m.get('order_distance', float('nan')):.3f}  "
            f"bNMI={m.get('block_nmi', float('nan')):.3f}  "
            f"θRMSE={m.get('theta_rmse', float('nan')):.3f}  "
            f"t={record['runtime_seconds']:.1f}s"
        )

    # ── Write consolidated results ─────────────────────────────────────────────
    all_path = out_dir / "all_results.json"
    all_path.write_text(
        json.dumps(_to_jsonable(results), indent=2), encoding="utf-8"
    )

    total_time = sum(r.get("runtime_seconds", 0) for r in results)
    print(f"\nDone.  {len(results)} results written to {out_dir}")
    print(f"Total runtime : {total_time / 60:.1f} min")


if __name__ == "__main__":
    main()
