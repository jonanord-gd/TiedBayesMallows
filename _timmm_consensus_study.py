"""
TiMMM Consensus Recovery Study
================================
Extension of _timmm_oat_study.py that adds a consensus-clustering post-processing
step identical to the approach in F1GridVisualise.ipynb:

  1. Run n_restarts independent MCMC chains on each synthetic dataset.
  2. Collect the MAP z-assignment (z_map) from every chain.
  3. Build an N×N co-occurrence matrix:  co[i,j] = fraction of chains where
     assessors i and j ended up in the same cluster.
  4. Spectral-cluster the co-occurrence (used as an affinity matrix), sweeping
     k and selecting the best number of groups by silhouette score on (1 − co).
  5. For each consensus group fit a single-cluster TiMMM and read off its MAP
     block structure and θ.
  6. Evaluate all four evaluation lines per OAT scenario axis:
       rand_mean       — mean best-MAP metric over seeds  (random init)
       rand_consensus  — mean consensus metric over seeds (random init chains)
       spec_mean       — mean best-MAP metric over seeds  (spectral init)
       spec_consensus  — mean consensus metric over seeds (spectral init chains)

OAT design and metrics are identical to _timmm_oat_study.py, but uses 3 data
seeds and 10 restarts per scenario (vs 3 seeds / 3 restarts in the OAT study).
The 10 restarts are the basis for the co-occurrence matrix used in consensus
clustering — more restarts yield a richer affinity signal.

Usage
-----
  python _timmm_consensus_study.py
  python _timmm_consensus_study.py --limit 5 --n-iter 2000 --burn-in 1000 --thin 10 --n-restarts 10
  python _timmm_consensus_study.py --resume-dir <dir>
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
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score as _sklearn_nmi
from sklearn.metrics import silhouette_score

from helper_functions.Generate_mixture_data import generate_mixture_data
from model.TiedMallowsModel import MixtureRankingModel
from model.initialization import init_spectral_with_z


# ─────────────────────────────────────────────────────────────────────────────
# OAT design constants  (same grid as _timmm_oat_study.py, but 10 data seeds)
# ─────────────────────────────────────────────────────────────────────────────

SEEDS     = (42, 123, 999)   # 3 data seeds — restarts carry the variance for consensus
C_VALUES  = (2, 5, 10, 15, 20)   # C=1 excluded — consensus clustering is trivial/degenerate for a single cluster
N_VALUES  = (20, 50, 100, 200, 500, 1000)
N_ITEMS   = (5, 10, 20, 50, 100)
BD_VALUES = (0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
TH_VALUES = (0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)

C_DEF, N_DEF, NI_DEF, BD_DEF, TH_DEF = 5, 200, 20, 0.4, 1.0


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
# Metric helpers  (mirrors _timmm_oat_study.py exactly)
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
    # denom == 0 when clustering is trivially degenerate (all-in-one or all-
    # singletons); in that case numerator is also 0 and agreement is perfect.
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
    cluster_nmi = (
        1.0 if (n_true_clusters == 1 and n_pred_active == 1)
        else float(_sklearn_nmi(z_true, z_pred, average_method='arithmetic'))
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
            w_kemeny    += w * (1.0 if max_pair_dist > 0 else 0.0)
            w_block_ari += 0.0
        else:
            pred_cons = pred_blocks_all[pc]
            w_kemeny    += w * _normalized_kemeny(true_cons, pred_cons, n_items)
            true_bix = _block_index(true_cons, n_items)
            pred_bix = _block_index(pred_cons, n_items)
            w_block_ari += w * _adjusted_rand_index(true_bix, pred_bix)
            w_block_nmi += w * (
                1.0 if len(set(true_bix)) == 1 and len(set(pred_bix)) == 1
                else float(_sklearn_nmi(true_bix, pred_bix, average_method='arithmetic'))
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
# Co-occurrence and consensus pipeline  (mirrors F1GridVisualise.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def build_co_occurrence(z_maps: list[list[int]], n_assessors: int) -> np.ndarray:
    """Return N×N co-occurrence matrix (fraction of chains where i,j same cluster)."""
    co = np.zeros((n_assessors, n_assessors), dtype=np.float64)
    for z in z_maps:
        za = np.asarray(z, dtype=int)
        co += (za[:, None] == za[None, :]).astype(np.float64)
    return co / max(len(z_maps), 1)


def select_consensus_k(
    co: np.ndarray,
    n_true: int,
    n_assessors: int,
    *,
    random_state: int = 0,
) -> tuple[int, np.ndarray]:
    """
    Spectral clustering on the co-occurrence affinity, picking k by silhouette.

    Sweeps k from 2 to min(N-1, 3*C_true+1).  If no k≥2 yields a positive
    silhouette score the function falls back to k=1 (all assessors in one group).
    """
    # Regularise co-occurrence to ensure a fully connected affinity graph.
    # Without this, zero off-diagonal entries disconnect the graph and cause
    # sklearn's SpectralClustering to emit a warning and produce unreliable
    # embeddings (common for large C where some pairs never share a cluster).
    epsilon = 1e-6
    co_reg = co.copy()
    off_diag = ~np.eye(n_assessors, dtype=bool)
    co_reg[off_diag] = np.clip(co_reg[off_diag], epsilon, 1.0)
    np.fill_diagonal(co_reg, 1.0)

    dist = np.clip(1.0 - co_reg, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    k_max = min(n_assessors - 1, max(4, 3 * n_true + 1))

    best_k      = 1
    best_sil    = -2.0
    best_labels = np.zeros(n_assessors, dtype=int)

    for k in range(2, k_max + 1):
        if k >= n_assessors:
            break
        try:
            sc = SpectralClustering(
                n_clusters=k,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=random_state % (2 ** 31),
                n_init=10,
            )
            labels = sc.fit_predict(co_reg)
            if len(np.unique(labels)) < 2:
                continue
            sil = float(silhouette_score(dist, labels, metric="precomputed"))
        except Exception:
            continue
        if sil > best_sil:
            best_sil    = sil
            best_k      = k
            best_labels = labels

    # Fallback: if no k ≥ 2 yielded a convincing silhouette, treat as one group.
    # Threshold > 0 guards against noise (e.g. C=1 data fitted with 2 clusters
    # produces co ≈ 0.5 everywhere → silhouette barely above 0 → spurious split).
    min_sil_threshold = 0.05
    if best_sil < min_sil_threshold:
        best_k      = 1
        best_labels = np.zeros(n_assessors, dtype=int)

    return best_k, best_labels


def fit_consensus_clusters(
    rankings: list[list[int]],
    n_items: int,
    z_consensus: np.ndarray,
    k_consensus: int,
    *,
    n_iter: int,
    burn_in: int,
    thin: int,
    seed: int,
) -> dict[str, Any]:
    """
    For each consensus group fit a single-cluster TiMMM and collect the MAP
    block structure and θ.  Returns a pseudo-map-result dict compatible with
    compute_metrics().
    """
    clusters_out: list[dict[str, Any]] = []

    for ki in range(k_consensus):
        idxs = np.where(z_consensus == ki)[0]
        if len(idxs) == 0:
            # Dummy for empty group — all items in separate blocks.
            clusters_out.append({
                "blocks": [[i] for i in range(n_items)],
                "theta":  0.01,
            })
            continue

        subset = [rankings[i] for i in idxs]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MixtureRankingModel(
                subset,
                n_clusters=1,
                seed=seed + ki * 997,
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
        clusters_out.append({
            "blocks": result["clusters"][0]["blocks"],
            "theta":  result["clusters"][0]["theta"],
        })

    return {
        "z":            z_consensus.tolist(),
        "clusters":     clusters_out,
        "logp_refined": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main scenario runner
# ─────────────────────────────────────────────────────────────────────────────

def fit_scenario_full(
    scenario: Scenario,
    *,
    n_iter: int,
    burn_in: int,
    thin: int,
    n_restarts: int,
) -> dict[str, Any]:
    """
    Run both random and spectral init for one (data_seed, base_scenario) pair.

    For each init strategy:
      - n_restarts independent chains  →  z_maps  +  best-MAP individual metrics
      - co-occurrence from n_restarts z_maps  →  spectral consensus  →  single-cluster TiMMMs
      →  consensus metrics

    Returns a dict with rand / spec sub-dicts, each containing
    mean_metrics, consensus_metrics, and k_consensus.
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

    n_fit     = 2 * scenario.n_clusters
    init_mu   = [0.001]

    # Shorter chains for the single-cluster consensus TiMMMs (warm start).
    cons_n_iter  = max(n_iter  // 2, 1000)
    cons_burn_in = max(burn_in // 2,  500)

    out: dict[str, Any] = {}

    for use_spectral in (False, True):
        init_key = "spec" if use_spectral else "rand"

        z_maps: list[list[int]]          = []
        individual_metrics: list[dict]   = []
        best_overall: dict | None        = None

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
            result = model.find_map(samples, refine=True, verbose=False)

            z_maps.append(result["z"])
            m = compute_metrics(
                true_blocks=true_blocks,
                z_true=z_true,
                map_result=result,
                n_true_clusters=scenario.n_clusters,
                n_items=scenario.n_items,
                true_theta=scenario.theta,
            )
            individual_metrics.append(m)

            if best_overall is None or result["logp_refined"] > best_overall["logp_refined"]:
                best_overall = result

        # ── Mean metrics (averaged over restarts) ─────────────────────────────
        metric_keys = list(individual_metrics[0].keys())
        mean_metrics: dict[str, float] = {}
        for k in metric_keys:
            vals = [m[k] for m in individual_metrics if not math.isnan(m[k])]
            mean_metrics[k] = float(np.mean(vals)) if vals else float("nan")

        # ── Consensus pipeline ────────────────────────────────────────────────
        co = build_co_occurrence(z_maps, scenario.n_assessors)
        k_cons, z_cons = select_consensus_k(
            co,
            scenario.n_clusters,
            scenario.n_assessors,
            random_state=scenario.seed,
        )
        consensus_map = fit_consensus_clusters(
            rankings,
            scenario.n_items,
            z_cons,
            k_cons,
            n_iter=cons_n_iter,
            burn_in=cons_burn_in,
            thin=thin,
            seed=scenario.seed,
        )
        consensus_metrics = compute_metrics(
            true_blocks=true_blocks,
            z_true=z_true,
            map_result=consensus_map,
            n_true_clusters=scenario.n_clusters,
            n_items=scenario.n_items,
            true_theta=scenario.theta,
        )

        out[init_key] = {
            "mean_metrics":      mean_metrics,
            "consensus_metrics": consensus_metrics,
            "k_consensus":       int(k_cons),
        }

    return {
        "scenario": asdict(scenario),
        "settings": {
            "n_iter":        n_iter,
            "burn_in":       burn_in,
            "thin":          thin,
            "n_restarts":    n_restarts,
            "fit_n_clusters": n_fit,
        },
        "rand":             out["rand"],
        "spec":             out["spec"],
        "runtime_seconds":  time.time() - t0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OAT scenario design  (same grid, 10 seeds)
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
        description="TiMMM OAT with consensus-clustering comparison (4 evaluation lines)."
    )
    parser.add_argument("--n-iter",     type=int, default=10000,
                        help="MCMC iterations per chain.")
    parser.add_argument("--burn-in",    type=int, default=5000,
                        help="Burn-in iterations per chain.")
    parser.add_argument("--thin",       type=int, default=20,
                        help="Thinning interval.")
    parser.add_argument(
        "--n-restarts", type=int, default=10,
        help=(
            "Chains per (scenario, init_strategy).  These chains supply both "
            "the z_maps for the co-occurrence matrix AND the individual MAP "
            "metrics that are averaged for the *_mean lines.  "
            "More restarts → richer co-occurrence but longer runtime."
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of scenarios (useful for smoke-tests).",
    )
    parser.add_argument(
        "--resume-dir", type=str, default=None,
        help="Resume an interrupted run from this directory.",
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
        timestamp  = time.strftime("%Y%m%d_%H%M%S")
        out_dir    = Path("simulation_recovery_runs") / f"recovery_timmm_consensus_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        completed  = []
        metadata   = {
            "mode":            "timmm_consensus",
            "n_iter":          args.n_iter,
            "burn_in":         args.burn_in,
            "thin":            args.thin,
            "n_restarts":      args.n_restarts,
            "n_seeds":         len(SEEDS),
            "total_scenarios": len(scenarios),
            "metrics":         ["cluster_ari", "order_distance", "block_ari", "theta_rmse"],
            "lines":           ["rand_mean", "rand_consensus", "spec_mean", "spec_consensus"],
        }
        (out_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"Output: {out_dir}")
        print(
            f"  {len(scenarios)} scenarios  "
            f"(n_iter={args.n_iter}, burn_in={args.burn_in}, "
            f"thin={args.thin}, n_restarts={args.n_restarts})"
        )

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
            result = fit_scenario_full(
                scenario,
                n_iter=args.n_iter,
                burn_in=args.burn_in,
                thin=args.thin,
                n_restarts=args.n_restarts,
            )
        except Exception as exc:
            print(f"ERROR: {exc}")
            continue

        rm = result["rand"]["mean_metrics"]
        rc = result["rand"]["consensus_metrics"]
        sm = result["spec"]["mean_metrics"]
        sc_ = result["spec"]["consensus_metrics"]
        print(
            f"rand  ARI={rm['cluster_ari']:.3f}/{rc['cluster_ari']:.3f}"
            f"  k_rand={result['rand']['k_consensus']}"
            f"  spec  ARI={sm['cluster_ari']:.3f}/{sc_['cluster_ari']:.3f}"
            f"  k_spec={result['spec']['k_consensus']}"
            f"  ({result['runtime_seconds']:.1f}s)"
        )

        (out_dir / f"{scenario.name}.json").write_text(
            json.dumps(_to_jsonable(result), indent=2), encoding="utf-8"
        )
        results.append(result)

    # ── Aggregate results ──────────────────────────────────────────────────────
    (out_dir / "all_results.json").write_text(
        json.dumps(_to_jsonable(results), indent=2), encoding="utf-8"
    )

    if not results:
        print("No results to summarise.")
        return

    n = len(results)
    print(f"\n{'─' * 70}")
    print(f"  Scenarios completed: {n}/{n_total}")
    for init_key, label in [("rand", "Random"), ("spec", "Spectral")]:
        for line_key, line_label in [
            ("mean_metrics", "mean"), ("consensus_metrics", "consensus")
        ]:
            vals = [
                r[init_key][line_key]["cluster_ari"]
                for r in results
                if not math.isnan(r[init_key][line_key]["cluster_ari"])
            ]
            if vals:
                print(
                    f"  Mean cluster ARI ({label} {line_label:10s}): "
                    f"{float(np.mean(vals)):.3f}"
                )
    print(f"{'─' * 70}")
    print(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
