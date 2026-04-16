"""Simulation study for recovery under the tied Mallows mixture model.

This script generates synthetic datasets from the helper Mallows-with-ties
generator, fits the current tied Mallows mixture model with:

  - no Pitman-Yor prior
  - no block-order prior

and reports:

  - cluster recovery (aligned accuracy, ARI)
  - consensus recovery for the matched cluster pair
  - Kemeny-type distance between true and recovered weak consensuses
  - strict inversion count between the induced block orders
  - pairwise same-block precision / recall / F1
  - similarity in number of blocks

Default behaviour runs a small set of early-test scenarios so the setup can be
verified before launching a larger sweep.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from helper_functions.Generate_mixture_data import generate_mixture_data
from model.TiedMallowsModel import MixtureRankingModel


@dataclass(frozen=True)
class Scenario:
    name: str
    n_clusters: int
    n_assessors: int
    n_items: int
    theta: float
    block_density: float
    seed: int


def build_scenarios(base_specs: list[tuple[str, int, int, int, float, float]], seeds: list[int]) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for base_name, n_clusters, n_assessors, n_items, block_density, theta in base_specs:
        for seed in seeds:
            scenarios.append(
                Scenario(
                    name=f"{base_name}_seed{seed}",
                    n_clusters=n_clusters,
                    n_assessors=n_assessors,
                    n_items=n_items,
                    theta=theta,
                    block_density=block_density,
                    seed=seed,
                )
            )
    return scenarios


def early_test_scenarios() -> list[Scenario]:
    """Targeted early tests: sweep block_density at fixed moderate difficulty."""
    return build_scenarios(
        [
            # Vary block_density at fixed (C=3, N=300, m=15, theta=1.5)
            ("c3_n300_m15_bd0p20_theta1p5", 3, 300, 15, 0.20, 1.5),
            ("c3_n300_m15_bd0p40_theta1p5", 3, 300, 15, 0.40, 1.5),
            ("c3_n300_m15_bd0p60_theta1p5", 3, 300, 15, 0.60, 1.5),
            ("c3_n300_m15_bd0p80_theta1p5", 3, 300, 15, 0.80, 1.5),
            # Sanity check: easy and hard extremes
            ("c2_n400_m12_bd0p40_theta3p0", 2, 400, 12, 0.40, 3.0),  # should be easy
            ("c4_n200_m24_bd0p40_theta0p5", 4, 200, 24, 0.40, 0.5),  # should be hard
        ],
        seeds=[42],
    )


def cluster_values(include_c10: bool = False) -> tuple[int, ...]:
    values = [2, 3, 5]
    if include_c10:
        values.append(10)
    return tuple(values)


def larger_grid_scenarios(include_c10: bool = False) -> list[Scenario]:
    """Exploration grid: 1 seed per combo, wide parameter variation.

    Design philosophy: find where the model breaks by varying each axis broadly.
    - theta: spans near-uniform (0.3) to strong signal (4.0)
    - block_density: spans nearly fully tied (0.15) to nearly strict (0.85)
    - n_clusters: 2–5, optionally also 10
    - n_assessors: low to high (small total N stresses cluster recovery)
    - n_items: small to large (large m stresses consensus recovery)
    """
    base_specs: list[tuple[str, int, int, int, float, float]] = []
    for n_clusters in cluster_values(include_c10):
        for n_assessors in (100, 300, 700):
            for n_items in (10, 16, 24):
                for block_density in (0.15, 0.40, 0.70):
                    bd_str = f"{block_density:.2f}".replace('.', 'p')
                    for theta in (0.4, 1.0, 2.0, 4.0):
                        base_specs.append((
                            f"c{n_clusters}_n{n_assessors}_m{n_items}_bd{bd_str}_theta{str(theta).replace('.', 'p')}",
                            n_clusters, n_assessors, n_items, block_density, theta,
                        ))
    return build_scenarios(base_specs, seeds=[101])


def contrast_grid_scenarios(include_c10: bool = False) -> list[Scenario]:
    """Focused contrast grid with larger item counts and more assessors.

    This grid emphasizes clear low/high regimes for signal and tie density while
    extending the item-count range up to 100.
    """
    base_specs: list[tuple[str, int, int, int, float, float]] = []
    for n_clusters in cluster_values(include_c10):
        for n_assessors in (200, 500, 1000):
            for n_items in (20, 50, 100):
                for block_density in (0.15, 0.70):
                    bd_str = f"{block_density:.2f}".replace('.', 'p')
                    for theta in (0.3, 3.0):
                        theta_str = str(theta).replace('.', 'p')
                        base_specs.append((
                            f"c{n_clusters}_n{n_assessors}_m{n_items}_bd{bd_str}_theta{theta_str}",
                            n_clusters, n_assessors, n_items, block_density, theta,
                        ))
    return build_scenarios(base_specs, seeds=[101])


def block_index(blocks: list[list[int]], n_items: int) -> list[int]:
    out = [-1] * n_items
    for block_id, block in enumerate(blocks):
        for item in block:
            out[item] = block_id
    return out


def weak_relation(block_of: list[int], i: int, j: int) -> int:
    bi = block_of[i]
    bj = block_of[j]
    if bi == bj:
        return 0
    return -1 if bi < bj else 1


def kemeny_p_half_weak_vs_weak(true_blocks: list[list[int]], pred_blocks: list[list[int]], n_items: int) -> float:
    true_idx = block_index(true_blocks, n_items)
    pred_idx = block_index(pred_blocks, n_items)
    distance = 0.0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            rel_true = weak_relation(true_idx, i, j)
            rel_pred = weak_relation(pred_idx, i, j)
            if rel_true == rel_pred:
                continue
            if rel_true == 0 or rel_pred == 0:
                distance += 0.5
            else:
                distance += 1.0
    return distance


def normalized_kemeny_p_half_weak_vs_weak(true_blocks: list[list[int]], pred_blocks: list[list[int]], n_items: int) -> float:
    max_distance = n_items * (n_items - 1) / 2
    if max_distance <= 0:
        return 0.0
    return kemeny_p_half_weak_vs_weak(true_blocks, pred_blocks, n_items) / max_distance


def strict_inversion_count(true_blocks: list[list[int]], pred_blocks: list[list[int]], n_items: int) -> int:
    true_idx = block_index(true_blocks, n_items)
    pred_idx = block_index(pred_blocks, n_items)
    inversions = 0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            rel_true = weak_relation(true_idx, i, j)
            rel_pred = weak_relation(pred_idx, i, j)
            if rel_true != 0 and rel_pred != 0 and rel_true != rel_pred:
                inversions += 1
    return inversions


def same_block_pair_stats(true_blocks: list[list[int]], pred_blocks: list[list[int]], n_items: int) -> dict[str, float]:
    true_idx = block_index(true_blocks, n_items)
    pred_idx = block_index(pred_blocks, n_items)
    tp = fp = fn = 0
    for i in range(n_items):
        for j in range(i + 1, n_items):
            true_same = true_idx[i] == true_idx[j]
            pred_same = pred_idx[i] == pred_idx[j]
            if true_same and pred_same:
                tp += 1
            elif pred_same and not true_same:
                fp += 1
            elif true_same and not pred_same:
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {
        "same_block_precision": precision,
        "same_block_recall": recall,
        "same_block_f1": f1,
        "same_block_jaccard": jaccard,
    }


def aligned_cluster_mapping(
    z_true: list[int],
    z_pred: list[int],
    n_true_clusters: int,
    n_pred_clusters: int,
) -> tuple[dict[int, int], list[list[int]], list[int], float]:
    """Find the best one-to-one alignment from predicted to true clusters.

    The previous implementation enumerated all permutations of predicted labels,
    which becomes infeasible for overfitted models like C=10 with 20 fitted
    clusters. This version uses exact dynamic programming over subsets of the
    active predicted labels, preserving the same optimal objective while keeping
    the computation tractable.
    """
    confusion = [[0 for _ in range(n_pred_clusters)] for _ in range(n_true_clusters)]
    for true_label, pred_label in zip(z_true, z_pred):
        if 0 <= pred_label < n_pred_clusters:
            confusion[true_label][pred_label] += 1

    active_pred_labels = sorted(set(z_pred))
    dummy_labels = [-(i + 1) for i in range(max(0, n_true_clusters - len(active_pred_labels)))]
    candidate_labels = active_pred_labels + dummy_labels

    if not candidate_labels:
        z_aligned = [-1 for _ in z_pred]
        return {}, confusion, z_aligned, 0.0

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def best_score(true_idx: int, used_mask: int) -> int:
        if true_idx == n_true_clusters:
            return 0

        best = -1
        for pos, label in enumerate(candidate_labels):
            if used_mask & (1 << pos):
                continue
            gain = confusion[true_idx][label] if label >= 0 else 0
            total = gain + best_score(true_idx + 1, used_mask | (1 << pos))
            if total > best:
                best = total
        return best

    pred_to_true: dict[int, int] = {}
    used_mask = 0
    for true_idx in range(n_true_clusters):
        chosen_pos = -1
        chosen_label = -1
        chosen_total = -1
        for pos, label in enumerate(candidate_labels):
            if used_mask & (1 << pos):
                continue
            gain = confusion[true_idx][label] if label >= 0 else 0
            total = gain + best_score(true_idx + 1, used_mask | (1 << pos))
            if total > chosen_total:
                chosen_total = total
                chosen_pos = pos
                chosen_label = label

        used_mask |= 1 << chosen_pos
        if chosen_label >= 0:
            pred_to_true[chosen_label] = true_idx

    z_aligned = [pred_to_true.get(label, -1) for label in z_pred]
    accuracy = sum(int(a == b) for a, b in zip(z_true, z_aligned)) / len(z_true)
    return pred_to_true, confusion, z_aligned, accuracy


def adjusted_rand_index(z_true: list[int], z_pred: list[int]) -> float:
    def comb2(x: int) -> int:
        return x * (x - 1) // 2

    true_labels = sorted(set(z_true))
    pred_labels = sorted(set(z_pred))
    true_to_idx = {label: idx for idx, label in enumerate(true_labels)}
    pred_to_idx = {label: idx for idx, label in enumerate(pred_labels)}
    contingency = [[0 for _ in range(len(pred_labels))] for _ in range(len(true_labels))]
    for true_label, pred_label in zip(z_true, z_pred):
        contingency[true_to_idx[true_label]][pred_to_idx[pred_label]] += 1

    row_sums = [sum(row) for row in contingency]
    col_sums = [sum(contingency[r][c] for r in range(len(true_labels))) for c in range(len(pred_labels))]
    n = len(z_true)

    sum_comb = sum(comb2(cell) for row in contingency for cell in row)
    sum_row = sum(comb2(v) for v in row_sums)
    sum_col = sum(comb2(v) for v in col_sums)
    total = comb2(n)
    if total == 0:
        return 1.0
    expected = (sum_row * sum_col) / total
    max_index = 0.5 * (sum_row + sum_col)
    denom = max_index - expected
    if denom == 0:
        return 0.0
    return (sum_comb - expected) / denom


def fitted_cluster_count(n_true_clusters: int) -> int:
    """Over-saturate the fitted model with more clusters than expected."""
    return 2 * n_true_clusters


def init_mu_small(n_clusters_fit: int) -> list[float]:
    """Small Dirichlet prior, matching the overfitted notebook pattern."""
    return [1.0 / (2.0 * n_clusters_fit)]


def fit_model(
    rankings: list[list[int]],
    n_clusters_fit: int,
    n_iter: int,
    burn_in: int,
    thin: int,
    seed: int,
    n_restarts: int,
    use_annealing: bool = True,
    annealing_schedule_type: str = "linear",
    annealing_plateau_frac: float = 0.5,
    temp_min: float = 0.1,
    temp_max: float = 1.0,
) -> dict[str, Any]:
    init_mu = init_mu_small(n_clusters_fit)
    best_result: dict[str, Any] | None = None
    restart_summaries: list[dict[str, Any]] = []
    for restart in range(n_restarts):
        restart_seed = seed + 1000 * restart
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MixtureRankingModel(
                rankings,
                n_clusters=n_clusters_fit,
                init_mu=init_mu,
                seed=restart_seed,
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
            use_annealing=use_annealing,
            annealing_schedule_type=annealing_schedule_type,
            annealing_plateau_frac=annealing_plateau_frac,
            temp_min=temp_min,
            temp_max=temp_max,
        )
        assert samples is not None
        map_result = model.find_map(samples, refine=True, verbose=False)
        restart_info = {
            "restart": restart,
            "seed": restart_seed,
            "n_clusters_fit": n_clusters_fit,
            "init_mu_value": init_mu[0],
            "logp_chain": map_result["logp_chain"],
            "logp_refined": map_result["logp_refined"],
        }
        restart_summaries.append(restart_info)
        if best_result is None or map_result["logp_refined"] > best_result["logp_refined"]:
            best_result = map_result

    assert best_result is not None
    best_result = dict(best_result)
    best_result["restart_summaries"] = restart_summaries
    return best_result


def evaluate_recovery(
    true_blocks: list[list[list[int]]],
    z_true: list[int],
    map_result: dict[str, Any],
    n_true_clusters: int,
    n_items: int,
) -> dict[str, Any]:
    z_pred = map_result["z"]
    pred_blocks = [cluster["blocks"] for cluster in map_result["clusters"]]
    n_fit_clusters = len(pred_blocks)   # full range of label indices (for confusion matrix)
    n_pred_clusters = len(set(z_pred))  # actual non-empty clusters at convergence

    pred_to_true, confusion, z_aligned, aligned_acc = aligned_cluster_mapping(
        z_true, z_pred, n_true_clusters, n_fit_clusters
    )
    ari = adjusted_rand_index(z_true, z_pred)

    per_cluster = []
    cluster_sizes = [sum(1 for z in z_true if z == c) for c in range(n_true_clusters)]
    weighted_kemeny = 0.0
    weighted_normalized_kemeny = 0.0
    weighted_inversions = 0.0
    weighted_block_f1 = 0.0
    weighted_block_error = 0.0
    total_weight = sum(cluster_sizes)

    true_to_pred = {true_label: pred_label for pred_label, true_label in pred_to_true.items()}
    max_pair_distance = n_items * (n_items - 1) / 2

    for true_cluster in range(n_true_clusters):
        pred_cluster = true_to_pred.get(true_cluster)
        true_consensus = true_blocks[true_cluster]

        if pred_cluster is None:
            pred_consensus = []
            kemeny = max_pair_distance
            normalized_kemeny = 1.0 if max_pair_distance > 0 else 0.0
            inversions = int(max_pair_distance)
            block_stats = {
                "same_block_precision": 0.0,
                "same_block_recall": 0.0,
                "same_block_f1": 0.0,
                "same_block_jaccard": 0.0,
            }
            block_error = float(len(true_consensus))
            overlap = 0.0
        else:
            pred_consensus = pred_blocks[pred_cluster]
            kemeny = kemeny_p_half_weak_vs_weak(true_consensus, pred_consensus, n_items)
            normalized_kemeny = normalized_kemeny_p_half_weak_vs_weak(true_consensus, pred_consensus, n_items)
            inversions = strict_inversion_count(true_consensus, pred_consensus, n_items)
            block_stats = same_block_pair_stats(true_consensus, pred_consensus, n_items)
            block_error = abs(len(true_consensus) - len(pred_consensus))
            overlap = confusion[true_cluster][pred_cluster] / cluster_sizes[true_cluster] if cluster_sizes[true_cluster] else 0.0

        weight = cluster_sizes[true_cluster] / total_weight if total_weight else 0.0
        weighted_kemeny += weight * kemeny
        weighted_normalized_kemeny += weight * normalized_kemeny
        weighted_inversions += weight * inversions
        weighted_block_f1 += weight * block_stats["same_block_f1"]
        weighted_block_error += weight * block_error

        per_cluster.append({
            "true_cluster": true_cluster,
            "matched_pred_cluster": pred_cluster,
            "true_cluster_size": cluster_sizes[true_cluster],
            "assignment_overlap": overlap,
            "true_n_blocks": len(true_consensus),
            "pred_n_blocks": len(pred_consensus),
            "block_count_abs_error": block_error,
            "kemeny_p_half": kemeny,
            "normalized_kemeny_p_half": normalized_kemeny,
            "strict_inversions": inversions,
            **block_stats,
            "true_blocks": true_consensus,
            "pred_blocks": pred_consensus,
        })

    return {
        "aligned_cluster_accuracy": aligned_acc,
        "adjusted_rand_index": ari,
        "weighted_kemeny_p_half": weighted_kemeny,
        "weighted_normalized_kemeny_p_half": weighted_normalized_kemeny,
        "weighted_strict_inversions": weighted_inversions,
        "weighted_same_block_f1": weighted_block_f1,
        "weighted_block_count_error": weighted_block_error,
        "n_true_clusters": n_true_clusters,
        "n_pred_clusters": n_pred_clusters,
        "cluster_mapping_pred_to_true": pred_to_true,
        "confusion": confusion,
        "per_cluster": per_cluster,
    }


def run_scenario(
    scenario: Scenario,
    *,
    n_iter: int,
    burn_in: int,
    thin: int,
    n_restarts: int,
    use_annealing: bool = True,
    annealing_schedule_type: str = "linear",
    annealing_plateau_frac: float = 0.5,
    temp_min: float = 0.1,
    temp_max: float = 1.0,
) -> dict[str, Any]:
    start = time.time()
    n_clusters_fit = fitted_cluster_count(scenario.n_clusters)
    init_mu = init_mu_small(n_clusters_fit)
    true_blocks, tau_true, z_true, rankings = generate_mixture_data(
        n_assessors=scenario.n_assessors,
        n_items=scenario.n_items,
        C=scenario.n_clusters,
        seed=scenario.seed,
        theta=scenario.theta,
        block_density=scenario.block_density,
    )
    map_result = fit_model(
        rankings=rankings,
        n_clusters_fit=n_clusters_fit,
        n_iter=n_iter,
        burn_in=burn_in,
        thin=thin,
        seed=scenario.seed,
        n_restarts=n_restarts,
        use_annealing=use_annealing,
        annealing_schedule_type=annealing_schedule_type,
        annealing_plateau_frac=annealing_plateau_frac,
        temp_min=temp_min,
        temp_max=temp_max,
    )
    metrics = evaluate_recovery(
        true_blocks=true_blocks,
        z_true=z_true,
        map_result=map_result,
        n_true_clusters=scenario.n_clusters,
        n_items=scenario.n_items,
    )
    elapsed = time.time() - start
    return {
        "scenario": asdict(scenario),
        "settings": {
            "n_iter": n_iter,
            "burn_in": burn_in,
            "thin": thin,
            "n_restarts": n_restarts,
            "use_py_prior": False,
            "include_order_prior": False,
            "use_annealing": use_annealing,
            "annealing_schedule_type": annealing_schedule_type,
            "annealing_plateau_frac": annealing_plateau_frac,
            "temp_min": temp_min,
            "temp_max": temp_max,
            "block_density": scenario.block_density,
            "true_n_clusters": scenario.n_clusters,
            "fit_n_clusters": n_clusters_fit,
            "init_mu": init_mu,
        },
        "runtime_seconds": elapsed,
        "tau_true": tau_true,
        "z_true": z_true,
        "true_blocks": true_blocks,
        "map_result": {
            "logp_chain": map_result["logp_chain"],
            "logp_refined": map_result["logp_refined"],
            "z": map_result["z"],
            "tau": map_result["tau"],
            "clusters": map_result["clusters"],
            "restart_summaries": map_result["restart_summaries"],
        },
        "metrics": metrics,
    }


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def print_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    THRESHOLD = 0.8
    W = 46  # width of the scenario descriptor column
    header = f"  {'Parameters':<{W}}  {'Acc':>6}  {'ARI':>6}  {'F1':>6}  {'nKem':>6}  {'|ΔK|':>5}  {'s':>5}"
    sep = "  " + "─" * (len(header) - 2)

    print()
    print(sep)
    print(header)
    print(sep)

    n_pass = 0
    metrics_list = []
    for result in results:
        sc = result["scenario"]
        m  = result["metrics"]
        metrics_list.append(m)
        acc = m["aligned_cluster_accuracy"]
        if acc >= THRESHOLD:
            n_pass += 1
        ok  = "✓" if acc >= THRESHOLD else "✗"
        bd  = sc.get("block_density", 0.0)
        desc = (
            f"C={sc['n_clusters']} N={sc['n_assessors']:>3} m={sc['n_items']:>2} "
            f"bd={bd:.2f} θ={sc['theta']}"
        )
        print(
            f"  {ok} {desc:<{W - 2}}"
            f"  {acc:>6.3f}"
            f"  {m['adjusted_rand_index']:>6.3f}"
            f"  {m['weighted_same_block_f1']:>6.3f}"
            f"  {m['weighted_normalized_kemeny_p_half']:>6.3f}"
            f"  {m['weighted_block_count_error']:>5.2f}"
            f"  {result['runtime_seconds']:>5.1f}"
        )

    print(sep)
    n = len(results)
    print(
        f"  {'─ Mean':<{W}}"
        f"  {statistics.mean(m['aligned_cluster_accuracy'] for m in metrics_list):>6.3f}"
        f"  {statistics.mean(m['adjusted_rand_index'] for m in metrics_list):>6.3f}"
        f"  {statistics.mean(m['weighted_same_block_f1'] for m in metrics_list):>6.3f}"
        f"  {statistics.mean(m['weighted_normalized_kemeny_p_half'] for m in metrics_list):>6.3f}"
        f"  {statistics.mean(m['weighted_block_count_error'] for m in metrics_list):>5.2f}"
        f"  {statistics.mean(r['runtime_seconds'] for r in results):>5.1f}"
    )
    print(sep)
    status = "all passed ✓" if n_pass == n else f"{n - n_pass} scenario(s) below acc {THRESHOLD:.0%}"
    print(f"  Passed {n_pass}/{n}  —  {status}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tied Mallows recovery simulations.")
    parser.add_argument("--mode", choices=("early", "grid", "contrast"), default="early")
    parser.add_argument("--include-c10", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--n-iter", type=int, default=8000)
    parser.add_argument("--burn-in", type=int, default=5000)
    parser.add_argument("--thin", type=int, default=20)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--use-annealing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--annealing-schedule-type", choices=("linear", "exponential", "plateau"), default="linear")
    parser.add_argument("--annealing-plateau-frac", type=float, default=0.5)
    parser.add_argument("--temp-min", type=float, default=0.1)
    parser.add_argument("--temp-max", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of scenarios to run.")
    parser.add_argument("--resume-dir", type=str, default=None, help="Resume an interrupted run from this directory.")
    args = parser.parse_args()

    if args.mode == "early":
        scenarios = early_test_scenarios()
    elif args.mode == "grid":
        scenarios = larger_grid_scenarios(include_c10=args.include_c10)
    else:
        scenarios = contrast_grid_scenarios(include_c10=args.include_c10)
    if args.limit is not None:
        scenarios = scenarios[: args.limit]

    # ── Resume mode: reuse existing directory, skip already-completed scenarios ──
    if args.resume_dir:
        out_dir = Path(args.resume_dir)
        if not out_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {out_dir}")
        done = {p.stem for p in out_dir.glob("*.json")
                if p.name not in {"run_metadata.json", "all_results.json", "scenario_summary.json"}}
        skipped = [s for s in scenarios if s.name in done]
        scenarios = [s for s in scenarios if s.name not in done]
        print(f"Resuming run in: {out_dir}")
        print(f"  Already done : {len(skipped)}, remaining: {len(scenarios)}")
        # Load already-completed results so the final summary covers everything
        results: list[dict[str, Any]] = [
            json.loads((out_dir / f"{s.name}.json").read_text(encoding="utf-8"))
            for s in skipped
        ]
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("simulation_recovery_runs") / f"recovery_{args.mode}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        base_scenario_names = sorted({s.name.rsplit("_seed", 1)[0] for s in scenarios})
        run_metadata = {
            "mode": args.mode,
            "n_iter": args.n_iter,
            "burn_in": args.burn_in,
            "thin": args.thin,
            "n_restarts": args.n_restarts,
            "include_c10": args.include_c10,
            "use_annealing": args.use_annealing,
            "annealing_schedule_type": args.annealing_schedule_type,
            "annealing_plateau_frac": args.annealing_plateau_frac,
            "temp_min": args.temp_min,
            "temp_max": args.temp_max,
            "n_trials": len(scenarios),
            "n_base_scenarios": len(base_scenario_names),
            "base_scenarios": base_scenario_names,
            "output_dir": str(out_dir),
            "created_at": timestamp,
        }
        with (out_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
            json.dump(run_metadata, fh, indent=2)
        results = []

    total = len(results) + len(scenarios)
    print(f"Running {len(scenarios)} scenario(s) in mode={args.mode}")
    print(f"Include C=10: {args.include_c10}")
    print(
        f"Annealing: {args.use_annealing}  "
        f"(type={args.annealing_schedule_type}, plateau_frac={args.annealing_plateau_frac}, "
        f"temp_min={args.temp_min}, temp_max={args.temp_max})"
    )
    print(f"Results will be saved in: {out_dir}")

    for idx, scenario in enumerate(scenarios, start=len(results) + 1):
        print()
        print(f"[{idx}/{total}] {scenario.name}")
        print(
            f"  C={scenario.n_clusters}, N={scenario.n_assessors}, n={scenario.n_items}, "
            f"bd={scenario.block_density:.2f}, theta={scenario.theta}, seed={scenario.seed}"
        )
        result = run_scenario(
            scenario,
            n_iter=args.n_iter,
            burn_in=args.burn_in,
            thin=args.thin,
            n_restarts=args.n_restarts,
            use_annealing=args.use_annealing,
            annealing_schedule_type=args.annealing_schedule_type,
            annealing_plateau_frac=args.annealing_plateau_frac,
            temp_min=args.temp_min,
            temp_max=args.temp_max,
        )
        results.append(result)
        with (out_dir / f"{scenario.name}.json").open("w", encoding="utf-8") as fh:
            json.dump(to_jsonable(result), fh, indent=2)

        metrics = result["metrics"]
        print(
            "  "
            f"acc={metrics['aligned_cluster_accuracy']:.3f}, "
            f"ARI={metrics['adjusted_rand_index']:.3f}, "
            f"nKem={metrics['weighted_normalized_kemeny_p_half']:.3f}, "
            f"BlkF1={metrics['weighted_same_block_f1']:.3f}, "
            f"|ΔK|={metrics['weighted_block_count_error']:.3f}, "
            f"time={result['runtime_seconds']:.1f}s"
        )

    with (out_dir / "all_results.json").open("w", encoding="utf-8") as fh:
        json.dump(to_jsonable(results), fh, indent=2)

    print_summary(results)


if __name__ == "__main__":
    main()