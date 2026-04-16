"""Visualize tied Mallows recovery simulation results.

Given one run directory produced by _simulation_recovery_study.py, this script
aggregates results over seed trials and saves:

  - scenario_summary.json
  - scenario_summary.csv
  - recovery_overview.png

The plots use run-level metadata (n_iter, burn_in, n_restarts, n_trials) in the
titles so the experiment setup stays attached to the figures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


METRIC_KEYS = [
    ("aligned_cluster_accuracy", "Cluster Accuracy"),
    ("adjusted_rand_index", "ARI"),
    ("weighted_normalized_kemeny_p_half", "Normalized Kemeny"),
    ("weighted_same_block_f1", "Block F1"),
]


def strip_seed(name: str) -> str:
    if "_seed" in name:
        return name.rsplit("_seed", 1)[0]
    return name


def latest_run_dir(base_dir: Path) -> Path:
    candidates = [p for p in base_dir.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found in {base_dir}")
    return sorted(candidates)[-1]


def load_run(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata_path = run_dir / "run_metadata.json"
    results_path = run_dir / "all_results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results file: {results_path}")

    metadata: dict[str, Any]
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {"output_dir": str(run_dir)}
    results = json.loads(results_path.read_text(encoding="utf-8"))
    return metadata, results


def aggregate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        base_name = strip_seed(result["scenario"]["name"])
        grouped.setdefault(base_name, []).append(result)

    summary_rows: list[dict[str, Any]] = []
    for base_name, rows in sorted(grouped.items()):
        example = rows[0]
        metrics = [row["metrics"] for row in rows]
        scenario = example["scenario"]
        out: dict[str, Any] = {
            "base_scenario": base_name,
            "n_trials": len(rows),
            "n_clusters_true": scenario["n_clusters"],
            "n_assessors": scenario["n_assessors"],
            "n_items": scenario["n_items"],
            "theta": scenario["theta"],
            "fit_n_clusters": example["settings"].get("fit_n_clusters"),
            "n_iter": example["settings"].get("n_iter"),
            "burn_in": example["settings"].get("burn_in"),
            "thin": example["settings"].get("thin"),
            "n_restarts": example["settings"].get("n_restarts"),
        }
        for key, _label in METRIC_KEYS + [
            ("weighted_strict_inversions", "Strict Inversions"),
            ("weighted_block_count_error", "Block Count Error"),
        ]:
            vals = [float(m[key]) for m in metrics]
            out[f"{key}_mean"] = statistics.mean(vals)
            out[f"{key}_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        summary_rows.append(out)
    return summary_rows


def save_summary(summary_rows: list[dict[str, Any]], run_dir: Path) -> None:
    json_path = run_dir / "scenario_summary.json"
    csv_path = run_dir / "scenario_summary.csv"
    json_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    if summary_rows:
        fieldnames = list(summary_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)


def plot_overview(metadata: dict[str, Any], summary_rows: list[dict[str, Any]], run_dir: Path) -> None:
    if not summary_rows:
        return

    labels = [row["base_scenario"] for row in summary_rows]
    x = list(range(len(labels)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    for ax, color, (metric_key, metric_label) in zip(axes, colors, METRIC_KEYS):
        means = [row[f"{metric_key}_mean"] for row in summary_rows]
        sds = [row[f"{metric_key}_sd"] for row in summary_rows]
        ax.errorbar(x, means, yerr=sds, fmt="o-", color=color, ecolor=color, capsize=4, linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(metric_label)
        ax.grid(alpha=0.25, linestyle=":")
        if metric_key in {"aligned_cluster_accuracy", "adjusted_rand_index", "weighted_normalized_kemeny_p_half", "weighted_same_block_f1"}:
            ax.set_ylim(0, 1.02)

    title = (
        "Recovery Overview\n"
        f"trials={metadata.get('n_trials', 'NA')}, n_iter={metadata.get('n_iter', 'NA')}, "
        f"burn_in={metadata.get('burn_in', 'NA')}, restarts={metadata.get('n_restarts', 'NA')}"
    )
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    fig.savefig(run_dir / "recovery_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tied Mallows recovery results.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a run directory. Defaults to the newest folder under simulation_recovery_runs.",
    )
    args = parser.parse_args()

    base_dir = Path("simulation_recovery_runs")
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir(base_dir)
    metadata, results = load_run(run_dir)
    summary_rows = aggregate_results(results)
    save_summary(summary_rows, run_dir)
    plot_overview(metadata, summary_rows, run_dir)

    print(f"Saved summary JSON/CSV and overview plot in: {run_dir}")


if __name__ == "__main__":
    main()