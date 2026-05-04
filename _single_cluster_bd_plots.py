"""Visualize single-cluster block-density recovery results.

Produces a figure that shows the key block-recovery metrics as a function of
block_density, faceted by n_items (columns), with one line per
(n_assessors, theta) combination.  This lets you read off both the main trend
and robustness to N and m simultaneously.

Metrics plotted (one row each):
  1. True vs. predicted number of blocks  (two lines, shared y-axis)
  2. Signed block-count error  (pred_k − true_k)
  3. Block ARI              (item-level partition similarity)
  4. Block F1               (pairwise same-block precision/recall)
  5. Normalized Kemeny      (ordering accuracy)

Usage
-----
    python _single_cluster_bd_plots.py                      # newest run
    python _single_cluster_bd_plots.py --run-dir <path>
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.lines as mlines


# ── ARI helper ──────────────────────────────────────────────────────────────

def _comb2(x: int) -> int:
    return x * (x - 1) // 2


def _ari(labels_a: list[int], labels_b: list[int]) -> float:
    """Adjusted Rand Index between two label vectors."""
    from collections import Counter
    n = len(labels_a)
    pairs_ab: dict[tuple[int, int], int] = Counter(zip(labels_a, labels_b))
    a_counts: dict[int, int] = Counter(labels_a)
    b_counts: dict[int, int] = Counter(labels_b)

    sum_comb = sum(_comb2(v) for v in pairs_ab.values())
    sum_a    = sum(_comb2(v) for v in a_counts.values())
    sum_b    = sum(_comb2(v) for v in b_counts.values())
    total    = _comb2(n)
    if total == 0:
        return 1.0
    expected = sum_a * sum_b / total
    maximum  = 0.5 * (sum_a + sum_b)
    denom    = maximum - expected
    return (sum_comb - expected) / denom if denom != 0 else 0.0


def _block_labels(blocks: list[Any], n_items: int) -> list[int]:
    """Convert a list-of-blocks representation to a per-item label vector."""
    out = [-1] * n_items
    for bid, block in enumerate(blocks):
        # blocks can be stored as lists of ints or space-separated strings
        items: list[int]
        if isinstance(block, str):
            items = [int(x) for x in block.split()]
        else:
            items = [int(x) for x in block]
        for item in items:
            out[item] = bid
    return out


def block_ari_from_per_cluster(pc: dict[str, Any], n_items: int) -> float:
    true_labels = _block_labels(pc["true_blocks"], n_items)
    pred_labels = _block_labels(pc["pred_blocks"], n_items)
    if all(l == -1 for l in pred_labels):
        return 0.0
    return _ari(true_labels, pred_labels)


# ── Data loading ─────────────────────────────────────────────────────────────

def latest_run_dir(base_dir: Path) -> Path:
    candidates = sorted(
        p for p in base_dir.iterdir()
        if p.is_dir() and "single_cluster_bd" in p.name
    )
    if not candidates:
        raise FileNotFoundError(f"No single_cluster_bd run found in {base_dir}")
    return candidates[-1]


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    p = run_dir / "all_results.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    # fall back to individual files
    results = []
    for f in sorted(run_dir.glob("*.json")):
        if f.name in {"run_metadata.json", "all_results.json", "scenario_summary.json"}:
            continue
        results.append(json.loads(f.read_text(encoding="utf-8")))
    return results


def extract_row(result: dict[str, Any]) -> dict[str, Any]:
    sc = result["scenario"]
    m  = result["metrics"]
    pc = m["per_cluster"][0]
    n_items = sc["n_items"]

    # derive block ARI if not stored (run predates new field)
    b_ari = m.get("weighted_block_ari")
    if b_ari is None:
        b_ari = block_ari_from_per_cluster(pc, n_items)

    signed_err = m.get("weighted_block_count_signed_error")
    if signed_err is None:
        signed_err = float(pc["pred_n_blocks"] - pc["true_n_blocks"])

    return {
        "block_density": sc["block_density"],
        "n_assessors":   sc["n_assessors"],
        "n_items":       n_items,
        "theta":         sc["theta"],
        "true_n_blocks": pc["true_n_blocks"],
        "pred_n_blocks": pc["pred_n_blocks"],
        "signed_error":  signed_err,
        "block_ari":     b_ari,
        "block_f1":      m["weighted_same_block_f1"],
        "norm_kemeny":   m["weighted_normalized_kemeny_p_half"],
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

N_ASSESSORS_VALUES = (100, 300, 700)
THETA_VALUES       = (0.1, 0.5, 1.0, 3.0)
BLOCK_DENSITIES    = (0.00, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00)

# Color encodes n_assessors; line style encodes theta
_N_COLORS  = {100: "#e41a1c", 300: "#377eb8", 700: "#4daf4a"}
_T_STYLES  = {0.1: (0, (1, 1)), 0.5: "--", 1.0: "-.", 3.0: "-"}
_T_LABELS  = {0.1: "θ=0.1", 0.5: "θ=0.5", 1.0: "θ=1.0", 3.0: "θ=3.0"}
_N_LABELS  = {100: "N=100", 300: "N=300", 700: "N=700"}


def _build_grid(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, int, float, float], dict[str, Any]]:
    """Index rows by (n_items, n_assessors, theta, block_density)."""
    grid: dict[tuple[int, int, float, float], dict[str, Any]] = {}
    for r in rows:
        key = (r["n_items"], r["n_assessors"], r["theta"], r["block_density"])
        grid[key] = r
    return grid


METRICS = [
    ("block_count",  "Number of blocks"),
    ("norm_kemeny",  "Normalized Kemeny distance"),
]

N_ITEMS_VALUES = (10, 20, 50, 100)


def plot_single_cluster_bd(
    rows: list[dict[str, Any]],
    run_dir: Path,
) -> None:
    grid = _build_grid(rows)

    n_rows = len(METRICS)
    n_cols = len(N_ITEMS_VALUES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.0 * n_rows),
        sharex=True,
    )

    bd_x = list(BLOCK_DENSITIES)

    for row_idx, (metric_key, metric_label) in enumerate(METRICS):
        for col_idx, n_items in enumerate(N_ITEMS_VALUES):
            ax = axes[row_idx][col_idx]

            for n_assessors in N_ASSESSORS_VALUES:
                for theta in THETA_VALUES:
                    color = _N_COLORS[n_assessors]
                    ls    = _T_STYLES[theta]

                    if metric_key == "block_count":
                        true_vals = []
                        pred_vals = []
                        for bd in BLOCK_DENSITIES:
                            r = grid.get((n_items, n_assessors, theta, bd))
                            if r:
                                true_vals.append(r["true_n_blocks"])
                                pred_vals.append(r["pred_n_blocks"])
                            else:
                                true_vals.append(float("nan"))
                                pred_vals.append(float("nan"))
                        # True blocks: thicker grey line (same for all N/theta)
                        if n_assessors == N_ASSESSORS_VALUES[0] and theta == THETA_VALUES[0]:
                            ax.plot(bd_x, true_vals, color="black", lw=2, ls=":", label="true")
                        ax.plot(bd_x, pred_vals, color=color, ls=ls, lw=1.5, marker="o", ms=4)
                    else:
                        vals = []
                        for bd in BLOCK_DENSITIES:
                            r = grid.get((n_items, n_assessors, theta, bd))
                            vals.append(r[metric_key] if r else float("nan"))
                        ax.plot(bd_x, vals, color=color, ls=ls, lw=1.5, marker="o", ms=4)

            # reference lines
            if metric_key == "signed_error":
                ax.axhline(0, color="black", lw=0.8, ls="--")
            elif metric_key in ("block_ari", "block_f1"):
                ax.set_ylim(-0.05, 1.05)
            elif metric_key == "norm_kemeny":
                ax.set_ylim(0, 1.0)

            if row_idx == 0:
                ax.set_title(f"m = {n_items}", fontsize=13, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=10)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Block density", fontsize=10)

            ax.set_xticks(bd_x)
            ax.grid(alpha=0.25, linestyle=":")

    # Legend
    handles = []
    for n in N_ASSESSORS_VALUES:
        handles.append(mlines.Line2D([], [], color=_N_COLORS[n], lw=2, label=_N_LABELS[n]))
    for t in THETA_VALUES:
        handles.append(mlines.Line2D([], [], color="grey", lw=1.5, ls=_T_STYLES[t], label=_T_LABELS[t]))
    handles.append(mlines.Line2D([], [], color="black", lw=2,  ls=":", label="true K"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               bbox_to_anchor=(0.5, 1.01), fontsize=10, frameon=True)

    fig.suptitle(
        "Single-cluster block-density recovery  (C=1, seed=101)",
        fontsize=15, y=1.04,
    )
    fig.tight_layout()
    out_path = run_dir / "single_cluster_bd_recovery.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, default=None)
    args = parser.parse_args()

    base_dir = Path("simulation_recovery_runs")
    run_dir  = Path(args.run_dir) if args.run_dir else latest_run_dir(base_dir)

    print(f"Loading results from: {run_dir}")
    results = load_results(run_dir)
    rows    = [extract_row(r) for r in results]
    print(f"  {len(rows)} scenarios loaded")

    plot_single_cluster_bd(rows, run_dir)


if __name__ == "__main__":
    main()
