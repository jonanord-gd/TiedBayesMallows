"""
Test the identifiability criterion against empirical recovery metrics.

Loads the existing OAT simulation results, computes D_min from the stored
ground-truth blocks and thetas (no MCMC needed), and prints/plots the
D_min/log(C) ratio against empirical cluster NMI and ARI.
"""
import json
import math
import random
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))
from model.identifiability import log_one_over_beta_pair

# ── Load data ─────────────────────────────────────────────────────────────────
RUN_DIR = Path(__file__).parent / "simulation_recovery_runs" / "recovery_timmm_raw_oat_spectral_20260508_084453"
files = sorted(RUN_DIR.glob("*.json"))
all_data = []
for f in files:
    with open(f) as fh:
        chunk = json.load(fh)
    if isinstance(chunk, list):
        all_data.extend(x for x in chunk if isinstance(x, dict) and "scenario" in x)
    elif isinstance(chunk, dict) and "scenario" in chunk:
        all_data.append(chunk)

print(f"Loaded {len(all_data)} scenarios.")

# ── Compute D_min for each scenario ──────────────────────────────────────────
rng = random.Random(7)
N_SIS = 400   # SIS samples per pair

records = []
for idx, item in enumerate(all_data):
    sc  = item["scenario"]
    gt  = item["ground_truth"]["clusters"]
    C   = sc["n_clusters"]
    n   = sc["n_items"]
    N   = sc["n_assessors"]
    theta_true = sc["theta"]
    bd         = sc["block_density"]
    nmi  = item["metrics"]["cluster_nmi"]
    ari  = item["metrics"]["cluster_ari"]

    blocks_list = [c["blocks"] for c in gt]
    theta_list  = [c["theta"]  for c in gt]

    # Pairwise log(1/beta) from true parameters
    log_C     = math.log(C)
    pair_vals = []
    for ci in range(C):
        for cp in range(ci + 1, C):
            v = log_one_over_beta_pair(
                blocks_list[ci], theta_list[ci],
                blocks_list[cp], theta_list[cp],
                n, n_samples=N_SIS, rng=rng,
            )
            pair_vals.append(v)

    dmin  = N * min(pair_vals)
    ratio = dmin / log_C   # > 1 → recoverable; < 1 → not

    records.append({
        "N": N, "n": n, "C": C,
        "theta": theta_true, "bd": bd,
        "dmin": dmin, "log_C": log_C, "ratio": ratio,
        "nmi": nmi, "ari": ari,
    })

    if (idx + 1) % 50 == 0:
        print(f"  processed {idx+1}/{len(all_data)}")

print(f"\nDone. {len(records)} records.")

# ── Print a quick summary ─────────────────────────────────────────────────────
def group_summary(records, key):
    groups = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    print(f"\n{'='*60}")
    print(f"  Varying {key}")
    print(f"{'='*60}")
    print(f"  {key:>8}  ratio_median  ratio_min  NMI_med  ARI_med  recoverable%")
    for val in sorted(groups):
        g = groups[val]
        ratios = [r["ratio"] for r in g]
        nmis   = [r["nmi"]   for r in g]
        aris   = [r["ari"]   for r in g]
        pct_rec = 100 * sum(r > 1 for r in ratios) / len(ratios)
        print(f"  {val:>8}  {np.median(ratios):12.2f}  {min(ratios):9.2f}  "
              f"{np.median(nmis):7.3f}  {np.median(aris):7.3f}  {pct_rec:6.1f}%")

for key in ["N", "n", "C", "theta", "bd"]:
    group_summary(records, key)

# ── Plot: D_min/log(C) vs NMI, faceted by varying parameter ─────────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
param_labels = {
    "N":     ("N (assessors)", [50, 100, 500]),
    "n":     ("n (items)",     [10, 20, 50]),
    "C":     ("C (clusters)",  [5, 20, 40]),
    "theta": ("θ̄",             [0.1, 0.5, 1.0]),
    "bd":    ("block density", [0.1, 0.5, 1.0]),
}
colors = plt.cm.tab10.colors

for ax, (key, (label, vals)) in zip(axes, param_labels.items()):
    for vi, val in enumerate(vals):
        subset = [r for r in records if r[key] == val]
        xs = [r["ratio"] for r in subset]
        ys = [r["nmi"]   for r in subset]
        ax.scatter(xs, ys, alpha=0.35, s=18, color=colors[vi], label=str(val))
    ax.axvline(1.0, color="red", lw=1.2, ls="--", label="threshold")
    ax.set_xlabel(r"$\hat{D}_{\min}\,/\,\log C$", fontsize=10)
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=7, title=key)
    ax.set_xlim(left=0)

axes[0].set_ylabel("Cluster NMI (empirical)", fontsize=10)
fig.suptitle(r"Identifiability ratio $\hat{D}_{\min}/\log C$ vs empirical cluster recovery"
             r"(true parameters used for $\hat{D}_{min}$)", fontsize=11)
fig.tight_layout()
out = Path(__file__).parent / "figures" / "identifiability_vs_nmi.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nFigure saved → {out}")
plt.show()
