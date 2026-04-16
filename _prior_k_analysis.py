"""
Prior analysis: how (γ, δ) control the expected number of blocks K.

Since tie_penalty p cancels analytically between distance and Z*, the
only model components that influence the number of ties are:

  1. PY EPPF prior  →  controlled by (γ, δ)
  2. Z* normalizer  →  controlled by θ (always favours singletons)
  3. Order prior    →  1/K!  (always favours ties)
  4. Data fit       →  cross-block disagreements (data-dependent)

This script computes E[K] under the "effective prior" (components 1–3,
no data) to show how (γ, δ) translate into an expected number of
distinct rank positions.  The complement n − E[K] is the expected
number of "tied" positions.

Outputs
-------
  prior_k_py_only.png       E[K] under pure PY prior (recursion, any n)
  prior_k_effective.png     E[K] under effective prior (PY + Z* + 1/K!)
  prior_k_decomposition.png Decomposition: PY vs Z* vs order prior
  prior_k_heatmap.png       Heatmap of E[K]/n over (γ, δ) grid
"""

import math
from collections import Counter
from itertools import product
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from model.priors import (
    log_Z_star_from_sizes,
    log_py_eppf_from_sizes,
    build_log_qfactorials,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PY prior E[K] via exact recursion  (works for any n)
# ═══════════════════════════════════════════════════════════════════════════════

def py_expected_K(n: int, gamma: float, delta: float) -> float:
    """Exact E[K_n] under PY(γ, δ) via the CRP recursion.

    In the Chinese Restaurant Process, customer i starts a new table
    with probability (γ + K·δ) / (γ + i − 1).  Taking expectations:

        E[K_1] = 1
        E[K_i] = E[K_{i−1}] · (γ + i − 1 + δ)/(γ + i − 1)
                 + γ / (γ + i − 1)
    """
    if n <= 0:
        return 0.0
    EK = 1.0
    for i in range(2, n + 1):
        EK = EK * (gamma + i - 1 + delta) / (gamma + i - 1) + gamma / (gamma + i - 1)
    return EK


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Integer-partition enumeration for the effective prior
# ═══════════════════════════════════════════════════════════════════════════════

def _integer_partitions(n: int) -> List[Tuple[int, ...]]:
    """All partitions of n as non-increasing tuples."""
    results: List[Tuple[int, ...]] = []

    def _recurse(remaining, max_part, current):
        if remaining == 0:
            results.append(tuple(current))
            return
        for p in range(min(remaining, max_part), 0, -1):
            current.append(p)
            _recurse(remaining - p, p, current)
            current.pop()

    _recurse(n, n, [])
    return results


def _log_multinomial(sizes: List[int]) -> float:
    """log( n! / (s_1! ⋯ s_K!) )."""
    n = sum(sizes)
    return math.lgamma(n + 1) - sum(math.lgamma(s + 1) for s in sizes)


def _log_ordering_count(sizes: List[int]) -> float:
    """log( K! / ∏ m_j! )  — number of distinct orderings of the block sizes."""
    K = len(sizes)
    counts = Counter(sizes)
    return math.lgamma(K + 1) - sum(math.lgamma(m + 1) for m in counts.values())


def effective_prior_expected_K(
    n: int,
    gamma: float,
    delta: float,
    theta: float,
    N: int,
    include_order_prior: bool = True,
    include_Z_star: bool = True,
    include_py: bool = True,
) -> Tuple[float, float]:
    """Compute E[K] and std[K] under the effective prior by partition enumeration.

    Effective prior weight for an unordered partition λ = (s_1 ≥ … ≥ s_K):

        w(λ) = [n!/(∏s_k!)] × PY_EPPF(λ) × exp(−N·logZ*(λ)) × (1/K!)
               × [K!/∏m_j!]   (ordering multiplicity)

    Returns (E[K], std[K]).
    """
    partitions = _integer_partitions(n)

    log_weights = []
    Ks = []

    for part in partitions:
        sizes = list(part)
        K = len(sizes)
        Ks.append(K)

        # Multinomial: number of labelled partitions with these sizes
        log_w = _log_multinomial(sizes)

        # PY EPPF
        if include_py:
            lpy = log_py_eppf_from_sizes(sizes, gamma, delta)
            if lpy == float("-inf"):
                log_weights.append(float("-inf"))
                continue
            log_w += lpy

        # Z* normaliser (per assessor): −N · logZ*(sizes; θ)
        if include_Z_star:
            logZ = log_Z_star_from_sizes(sizes, theta)
            log_w -= N * logZ

        # Order prior: 1/K!
        if include_order_prior:
            log_w -= math.lgamma(K + 1)

        # Ordering multiplicity: K!/∏m_j!
        log_w += _log_ordering_count(sizes)

        log_weights.append(log_w)

    # Normalise in log-space
    finite_mask = [lw != float("-inf") for lw in log_weights]
    if not any(finite_mask):
        return float("nan"), float("nan")

    max_lw = max(lw for lw, f in zip(log_weights, finite_mask) if f)
    weights = []
    for lw in log_weights:
        if lw == float("-inf"):
            weights.append(0.0)
        else:
            weights.append(math.exp(lw - max_lw))

    total = sum(weights)
    EK = sum(K * w for K, w in zip(Ks, weights)) / total
    EK2 = sum(K * K * w for K, w in zip(Ks, weights)) / total
    std_K = math.sqrt(max(0.0, EK2 - EK ** 2))

    return EK, std_K


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_py_prior_EK():
    """Plot 1: Pure PY prior E[K] for various n, γ, δ."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # --- Panel (a): E[K] vs n for fixed (γ, δ) pairs ---
    ax = axes[0]
    ns = np.arange(2, 101)
    configs = [
        (0.5, 0.0, "C0", "γ=0.5, δ=0"),
        (1.0, 0.0, "C1", "γ=1, δ=0"),
        (2.0, 0.0, "C2", "γ=2, δ=0"),
        (1.0, 0.25, "C3", "γ=1, δ=0.25"),
        (1.0, 0.5, "C4", "γ=1, δ=0.5"),
        (1.0, 0.75, "C5", "γ=1, δ=0.75"),
    ]
    for gamma, delta, color, label in configs:
        EKs = [py_expected_K(int(nn), gamma, delta) for nn in ns]
        style = "--" if delta == 0 else "-"
        ax.plot(ns, EKs, style, color=color, label=label, linewidth=1.5)
    ax.set_xlabel("n (items)")
    ax.set_ylabel("E[K]  (expected blocks)")
    ax.set_title("(a) PY prior: E[K] vs n")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, None)

    # --- Panel (b): E[K] vs γ for fixed δ (n=20) ---
    ax = axes[1]
    n_val = 20
    gammas = np.linspace(0.1, 5.0, 50)
    for delta, color, ls in [(0.0, "C0", "--"), (0.25, "C3", "-"), (0.5, "C4", "-"), (0.75, "C5", "-")]:
        EKs = [py_expected_K(n_val, g, delta) for g in gammas]
        ax.plot(gammas, EKs, ls, color=color, label=f"δ={delta}", linewidth=1.5)
    ax.axhline(n_val, color="grey", linestyle=":", alpha=0.5, label="n (all singletons)")
    ax.axhline(1, color="grey", linestyle=":", alpha=0.3)
    ax.set_xlabel("γ (concentration)")
    ax.set_ylabel("E[K]")
    ax.set_title(f"(b) PY prior: E[K] vs γ  (n={n_val})")
    ax.legend(fontsize=8)

    # --- Panel (c): E[K] vs δ for fixed γ (n=20) ---
    ax = axes[2]
    deltas = np.linspace(0.0, 0.95, 50)
    for gamma, color in [(0.5, "C0"), (1.0, "C1"), (2.0, "C2"), (5.0, "C6")]:
        EKs = [py_expected_K(n_val, gamma, d) for d in deltas]
        ax.plot(deltas, EKs, "-", color=color, label=f"γ={gamma}", linewidth=1.5)
    ax.axhline(n_val, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("δ (discount)")
    ax.set_ylabel("E[K]")
    ax.set_title(f"(c) PY prior: E[K] vs δ  (n={n_val})")
    ax.legend(fontsize=8)

    fig.suptitle("PY Prior: Expected Number of Blocks (no Z*, no order prior, no data)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig("prior_k_py_only.png", dpi=150, bbox_inches="tight")
    print("Saved: prior_k_py_only.png")
    plt.close(fig)


def plot_effective_prior_EK():
    """Plot 2: Effective prior E[K] (PY + Z* + order) for small n."""
    n_val = 12  # small enough for fast partition enumeration
    N_val = 30  # representative number of assessors
    theta_vals = [0.5, 1.0, 2.0, 5.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, theta in enumerate(theta_vals):
        ax = axes[idx // 2][idx % 2]
        gammas = np.linspace(0.1, 4.0, 20)
        deltas = [0.0, 0.25, 0.5, 0.75]
        colors = ["C0", "C3", "C4", "C5"]
        styles = ["--", "-", "-", "-"]

        for delta, color, ls in zip(deltas, colors, styles):
            EKs = []
            for gamma in gammas:
                ek, _ = effective_prior_expected_K(
                    n_val, gamma, delta, theta, N_val,
                    include_order_prior=True, include_Z_star=True, include_py=True,
                )
                EKs.append(ek)
            ax.plot(gammas, EKs, ls, color=color, label=f"δ={delta}", linewidth=1.5)

        # Also show pure PY (dashed grey) for reference
        for delta, color in zip(deltas, colors):
            EK_py = [py_expected_K(n_val, g, delta) for g in gammas]
            ax.plot(gammas, EK_py, ":", color=color, alpha=0.35, linewidth=1)

        ax.axhline(n_val, color="grey", linestyle=":", alpha=0.3)
        ax.axhline(1, color="grey", linestyle=":", alpha=0.3)
        ax.set_xlabel("γ")
        ax.set_ylabel("E[K]")
        ax.set_title(f"θ = {theta},  N = {N_val}")
        ax.legend(fontsize=8)
        ax.set_ylim(0.5, n_val + 0.5)

    fig.suptitle(
        f"Effective Prior E[K]  (PY + Z* + 1/K! order prior,  n={n_val})\n"
        "Solid/dashed = effective prior;  dotted = pure PY prior (for reference)",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("prior_k_effective.png", dpi=150, bbox_inches="tight")
    print("Saved: prior_k_effective.png")
    plt.close(fig)


def plot_decomposition():
    """Plot 3: Show how each component shifts E[K]."""
    n_val = 12
    N_val = 30
    theta = 2.0

    gammas = np.linspace(0.1, 4.0, 25)
    delta = 0.5  # fix one δ for clarity

    labels_configs = [
        ("PY only",         dict(include_py=True,  include_Z_star=False, include_order_prior=False)),
        ("PY + Z*",         dict(include_py=True,  include_Z_star=True,  include_order_prior=False)),
        ("PY + 1/K!",       dict(include_py=True,  include_Z_star=False, include_order_prior=True)),
        ("PY + Z* + 1/K!",  dict(include_py=True,  include_Z_star=True,  include_order_prior=True)),
    ]
    colors = ["C0", "C1", "C2", "C4"]
    styles = ["--", "-.", ":", "-"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for (label, kwargs), color, ls in zip(labels_configs, colors, styles):
        EKs = []
        for gamma in gammas:
            ek, _ = effective_prior_expected_K(
                n_val, gamma, delta, theta, N_val, **kwargs
            )
            EKs.append(ek)
        ax.plot(gammas, EKs, ls, color=color, label=label, linewidth=2)

    ax.axhline(n_val, color="grey", linestyle=":", alpha=0.3, label=f"n={n_val} (all singletons)")
    ax.axhline(1, color="grey", linestyle=":", alpha=0.3)
    ax.set_xlabel("γ (concentration)", fontsize=11)
    ax.set_ylabel("E[K]  (expected number of blocks)", fontsize=11)
    ax.set_title(
        f"Decomposition: how each component shifts E[K]\n"
        f"(n={n_val}, N={N_val}, θ={theta}, δ={delta})",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.set_ylim(0.5, n_val + 0.5)

    fig.tight_layout()
    fig.savefig("prior_k_decomposition.png", dpi=150, bbox_inches="tight")
    print("Saved: prior_k_decomposition.png")
    plt.close(fig)


def plot_heatmap():
    """Plot 4: Heatmap of E[K]/n over (γ, δ) grid."""
    n_val = 12
    N_val = 30

    gamma_grid = np.linspace(0.2, 4.0, 20)
    delta_grid = np.linspace(0.0, 0.9, 19)
    theta_vals = [1.0, 3.0]

    fig, axes = plt.subplots(1, len(theta_vals) + 1, figsize=(5 * (len(theta_vals) + 1), 4.5))

    # Pure PY prior
    ax = axes[0]
    EK_grid = np.zeros((len(delta_grid), len(gamma_grid)))
    for i, delta in enumerate(delta_grid):
        for j, gamma in enumerate(gamma_grid):
            EK_grid[i, j] = py_expected_K(n_val, gamma, delta)

    im = ax.imshow(
        EK_grid / n_val, aspect="auto", origin="lower",
        extent=[gamma_grid[0], gamma_grid[-1], delta_grid[0], delta_grid[-1]],
        vmin=0, vmax=1, cmap="RdYlBu_r",
    )
    ax.set_xlabel("γ")
    ax.set_ylabel("δ")
    ax.set_title("Pure PY prior")
    # Add contour lines
    cs = ax.contour(
        gamma_grid, delta_grid, EK_grid,
        levels=[2, 4, 6, 8, 10], colors="black", linewidths=0.8, alpha=0.6,
    )
    ax.clabel(cs, fmt="K=%.0f", fontsize=8)

    # Effective prior for each θ
    for t_idx, theta in enumerate(theta_vals):
        ax = axes[t_idx + 1]
        EK_eff = np.zeros((len(delta_grid), len(gamma_grid)))
        for i, delta in enumerate(delta_grid):
            for j, gamma in enumerate(gamma_grid):
                ek, _ = effective_prior_expected_K(
                    n_val, gamma, delta, theta, N_val
                )
                EK_eff[i, j] = ek

        im = ax.imshow(
            EK_eff / n_val, aspect="auto", origin="lower",
            extent=[gamma_grid[0], gamma_grid[-1], delta_grid[0], delta_grid[-1]],
            vmin=0, vmax=1, cmap="RdYlBu_r",
        )
        ax.set_xlabel("γ")
        ax.set_ylabel("δ")
        ax.set_title(f"Effective prior (θ={theta}, N={N_val})")
        cs = ax.contour(
            gamma_grid, delta_grid, EK_eff,
            levels=[2, 4, 6, 8, 10], colors="black", linewidths=0.8, alpha=0.6,
        )
        ax.clabel(cs, fmt="K=%.0f", fontsize=8)

    cb = fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="E[K] / n  (1 = all singletons)")

    fig.suptitle(
        f"Expected fraction of distinct ranks  (n={n_val}, N={N_val})\n"
        "Blue = more ties,  Red = fewer ties",
        fontsize=12, y=1.05,
    )
    fig.tight_layout()
    fig.savefig("prior_k_heatmap.png", dpi=150, bbox_inches="tight")
    print("Saved: prior_k_heatmap.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Textual summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary_table():
    """Print a concise numerical table for a few representative configs."""
    print()
    print("=" * 85)
    print("  NUMERICAL SUMMARY: PY prior expected K for various (γ, δ, n)")
    print("=" * 85)
    print()

    ns = [10, 20, 50, 100]
    configs = [
        (0.5, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (0.5, 0.25),
        (1.0, 0.25),
        (1.0, 0.5),
        (2.0, 0.5),
        (1.0, 0.75),
    ]

    header = f"{'γ':>5} {'δ':>5}" + "".join(f"{'n='+str(n):>12}" for n in ns)
    print(header)
    print("-" * len(header))
    for gamma, delta in configs:
        row = f"{gamma:>5.1f} {delta:>5.2f}"
        for n in ns:
            ek = py_expected_K(n, gamma, delta)
            frac = ek / n
            row += f"  {ek:>5.1f} ({frac:.0%})"
        print(row)

    print()
    print("Values shown as: E[K] (E[K]/n).  E[K]/n = 1 means all singletons (no ties).")
    print()

    # Effective prior table
    n_val = 12
    N_val = 30
    print()
    print("=" * 75)
    print(f"  EFFECTIVE PRIOR E[K]  (n={n_val}, N={N_val}, with Z* + 1/K! order prior)")
    print("=" * 75)
    print()

    thetas = [0.5, 1.0, 2.0, 5.0]
    header2 = f"{'γ':>5} {'δ':>5}" + "".join(f"{'θ='+str(t):>12}" for t in thetas) + f"{'PY only':>12}"
    print(header2)
    print("-" * len(header2))
    for gamma, delta in configs:
        row = f"{gamma:>5.1f} {delta:>5.2f}"
        for theta in thetas:
            ek, _ = effective_prior_expected_K(n_val, gamma, delta, theta, N_val)
            row += f"  {ek:>5.1f} ({ek/n_val:.0%})"
        ek_py = py_expected_K(n_val, gamma, delta)
        row += f"  {ek_py:>5.1f} ({ek_py/n_val:.0%})"
        print(row)

    print()
    print("KEY INSIGHTS:")
    print("  • γ (concentration): higher γ → more blocks → fewer ties")
    print("  • δ (discount): higher δ → E[K] grows as n^δ → many more blocks for large n")
    print("  • δ = 0 (Dirichlet process): E[K] ~ γ·log(n), slow growth → ties persist at large n")
    print("  • Z* always pushes toward singletons (more blocks)")
    print("  • 1/K! order prior always pushes toward ties (fewer blocks)")
    print("  • At low θ: Z* is weak, so PY + order prior dominate → more ties")
    print("  • At high θ: Z* is strong, overwhelms PY → singletons regardless of γ,δ")
    print("  • The 'sweet spot' for controlled ties: moderate θ, small γ, small δ")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print_summary_table()
    print("Generating plots...")
    plot_py_prior_EK()
    plot_effective_prior_EK()
    plot_decomposition()
    plot_heatmap()
    print("\nDone.")
