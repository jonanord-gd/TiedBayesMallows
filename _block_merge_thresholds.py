"""Generalized threshold for merging two blocks.

For merging adjacent consensus blocks of sizes s1 and s2, the data-driven
threshold is obtained from

    ΔlogL = θ · m - N · ΔlogZ*

where m is the total disagreement count and ΔlogZ* is the increase in the
Mallows normalizer when the two blocks are merged. If \bar{h} denotes the
critical disagreement fraction, then

    \bar{h}(s1, s2, θ) = ΔlogZ*(s1, s2, θ) / θ.

For finite N, the observable fractions lie on the grid

    {0, 1/N, 2/N, ..., 1/2},

so the finite-N threshold is the smallest grid value strictly above the
analytical \bar{h}.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from model.priors import build_log_qfactorials


def block_merge_delta_logz(s1: int, s2: int, theta: float, log_qfact: list[float]) -> float:
    """Increase in log Z* when merging blocks of sizes s1 and s2.

    Only the local block terms matter; all other block contributions cancel.
    """
    if theta <= 0:
        return float("inf")

    return (
        math.lgamma(s1 + s2 + 1)
        - math.lgamma(s1 + 1)
        - math.lgamma(s2 + 1)
        + log_qfact[s1]
        + log_qfact[s2]
        - log_qfact[s1 + s2]
    )


def block_merge_threshold(s1: int, s2: int, theta: float, log_qfact: list[float]) -> float:
    """Critical fraction threshold \bar{h}/N for preferring a merge."""
    return block_merge_delta_logz(s1, s2, theta, log_qfact) / theta


def finite_n_threshold(analytical_threshold: float, n_assessors: int) -> float:
    """Smallest observable fraction above the analytical threshold."""
    return (math.floor(n_assessors * analytical_threshold) + 1) / n_assessors


def plot_block_merge_thresholds():
    thetas = np.linspace(0.05, 10.0, 500)
    finite_n_thetas = np.linspace(0.05, 10.0, 4000)
    merge_pairs = [(1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (4, 2), (4, 3)]
    n_assessors_example = 2000
    max_size = max(s1 + s2 for s1, s2 in merge_pairs)

    analytical_curves: dict[tuple[int, int], list[float]] = {}
    empirical_curves: dict[tuple[int, int], list[float]] = {}
    for s1, s2 in merge_pairs:
        analytical_vals = []
        for theta in thetas:
            q = math.exp(-theta)
            log_qfact = build_log_qfactorials(max_size, q)
            threshold = block_merge_threshold(s1, s2, theta, log_qfact)
            analytical_vals.append(threshold)
        analytical_curves[(s1, s2)] = analytical_vals

        empirical_vals = []
        for theta in finite_n_thetas:
            q = math.exp(-theta)
            log_qfact = build_log_qfactorials(max_size, q)
            threshold = block_merge_threshold(s1, s2, theta, log_qfact)
            empirical_vals.append(finite_n_threshold(threshold, n_assessors_example))
        empirical_curves[(s1, s2)] = empirical_vals

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6.2))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(merge_pairs)))
    for color, (s1, s2) in zip(colors, merge_pairs):
        ax.plot(thetas, analytical_curves[(s1, s2)], color=color, linewidth=2.3, label=f"({s1},{s2})")
        dashed_line = ax.step(
            finite_n_thetas,
            empirical_curves[(s1, s2)],
            where="post",
            color="#2b2b2b",
            linestyle="--",
            linewidth=1.15,
            alpha=0.8,
        )
        dashed_line[0].set_path_effects([
            pe.Stroke(linewidth=2.8, foreground="white", alpha=0.95),
            pe.Normal(),
        ])
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.6)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel(r"$\theta$", fontsize=13)
    ax.set_ylabel(r"Threshold $\bar{h}$", fontsize=13)
    ax.set_title(r"Block-merge threshold $\bar{h}(\theta)$", fontsize=15)
    ax.legend(title="Merge sizes (s1,s2)", fontsize=9)

    fig.suptitle(
        rf"Solid: analytical threshold, dashed black: finite-$N$ threshold with $N={n_assessors_example}$",
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig("block_merge_thresholds.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: block_merge_thresholds.png")

    print()
    print("Representative values:")
    for theta in (0.5, 1.0, 2.0, 5.0):
        q = math.exp(-theta)
        log_qfact = build_log_qfactorials(max_size, q)
        parts = []
        for s1, s2 in merge_pairs:
            parts.append(f"({s1},{s2}): {block_merge_threshold(s1, s2, theta, log_qfact):.3f}")
        print(f"  theta={theta:.1f} -> " + ", ".join(parts))

    print()
    print(f"Finite-N overlay uses N={n_assessors_example} assessors.")

if __name__ == "__main__":
    plot_block_merge_thresholds()
