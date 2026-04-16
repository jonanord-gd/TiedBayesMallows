"""
Tie threshold analysis: relationship between p and data disagreement.
=====================================================================

Analytical result (pair of items i,j):
  Let m = min(f, 1-f) where f = fraction of assessors ranking i > j.
  Tying i,j is preferred (higher posterior) when:

      m  >  p  +  log(2 / (1+q)) / theta        where q = exp(-theta)

  As theta -> inf:   m > p   (threshold = p exactly)
  As theta -> 0:     threshold -> 0.5 + infinity -> ties never preferred

This script:
  1. Plots the analytical threshold curve m*(p) for several theta values
  2. Verifies numerically with fabricated data (n items, controlled pair disagreement)
  3. Sweeps (p, m) space to show the tie-preference boundary
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model.priors import log_Z_star_from_sizes
from model.distance import total_distance_fast

# ── 1. Analytical threshold ──────────────────────────────────────────

def analytical_threshold(p: float, theta: float) -> float:
    """Critical minority fraction above which ties are preferred."""
    q = math.exp(-theta)
    delta_Z = math.log(2.0 / (1.0 + q))
    return p + delta_Z / theta


def max_p_for_ties(theta: float) -> float:
    """Maximum p at which ties are ever possible (threshold < 0.5)."""
    q = math.exp(-theta)
    delta_Z = math.log(2.0 / (1.0 + q))
    return 0.5 - delta_Z / theta


# ── 2. Numerical verification with fabricated data ───────────────────

def fabricate_rankings(n: int, N: int, swap_pair: tuple, minority_frac: float, seed: int = 42):
    """
    Create N rankings of n items.
    All agree on items 2..n-1 (fixed order).
    For the swap_pair (default 0,1): fraction (1-minority_frac) rank 0<1,
    fraction minority_frac rank 1<0.
    """
    rng = np.random.RandomState(seed)
    base = list(range(n))  # 0, 1, 2, ..., n-1
    n_minority = int(round(minority_frac * N))
    rankings = []
    for i in range(N):
        r = base.copy()
        if i < n_minority:
            # swap the pair
            a, b = swap_pair
            ia, ib = r.index(a), r.index(b)
            r[ia], r[ib] = r[ib], r[ia]
        rankings.append(r)
    return rankings


def log_posterior_pair(rankings, n, theta, p, tied: bool):
    """Log posterior for singletons vs tying items 0,1. No PY, no order prior."""
    if tied:
        blocks = [[0, 1]] + [[i] for i in range(2, n)]
    else:
        blocks = [[i] for i in range(n)]
    N = len(rankings)
    S = total_distance_fast(rankings, blocks, tie_penalty=p)
    sizes = [len(b) for b in blocks]
    logZ = log_Z_star_from_sizes(sizes, theta)
    return -theta * S - N * logZ


def numerical_threshold(n: int, N: int, theta: float, p: float,
                        m_grid=np.linspace(0.0, 0.5, 201)):
    """Find the minority fraction at which tie becomes preferred, numerically."""
    for m in m_grid:
        R = fabricate_rankings(n, N, (0, 1), m)
        lp_untied = log_posterior_pair(R, n, theta, p, tied=False)
        lp_tied = log_posterior_pair(R, n, theta, p, tied=True)
        if lp_tied > lp_untied:
            return m
    return None  # ties never preferred


# ── 3. Plots ─────────────────────────────────────────────────────────

def plot_analytical_thresholds():
    """Plot m*(p) curves for different theta values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: threshold curves
    ax = axes[0]
    p_vals = np.linspace(0, 0.5, 200)
    thetas = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(thetas)))

    for theta, c in zip(thetas, colors):
        m_star = [analytical_threshold(p, theta) for p in p_vals]
        ax.plot(p_vals, m_star, color=c, linewidth=2, label=f'θ={theta}')

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='m=0.5 (maximum)')
    ax.fill_between(p_vals, 0, p_vals, alpha=0.08, color='green', label='m > p region (θ→∞)')
    ax.set_xlabel('Tie penalty p')
    ax.set_ylabel('Minimum disagreement fraction m* for tie')
    ax.set_title('Analytical tie threshold: m* = p + log(2/(1+q))/θ')
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Right: max p that allows ties
    ax = axes[1]
    thetas_dense = np.linspace(0.3, 50, 300)
    max_ps = [max_p_for_ties(t) for t in thetas_dense]
    ax.plot(thetas_dense, max_ps, 'b-', linewidth=2)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('θ')
    ax.set_ylabel('Maximum p allowing ties')
    ax.set_title('How θ controls the max p for which ties are possible')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 0.55)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tie_threshold_analytical.png', dpi=150)
    plt.close()
    print("Saved: tie_threshold_analytical.png")


def plot_numerical_verification():
    """Verify analytical thresholds with numerical computation."""
    n, N = 6, 100
    thetas = [1.0, 3.0, 10.0]
    p_vals = np.linspace(0.01, 0.45, 30)

    fig, ax = plt.subplots(figsize=(8, 6))

    for theta in thetas:
        # Analytical
        m_analytic = [analytical_threshold(p, theta) for p in p_vals]
        line, = ax.plot(p_vals, m_analytic, '--', linewidth=2, label=f'θ={theta} (analytical)')

        # Numerical
        m_numeric = []
        for p in p_vals:
            mt = numerical_threshold(n, N, theta, p)
            m_numeric.append(mt if mt is not None else np.nan)
        ax.scatter(p_vals, m_numeric, color=line.get_color(), s=20, zorder=5,
                   label=f'θ={theta} (numerical, n={n}, N={N})')

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Tie penalty p')
    ax.set_ylabel('Threshold minority fraction m*')
    ax.set_title(f'Tie threshold: analytical vs numerical (n={n}, N={N})')
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('tie_threshold_verification.png', dpi=150)
    plt.close()
    print("Saved: tie_threshold_verification.png")


def plot_heatmap():
    """Heatmap of log-posterior difference (tied - untied) over (p, m) space."""
    n, N = 6, 200
    theta = 1.0

    p_vals = np.linspace(0.01, 0.5, 60)
    m_vals = np.linspace(0.0, 0.5, 60)
    Z = np.full((len(m_vals), len(p_vals)), np.nan)

    for i, m in enumerate(m_vals):
        R = fabricate_rankings(n, N, (0, 1), m)
        for j, p in enumerate(p_vals):
            lp_tied = log_posterior_pair(R, n, theta, p, tied=True)
            lp_untied = log_posterior_pair(R, n, theta, p, tied=False)
            Z[i, j] = lp_tied - lp_untied

    fig, ax = plt.subplots(figsize=(8, 6))
    extent = [p_vals[0], p_vals[-1], m_vals[0], m_vals[-1]]
    im = ax.imshow(Z, origin='lower', extent=extent, aspect='auto',
                   cmap='RdBu_r', vmin=-50, vmax=50)
    ax.contour(p_vals, m_vals, Z, levels=[0], colors='black', linewidths=2)

    # Overlay analytical threshold
    p_line = np.linspace(0.01, 0.5, 200)
    m_line = [analytical_threshold(p, theta) for p in p_line]
    ax.plot(p_line, m_line, 'k--', linewidth=1.5, label='Analytical threshold')

    plt.colorbar(im, ax=ax, label='Δ log-posterior (tie − untie)')
    ax.set_xlabel('Tie penalty p')
    ax.set_ylabel('Minority disagreement fraction m')
    ax.set_title(f'Tie preference heatmap (θ={theta}, n={n}, N={N})')
    ax.legend()
    ax.set_xlim(p_vals[0], p_vals[-1])
    ax.set_ylim(m_vals[0], m_vals[-1])

    plt.tight_layout()
    plt.savefig('tie_threshold_heatmap.png', dpi=150)
    plt.close()
    print("Saved: tie_threshold_heatmap.png")


def print_summary_table():
    """Print a table of key threshold values."""
    print("\n" + "=" * 72)
    print("  ANALYTICAL TIE THRESHOLDS")
    print("  m* = p + log(2/(1+exp(-θ))) / θ")
    print("  Ties preferred when minority fraction m > m*")
    print("=" * 72)

    thetas = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 50.0]
    ps = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

    # Header
    header = f"{'θ':>6} | {'max p':>6}"
    for p in ps:
        header += f" | m*(p={p})"
    print(header)
    print("-" * len(header))

    for theta in thetas:
        q = math.exp(-theta)
        dZ = math.log(2.0 / (1 + q))
        mp = max_p_for_ties(theta)
        row = f"{theta:6.1f} | {mp:6.3f}"
        for p in ps:
            m = analytical_threshold(p, theta)
            if m > 0.5:
                row += f" |   {'n/a':>5} "
            else:
                row += f" |   {m:5.3f} "
        print(row)

    print()
    print("KEY INSIGHTS:")
    print("  • As θ→∞, threshold m* → p  (p directly controls the boundary)")
    print("  • At finite θ, the Z* penalty adds an extra offset log(2/(1+q))/θ")
    print("  • For θ=1: ties possible only when p < 0.119")
    print("  • For θ=3: ties possible only when p < 0.285")
    print("  • For θ=10: ties possible only when p < 0.431")
    print("  • Interpretation: if >m* fraction of assessors disagree on a pair,")
    print("    the model prefers to tie them rather than commit to either order.")
    print()


if __name__ == "__main__":
    print_summary_table()
    plot_analytical_thresholds()
    plot_numerical_verification()
    plot_heatmap()
