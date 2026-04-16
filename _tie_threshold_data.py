"""
Analytical tie threshold: when does data prefer a tie over a separation?

With p cancelled, the log-likelihood for a single cluster is:
    L(B) = -θ·S(B) - N·log Z*(B)

where S(B) = total cross-block disagreements, Z* = P(s)·[n]_q!/∏[s_k]_q!

Consider the simplest case: should items a and b be tied or separated?

SEPARATED (a before b in consensus, singletons):
    Each assessor who ranks b before a contributes 1 disagreement.
    Let m = min(h_ab, N - h_ab) where h_ab = #{assessors: a before b}.
    The optimal ordering has m disagreements for this pair.

TIED (a,b in same block):
    No cross-block disagreement for pair (a,b).
    But Z* increases: Z*_tie / Z*_sep = 2/[2]_q = 2/(1+q), where q = exp(-θ).

The change in log-likelihood when merging a and b:
    ΔlogL = θ·m - N·log(2/(1+q))

Tie preferred when ΔlogL > 0, i.e. when minority fraction f = m/N exceeds:

    ┌─────────────────────────────────────────┐
    │  f* = log(2 / (1 + exp(-θ))) / θ       │
    └─────────────────────────────────────────┘

Properties:
    • f*(θ→0) = 1/2   (ties only when exact 50-50 disagreement)
    • f*(θ→∞) = 0     (any disagreement at all leads to ties)
    • f* depends ONLY on θ — not on γ, δ, p, or n
    • This is the per-pair threshold; when f > f* for a pair, the model
      gains likelihood by tying them.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from model.priors import log_Z_star_from_sizes, build_log_qfactorials


# ═══════════════════════════════════════════════════════════════════════════════
# 1. The analytical threshold
# ═══════════════════════════════════════════════════════════════════════════════

def f_star(theta):
    """Critical minority fraction above which a tie is preferred."""
    q = math.exp(-theta)
    return math.log(2.0 / (1.0 + q)) / theta


def f_star_limit_theta_zero():
    """Limit as θ→0: f* → 1/2  (by L'Hôpital)."""
    return 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Numerical verification: compare log-posteriors directly
# ═══════════════════════════════════════════════════════════════════════════════

def verify_threshold(n=6, N=30, theta=2.0):
    """For a simple n-item problem, verify the threshold numerically."""
    # All singletons: sizes = [1]*n
    sizes_sep = [1] * n
    logZ_sep = log_Z_star_from_sizes(sizes_sep, theta)

    # One tie (merge items 0 and 1): sizes = [2, 1, ..., 1]
    sizes_tie = [2] + [1] * (n - 2)
    logZ_tie = log_Z_star_from_sizes(sizes_tie, theta)

    delta_logZ = logZ_tie - logZ_sep  # Z* cost of tying

    # The pair (0,1): m disagreements out of N assessors
    # ΔlogL = θ·m - N·δlogZ
    # Tie preferred when m > N·δlogZ/θ

    threshold_m = N * delta_logZ / theta
    threshold_f = delta_logZ / theta

    # Compare with analytical formula
    analytical_f = f_star(theta)

    return delta_logZ, threshold_m, threshold_f, analytical_f


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Multi-pair extension: when does the FIRST tie appear?
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pairwise_minority_fractions(rankings, n):
    """For each pair (a,b), compute the minority fraction.

    Returns an (n, n) matrix where entry [a,b] = fraction of assessors
    who rank a after b (disagreement if a is before b in consensus).
    """
    N = len(rankings)
    # Build position matrix: pos[i][item] = position of item in ranking i
    h = np.zeros((n, n), dtype=int)
    for r in rankings:
        pos = [0] * n
        for p, item in enumerate(r):
            pos[item] = p
        for a in range(n):
            for b in range(a + 1, n):
                if pos[a] < pos[b]:  # a before b
                    h[a, b] += 1
                else:
                    h[b, a] += 1

    # Minority fraction: min(h[a,b], h[b,a]) / N
    frac = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            m = min(h[a, b], h[b, a])
            frac[a, b] = frac[b, a] = m / N

    return frac


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_threshold_curve():
    """Plot 1: The analytical threshold f* vs θ."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel (a): f* vs θ ---
    ax = axes[0]
    thetas = np.linspace(0.01, 15, 500)
    fs = [f_star(t) for t in thetas]
    ax.plot(thetas, fs, "C0", linewidth=2.5)
    ax.fill_between(thetas, fs, 0.5, alpha=0.15, color="C0", label="Tie preferred")
    ax.fill_between(thetas, 0, fs, alpha=0.15, color="C3", label="Separation preferred")
    ax.set_xlabel(r"$\theta$ (concentration)", fontsize=12)
    ax.set_ylabel(r"Minority fraction $f = m/N$", fontsize=12)
    ax.set_title(r"(a) Tie threshold: $f^* = \frac{\log(2/(1+e^{-\theta}))}{\theta}$", fontsize=12)
    ax.set_ylim(0, 0.55)
    ax.set_xlim(0, 15)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    ax.legend(fontsize=10, loc="center right")

    # Annotate a few points
    for theta_val in [0.5, 1.0, 2.0, 5.0, 10.0]:
        fv = f_star(theta_val)
        ax.plot(theta_val, fv, "ko", markersize=4)
        ax.annotate(f"({theta_val}, {fv:.2f})",
                    (theta_val, fv), textcoords="offset points",
                    xytext=(8, 5), fontsize=8)

    # --- Panel (b): Interpretation with concrete numbers ---
    ax = axes[1]
    N_vals = [10, 30, 100, 500]
    thetas2 = np.linspace(0.1, 10, 200)
    for N_val, color in zip(N_vals, ["C0", "C1", "C2", "C4"]):
        m_stars = [f_star(t) * N_val for t in thetas2]
        ax.plot(thetas2, m_stars, color=color, linewidth=1.5, label=f"N={N_val}")
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel(r"$m^*$ (minimum disagreeing assessors)", fontsize=12)
    ax.set_title("(b) Absolute threshold count", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 10)

    # --- Panel (c): Verification against numerical log-posterior ---
    ax = axes[2]
    n_test = 6
    N_test = 30
    thetas3 = np.linspace(0.1, 8, 80)

    # For each theta, compute the actual ΔlogL for m = 0..N/2
    # and find the crossover
    m_range = np.arange(0, N_test // 2 + 1)
    crossover_m = []
    for theta in thetas3:
        q = math.exp(-theta)
        delta_logZ = math.log(2.0 / (1.0 + q))
        # ΔlogL = θ·m - N·δlogZ; tie preferred when m > N·δlogZ/θ
        m_crit = N_test * delta_logZ / theta
        crossover_m.append(m_crit)

    ax.plot(thetas3, crossover_m, "C0-", linewidth=2, label="Numerical (exact)")
    ax.plot(thetas3, [f_star(t) * N_test for t in thetas3], "k--",
            linewidth=1.5, label="Analytical formula")
    ax.fill_between(thetas3, crossover_m, N_test / 2,
                    alpha=0.12, color="C0", label="Tie preferred")
    ax.fill_between(thetas3, 0, crossover_m,
                    alpha=0.12, color="C3", label="Separation preferred")
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel(r"$m^*$ (min dissenters for a tie)", fontsize=12)
    ax.set_title(f"(c) Verification (n={n_test}, N={N_test})", fontsize=12)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, N_test / 2 + 1)

    fig.suptitle(
        "When does data prefer a tie?  "
        r"Tie when minority fraction $f > f^*(\theta)$",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("tie_threshold_data.png", dpi=150, bbox_inches="tight")
    print("Saved: tie_threshold_data.png")
    plt.close(fig)


def plot_multi_pair_analysis():
    """Plot 2: For synthetic data, show how many pairs cross the threshold."""
    import random

    random.seed(42)
    n = 10
    N = 50

    # Generate rankings with some structure (partial agreement)
    # True ranking: 0,1,2,...,n-1 with noise
    rankings = []
    for _ in range(N):
        r = list(range(n))
        # Random adjacent swaps to create noise
        for _ in range(n):
            i = random.randint(0, n - 2)
            r[i], r[i + 1] = r[i + 1], r[i]
        rankings.append(r)

    frac = compute_pairwise_minority_fractions(rankings, n)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel (a): Heatmap of minority fractions ---
    ax = axes[0]
    mask = np.triu(np.ones_like(frac, dtype=bool), k=0)
    frac_display = np.ma.masked_where(mask, frac)
    im = ax.imshow(frac_display, cmap="YlOrRd", vmin=0, vmax=0.5,
                   origin="upper")
    ax.set_xlabel("Item")
    ax.set_ylabel("Item")
    ax.set_title("(a) Pairwise minority fractions", fontsize=12)
    fig.colorbar(im, ax=ax, label="f = m/N", shrink=0.8)

    # --- Panel (b): Number of pairs that cross threshold vs θ ---
    ax = axes[1]
    n_pairs = n * (n - 1) // 2
    thetas = np.linspace(0.1, 10, 200)

    # Upper triangle entries
    upper_fracs = []
    for a in range(n):
        for b in range(a + 1, n):
            upper_fracs.append(frac[a, b])
    upper_fracs = np.array(upper_fracs)

    n_ties_preferred = []
    for theta in thetas:
        fs = f_star(theta)
        n_ties_preferred.append(np.sum(upper_fracs > fs))

    ax.plot(thetas, n_ties_preferred, "C0-", linewidth=2)
    ax.axhline(n_pairs, color="grey", linestyle=":", alpha=0.5,
               label=f"Total pairs = {n_pairs}")
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel("# pairs where tie is preferred", fontsize=12)
    ax.set_title("(b) How many pairs want to tie?", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(-1, n_pairs + 2)

    # --- Panel (c): Distribution of minority fractions + threshold ---
    ax = axes[2]
    ax.hist(upper_fracs, bins=20, density=True, alpha=0.6, color="C0",
            edgecolor="black", linewidth=0.5)
    for theta, color, ls in [(1.0, "C1", "--"), (2.0, "C2", "-"),
                              (5.0, "C4", "-.")]:
        fs = f_star(theta)
        ax.axvline(fs, color=color, linestyle=ls, linewidth=2,
                   label=r"$f^*(\theta=%g) = %.3f$" % (theta, fs))
    ax.set_xlabel("Minority fraction f", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("(c) Distribution of pairwise disagreement", fontsize=12)
    ax.legend(fontsize=9)

    fig.suptitle(
        f"Multi-pair tie analysis (n={n}, N={N}, synthetic noisy data)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("tie_threshold_pairs.png", dpi=150, bbox_inches="tight")
    print("Saved: tie_threshold_pairs.png")
    plt.close(fig)


def plot_larger_n_verification():
    """Plot 3: Verify for larger n that the pairwise threshold is accurate
    by comparing log-posteriors of singleton vs single-merge candidates."""
    import random

    random.seed(42)
    n = 8
    N = 40

    # Generate data with moderate noise
    rankings = []
    for _ in range(N):
        r = list(range(n))
        for _ in range(n):
            i = random.randint(0, n - 2)
            r[i], r[i + 1] = r[i + 1], r[i]
        rankings.append(r)

    frac = compute_pairwise_minority_fractions(rankings, n)
    thetas = [0.5, 1.0, 2.0, 4.0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, theta in enumerate(thetas):
        ax = axes[idx // 2][idx % 2]

        # For each pair, compute actual ΔlogL (singleton → merge that pair)
        sizes_sing = [1] * n
        logZ_sing = log_Z_star_from_sizes(sizes_sing, theta)

        delta_logLs = []
        minority_fracs = []

        for a in range(n):
            for b in range(a + 1, n):
                f = frac[a, b]
                m = f * N

                # Z* for merging a and b
                sizes_merge = [2] + [1] * (n - 2)
                logZ_merge = log_Z_star_from_sizes(sizes_merge, theta)

                # ΔlogL = θ·m - N·(logZ_merge - logZ_sing)
                delta = theta * m - N * (logZ_merge - logZ_sing)

                delta_logLs.append(delta)
                minority_fracs.append(f)

        ax.scatter(minority_fracs, delta_logLs, c="C0", alpha=0.7, s=40,
                   edgecolors="black", linewidth=0.3)
        ax.axhline(0, color="red", linewidth=1.5, linestyle="-", alpha=0.7)

        # Analytical threshold
        fs = f_star(theta)
        ax.axvline(fs, color="C2", linewidth=2, linestyle="--",
                   label=r"$f^* = %.3f$" % fs)

        ax.set_xlabel("Minority fraction f", fontsize=11)
        ax.set_ylabel(r"$\Delta\log L$ (tie − separate)", fontsize=11)
        ax.set_title(r"$\theta = %g$" % theta, fontsize=12)
        ax.legend(fontsize=10)

    fig.suptitle(
        r"Verification: $\Delta\log L > 0 \Leftrightarrow f > f^*$"
        f"  (n={n}, N={N})",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig("tie_threshold_verification_v2.png", dpi=150, bbox_inches="tight")
    print("Saved: tie_threshold_verification_v2.png")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Summary
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary():
    print()
    print("=" * 70)
    print("  DATA-DRIVEN TIE THRESHOLD  (no PY prior, no order prior)")
    print("=" * 70)
    print()
    print("  For a pair (a, b), let m = minority count = min(h_ab, N - h_ab)")
    print("  where h_ab = #{assessors who rank a before b}.")
    print()
    print("  The likelihood prefers TYING a and b when:")
    print()
    print("       m/N  >  f*(θ)  =  log(2 / (1 + e^{-θ})) / θ")
    print()
    print("  This threshold depends ONLY on θ:")
    print()
    print(f"    {'θ':>6}  {'f*':>8}  {'Interpretation':>50}")
    print("    " + "-" * 66)
    for theta in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        fs = f_star(theta)
        pct = fs * 100
        print(f"    {theta:>6.1f}  {fs:>8.3f}  "
              f"need >{pct:.0f}% disagreement to prefer a tie")
    print()
    print("  Intuition:")
    print("    • θ is the 'sharpness' of the Mallows distribution.")
    print("    • Higher θ → model is more sensitive to disagreements →")
    print("      even small minority fractions cause ties.")
    print("    • Lower θ → model tolerates disagreement → ties only")
    print("      when assessors are nearly 50-50 on a pair.")
    print("    • f* = 0.5 means 'never tie' (impossible to exceed 0.5)")
    print("    • f* = 0   means 'always tie' (any disagreement suffices)")
    print()
    print("  The Z* cost of a tie:")
    print("    Merging a pair increases Z* by a factor of 2/(1+q),")
    print("    where q = exp(-θ).  This is the 'price' of tying;")
    print("    the data must 'pay' it via reduced disagreements.")
    print()
    print("  Verification:")
    for n_test, N_test, theta in [(6, 30, 2.0), (10, 50, 1.0), (20, 100, 3.0)]:
        delta_logZ, m_thresh, f_thresh, f_analytical = verify_threshold(
            n_test, N_test, theta
        )
        print(f"    n={n_test:>3d}, N={N_test:>3d}, θ={theta:.1f}: "
              f"numerical f*={f_thresh:.6f}, analytical f*={f_analytical:.6f}, "
              f"match={'YES' if abs(f_thresh - f_analytical) < 1e-10 else 'NO'}")
    print()


if __name__ == "__main__":
    print_summary()
    print("Generating plots...")
    plot_threshold_curve()
    plot_multi_pair_analysis()
    plot_larger_n_verification()
    print("\nDone.")
