"""
Estimation of the penalty parameter p for the K(p) distance
in the Weak-Mallows Model.

Two approaches:
1. Simple confidence-interval estimate (Section 4.1, eq. near bottom of p.6)
2. Data-informed Bayesian estimate using posterior tie probabilities,
   Bradley-Terry model for alpha, Pitman-Yor prior for pi_1,
   and misclassification loss minimisation (Section 4.1, eqs. 19-26).

Extended to multiple clusters with cluster-weighted averaging.

Author: Implementation based on Jonas Nordstrøm (Dec 2025) working notes.
"""

from __future__ import annotations

import numpy as np
from scipy.special import comb, gammaln
from scipy.optimize import minimize_scalar, minimize
from typing import Optional
import warnings


# ---------------------------------------------------------------------------
# 1. Simple confidence-interval estimate
# ---------------------------------------------------------------------------

def estimate_p_simple(N: int) -> float:
    """
    Quick estimate of p from the 95% CI of the binomial proportion
    under the null hypothesis that a pair is tied (r_ij ~ Binom(N, 0.5)).

    p = (sqrt(N) - 1.96) / (2 * sqrt(N))

    Parameters
    ----------
    N : int
        Number of assessors (observations).

    Returns
    -------
    float
        Estimated p in (0, 0.5). Clipped to (1e-6, 0.5 - 1e-6).
    """
    sqN = np.sqrt(N)
    p = (sqN - 1.96) / (2.0 * sqN)
    return float(np.clip(p, 1e-6, 0.5 - 1e-6))


def estimate_p_simple_multiclusters(
    N_total: int,
    C: int,
) -> float:
    """
    Simple CI estimate corrected for clustering: use the average cluster
    size N_avg = N_total / C in place of N.

    Parameters
    ----------
    N_total : int
        Total number of assessors.
    C : int
        Number of clusters.

    Returns
    -------
    float
        Estimated p.
    """
    N_avg = max(N_total / C, 2)
    return estimate_p_simple(int(round(N_avg)))


# ---------------------------------------------------------------------------
# 2. Data-informed Bayesian estimation
# ---------------------------------------------------------------------------

# 2a. Pairwise summary statistics -----------------------------------------

def pairwise_counts(rankings: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    From an (N x n) matrix of strict rankings, compute pairwise counts.

    Parameters
    ----------
    rankings : ndarray of shape (N, n)
        rankings[l, i] = rank of item i by assessor l (lower = better).
        Must be strict (no ties in observed rankings).

    Returns
    -------
    s : ndarray (n, n)   s[i,j] = # times item i ranked above j  (r(i) < r(j))
    t : ndarray (n, n)   t[i,j] = # ties (always 0 for strict rankings)
    N : int              number of assessors
    """
    N, n = rankings.shape
    s = np.zeros((n, n), dtype=int)
    for l in range(N):
        r = rankings[l]
        # i ranked above j means r[i] < r[j]
        for i in range(n):
            for j in range(n):
                if i != j and r[i] < r[j]:
                    s[i, j] += 1
    t = np.zeros((n, n), dtype=int)  # strict rankings => no ties
    return s, t, N


def pairwise_counts_fast(rankings: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Vectorised version of pairwise_counts."""
    N, n = rankings.shape
    # r[:, i, None] < r[:, None, j] => item i ranked above j
    above = rankings[:, :, None] < rankings[:, None, :]  # (N, n, n)
    s = above.sum(axis=0)  # (n, n)
    t = np.zeros((n, n), dtype=int)
    return s, t, N


# 2b. Bradley-Terry MLE for alpha -----------------------------------------

def bradley_terry_mle(
    s: np.ndarray,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Maximum-likelihood estimation of Bradley-Terry parameters lambda.

    Uses the iterative algorithm of Hunter (2004):
        lambda_i^{new} = W_i / sum_{j != i} (s_ij + s_ji) / (lambda_i + lambda_j)

    where W_i = sum_{j} s_ij.

    Parameters
    ----------
    s : ndarray (n, n)
        s[i,j] = number of times i is ranked above j.

    Returns
    -------
    lam : ndarray (n,)
        Estimated BT parameters (normalised to sum to n).
    """
    n = s.shape[0]
    lam = np.ones(n)
    W = s.sum(axis=1)                # W[i] = wins of item i
    n_ij_mat = s + s.T               # n_ij_mat[i,j] = total comparisons between i and j

    for _ in range(max_iter):
        lam_old = lam.copy()
        # denom[i] = sum_{j} n_ij_mat[i,j] / (lam[i] + lam[j])
        # diagonal is 0 (n_ij_mat[i,i] = 0) so no masking needed
        lam_sum = lam[:, None] + lam[None, :]  # (n, n)
        denom = (n_ij_mat / lam_sum).sum(axis=1)  # (n,)
        mask = denom > 0
        lam[mask] = W[mask] / denom[mask]
        # normalise
        lam *= n / lam.sum()
        if np.max(np.abs(lam - lam_old)) < tol:
            break
    return lam


def bt_alpha(lam: np.ndarray) -> np.ndarray:
    """
    Convert BT parameters to pairwise probabilities.
    alpha[i,j] = P(i > j | T_ij=0) = lambda_i / (lambda_i + lambda_j).
    """
    alpha = lam[:, None] / (lam[:, None] + lam[None, :])
    np.fill_diagonal(alpha, 0.5)
    return alpha


# 2c. Pitman-Yor prior for pi_1 -------------------------------------------

def sample_pitman_yor_partition(
    n: int,
    delta: float,
    lam: float,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """
    Sample one partition (block sizes) from a Pitman-Yor process.

    Parameters
    ----------
    n : int
        Number of items.
    delta : float
        Discount parameter, 0 <= delta <= 1.
    lam : float
        Strength parameter, lam > -delta.

    Returns
    -------
    block_sizes : list[int]
        Sizes of the blocks in the partition.
    """
    if rng is None:
        rng = np.random.default_rng()

    blocks: list[int] = []
    for i in range(1, n + 1):
        K = len(blocks)
        total = lam + i - 1
        probs = []
        # probability of joining each existing block
        for k in range(K):
            probs.append(max(blocks[k] - delta, 0.0))
        # probability of starting a new block
        probs.append(lam + K * delta)
        probs = np.array(probs)
        probs /= probs.sum()
        choice = rng.choice(len(probs), p=probs)
        if choice < K:
            blocks[choice] += 1
        else:
            blocks.append(1)
    return blocks


def estimate_pi1_pitman_yor(
    n: int,
    delta: float,
    lam: float,
    n_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Estimate the prior tie probability pi_1 by Monte-Carlo sampling
    partitions from a Pitman-Yor process:

        pi_1 = E[ sum_k C(|B_k|, 2) / C(n, 2) ]

    (Eq. 22 in the notes.)

    Parameters
    ----------
    n : int
        Number of items.
    delta, lam : float
        Pitman-Yor parameters.
    n_samples : int
        Number of MC samples.

    Returns
    -------
    pi1 : float
        Estimated prior probability that a pair is tied.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    total_pairs = comb(n, 2, exact=True)
    if total_pairs == 0:
        return 0.0
    pi1_sum = 0.0
    for _ in range(n_samples):
        blocks = sample_pitman_yor_partition(n, delta, lam, rng)
        tied_pairs = sum(comb(bk, 2, exact=True) for bk in blocks)
        pi1_sum += tied_pairs / total_pairs
    return pi1_sum / n_samples


# 2d. Posterior tie probability q_ij ---------------------------------------

def log_binom_pmf(k: int, N: int, p: float) -> float:
    """Log of Binomial(k; N, p) PMF, handling edge cases."""
    if p <= 0.0:
        return 0.0 if k == 0 else -np.inf
    if p >= 1.0:
        return 0.0 if k == N else -np.inf
    return gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1) + k * np.log(p) + (N - k) * np.log(1 - p)


def log_beta_binomial_pmf(k: int, N: int, a: float, b: float) -> float:
    """
    Log of the Beta-Binomial(k; N, a, b) PMF:
        C(N,k) * B(k+a, N-k+b) / B(a, b)
    where B is the Beta function.

    This is the marginal likelihood of observing k successes in N trials
    when the success probability is integrated out under a Beta(a, b) prior.
    """
    from scipy.special import betaln
    return (gammaln(N + 1) - gammaln(k + 1) - gammaln(N - k + 1)
            + betaln(k + a, N - k + b) - betaln(a, b))


def posterior_tie_probability(
    s_ij: int,
    N: int,
    alpha_ij: float,
    pi1: float,
    method: str = "marginal",
    a_alpha: float = 1.0,
    b_alpha: float = 1.0,
) -> float:
    """
    Compute q_ij = P(T_ij = 1 | s_ij, N).

    Two methods:
    - "point": Uses the BT point estimate for alpha (eq. 24 directly).
      This tends to underestimate tie probability because BT adapts
      alpha to the observed ratio, making the strict hypothesis
      always look good.
    - "marginal" (recommended): Integrates out alpha under a Beta(a, b)
      prior for the strict hypothesis. The tied hypothesis still uses
      Binom(N, 0.5). This gives:
        p(s_ij | T=1) = C(N, s_ij) * 2^{-N}
        p(s_ij | T=0) = BetaBinomial(s_ij; N, a_alpha, b_alpha)

    Parameters
    ----------
    s_ij : int
        Number of times i ranked above j.
    N : int
        Number of assessors.
    alpha_ij : float
        BT probability P(i > j | not tied). Used only if method="point".
    pi1 : float
        Prior probability the pair is tied.
    method : str
        "marginal" or "point".
    a_alpha, b_alpha : float
        Beta prior parameters on alpha for the strict hypothesis
        (only used if method="marginal").

    Returns
    -------
    float
        Posterior probability q_ij in [0, 1].
    """
    # Tied hypothesis: s_ij ~ Binom(N, 0.5)
    log_tied = np.log(pi1 + 1e-300) + log_binom_pmf(s_ij, N, 0.5)

    if method == "point":
        log_strict = np.log(1 - pi1 + 1e-300) + log_binom_pmf(s_ij, N, alpha_ij)
    elif method == "marginal":
        log_strict = (np.log(1 - pi1 + 1e-300)
                      + log_beta_binomial_pmf(s_ij, N, a_alpha, b_alpha))
    else:
        raise ValueError(f"Unknown method: {method}")

    log_denom = np.logaddexp(log_tied, log_strict)
    q = np.exp(log_tied - log_denom)
    return float(q)


def compute_all_qij(
    s: np.ndarray,
    N: int,
    alpha: np.ndarray,
    pi1: float,
    method: str = "marginal",
    a_alpha: float = 1.0,
    b_alpha: float = 1.0,
) -> np.ndarray:
    """Compute q_ij for all i < j pairs."""
    n = s.shape[0]
    q = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            q[i, j] = posterior_tie_probability(
                s[i, j], N, alpha[i, j], pi1,
                method=method, a_alpha=a_alpha, b_alpha=b_alpha,
            )
            q[j, i] = q[i, j]
    return q


# 2e. Minimise misclassification loss to find p-hat -----------------------

def empirical_tie_weight(s: np.ndarray, N: int) -> np.ndarray:
    """w_ij = min(r_ij, 1 - r_ij), with r_ij = s_ij / N."""
    n = s.shape[0]
    r = s / max(N, 1)
    w = np.minimum(r, 1.0 - r)
    return w


def misclassification_loss(
    p: float,
    q: np.ndarray,
    w: np.ndarray,
) -> float:
    """
    Smooth misclassification loss (eq. 26):
        L(p) = sum_{i<j} (1{p < w_ij} - q_ij)^2
    """
    i_idx, j_idx = np.triu_indices(q.shape[0], k=1)
    indicator = (p < w[i_idx, j_idx]).astype(float)
    return float(np.sum((indicator - q[i_idx, j_idx]) ** 2))


def _misclassification_loss_grid(
    p_grid: np.ndarray,
    q: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Vectorised evaluation of misclassification_loss over an entire grid.
    Returns an array of losses, one per p value.
    """
    i_idx, j_idx = np.triu_indices(q.shape[0], k=1)
    w_pairs = w[i_idx, j_idx]          # (n_pairs,)
    q_pairs = q[i_idx, j_idx]          # (n_pairs,)
    # indicators shape: (grid_size, n_pairs)
    indicators = (p_grid[:, None] < w_pairs[None, :]).astype(float)
    return np.sum((indicators - q_pairs[None, :]) ** 2, axis=1)


def estimate_p_data_informed(
    rankings: np.ndarray,
    pi1: float | None = None,
    delta: float = 0.5,
    lam: float = 1.0,
    n_mc_pi1: int = 5000,
    loss: str = "smooth",
    grid_size: int = 1000,
    method: str = "marginal",
    a_alpha: float = 1.0,
    b_alpha: float = 1.0,
) -> dict:
    """
    Data-informed estimation of p for a single cluster.

    Parameters
    ----------
    rankings : ndarray (N, n)
        Strict rankings matrix.
    pi1 : float or None
        Prior tie probability. If None, estimated via Pitman-Yor MC.
    delta, lam : float
        Pitman-Yor parameters (used only if pi1 is None).
    n_mc_pi1 : int
        Number of MC samples for pi1 estimation.
    loss : str
        "smooth" for eq. 26, "hard" for eq. 25.
    grid_size : int
        Number of grid points to search over (0, 0.5).
    method : str
        "marginal" (recommended) or "point" for posterior tie computation.
        "marginal" integrates out alpha under Beta(a_alpha, b_alpha).
        "point" uses the BT MLE for alpha (eq. 24 directly).
    a_alpha, b_alpha : float
        Beta prior parameters on alpha for marginal method.

    Returns
    -------
    dict with keys:
        p_hat        : estimated p
        pi1          : prior tie probability used
        q            : (n, n) posterior tie probabilities
        alpha        : (n, n) BT pairwise probabilities
        bt_lambda    : (n,) BT parameters
        s            : (n, n) pairwise counts
        w            : (n, n) empirical tie weights
    """
    N_obs, n = rankings.shape

    # Pairwise counts
    s, _, N = pairwise_counts_fast(rankings)

    # BT parameters (used for point method and diagnostics)
    bt_lam = bradley_terry_mle(s)
    alpha = bt_alpha(bt_lam)

    # Prior tie probability
    if pi1 is None:
        pi1 = estimate_pi1_pitman_yor(n, delta, lam, n_mc_pi1)

    # Posterior tie probabilities
    q = compute_all_qij(s, N, alpha, pi1, method=method,
                        a_alpha=a_alpha, b_alpha=b_alpha)

    # Empirical tie weights
    w = empirical_tie_weight(s, N)

    # Grid search for p_hat (fully vectorised over the grid)
    p_grid = np.linspace(1e-6, 0.5 - 1e-6, grid_size)
    losses = _misclassification_loss_grid(p_grid, q, w)
    best_idx = np.argmin(losses)
    p_hat = float(p_grid[best_idx])

    return {
        "p_hat": p_hat,
        "pi1": pi1,
        "q": q,
        "alpha": alpha,
        "bt_lambda": bt_lam,
        "s": s,
        "w": w,
        "loss_curve": (p_grid, losses),
    }


# ---------------------------------------------------------------------------
# 3. Multi-cluster estimation
# ---------------------------------------------------------------------------

def estimate_p_multicluster(
    rankings,
    cluster_assignments,
    delta: float = 0.5,
    lam: float = 1.0,
    n_mc_pi1: int = 5000,
    grid_size: int = 1000,
) -> dict:
    """
    Estimate p across multiple clusters. Each cluster gets its own
    p_hat, and the overall p is a weighted average (weighted by
    number of assessors in each cluster).

    Parameters
    ----------
    rankings : list of lists or ndarray (N, n)
        Strict rankings matrix (all assessors). Can be a list of rankings 
        as generated by generate_mixture_data (list of lists), or a numpy array.
    cluster_assignments : list or ndarray (N,)
        Cluster label for each assessor. Can be a list or numpy array.
    delta, lam : float
        Pitman-Yor parameters for pi1 estimation.
    n_mc_pi1 : int
        MC samples for pi1 estimation (per cluster).
    grid_size : int
        Grid resolution for p optimisation.

    Returns
    -------
    dict with keys:
        p_hat_global      : weighted average p
        p_hat_per_cluster  : dict {cluster_label: p_hat}
        weights            : dict {cluster_label: weight}
        details            : dict {cluster_label: full result dict}
    """
    # Convert to numpy arrays if needed (handles list format from generate_mixture_data)
    rankings = np.asarray(rankings)
    cluster_assignments = np.asarray(cluster_assignments)
    
    # Convert from position→item format to item→rank format
    # generate_mixture_data returns rankings as [item_at_pos_0, item_at_pos_1, ...]
    # but estimate_p_data_informed expects [rank_of_item_0, rank_of_item_1, ...]
    # argsort(x)[i] gives the position where x[i] would be sorted to (i.e., rank of position i)
    N = rankings.shape[0]
    n_items = rankings.shape[1]
    rankings_converted = np.zeros_like(rankings)
    for assessor_idx in range(N):
        rankings_converted[assessor_idx] = np.argsort(rankings[assessor_idx])
    rankings = rankings_converted
    
    clusters = np.unique(cluster_assignments)
    N_total = len(cluster_assignments)

    cluster_results = {}
    p_hats = {}
    weights = {}

    for c in clusters:
        mask = cluster_assignments == c
        R_c = rankings[mask]
        N_c = R_c.shape[0]

        if N_c < 2:
            warnings.warn(f"Cluster {c} has < 2 assessors; skipping.")
            continue

        # Each cluster may have a different tie structure, so we estimate
        # pi1 separately per cluster using Pitman-Yor
        result = estimate_p_data_informed(
            R_c,
            pi1=None,
            delta=delta,
            lam=lam,
            n_mc_pi1=n_mc_pi1,
            grid_size=grid_size,
        )
        cluster_results[c] = result
        p_hats[c] = result["p_hat"]
        weights[c] = N_c

    # Weighted average
    total_weight = sum(weights.values())
    if total_weight == 0:
        p_hat_global = 0.25  # fallback
    else:
        p_hat_global = sum(
            p_hats[c] * weights[c] for c in p_hats
        ) / total_weight

    return {
        "p_hat_global": p_hat_global,
        "p_hat_per_cluster": p_hats,
        "weights": {c: w / total_weight for c, w in weights.items()},
        "details": cluster_results,
    }


# ---------------------------------------------------------------------------
# 4. Simple CI estimate for multi-cluster (no data needed)
# ---------------------------------------------------------------------------

def estimate_p_simple_multicluster_weighted(
    cluster_sizes: list[int],
) -> dict:
    """
    Simple CI estimate per cluster, weighted-averaged.

    Parameters
    ----------
    cluster_sizes : list[int]
        Number of assessors in each cluster.

    Returns
    -------
    dict with p_hat_global and per-cluster values.
    """
    total = sum(cluster_sizes)
    p_hats = {}
    for i, N_c in enumerate(cluster_sizes):
        p_hats[i] = estimate_p_simple(max(N_c, 2))
    p_global = sum(p_hats[i] * cluster_sizes[i] for i in range(len(cluster_sizes))) / total
    return {
        "p_hat_global": p_global,
        "p_hat_per_cluster": p_hats,
    }


# ---------------------------------------------------------------------------
# 5. Demo / example usage
# ---------------------------------------------------------------------------

def _generate_rankings_from_weak_order(
    weak_order: list[list[int]],
    N: int,
    noise: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate N strict rankings from a weak-order consensus.

    Items in the same block are randomly shuffled; with probability
    `noise`, adjacent items across blocks are swapped.

    Parameters
    ----------
    weak_order : list[list[int]]
        Blocks ordered best-to-worst, e.g. [[0,1], [2], [3,4,5]]
        means items 0,1 are tied at rank 1, item 2 at rank 2, etc.
    N : int
        Number of rankings to generate.
    noise : float
        Probability of swapping adjacent items across blocks.

    Returns
    -------
    rankings : ndarray (N, n)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = sum(len(b) for b in weak_order)
    rankings = np.zeros((N, n), dtype=int)

    for l in range(N):
        # Build a strict ordering by shuffling within blocks
        order = []
        for block in weak_order:
            shuffled = list(block)
            rng.shuffle(shuffled)
            order.extend(shuffled)

        # Apply noise: swap adjacent items with some probability
        for idx in range(len(order) - 1):
            if rng.random() < noise:
                order[idx], order[idx + 1] = order[idx + 1], order[idx]

        # Convert item ordering to ranks
        rank = np.zeros(n, dtype=int)
        for pos, item in enumerate(order):
            rank[item] = pos + 1
        rankings[l] = rank

    return rankings

# ---------------------------------------------------------------------------
# 5. Label switching and cluster alignment
# ---------------------------------------------------------------------------

def match_clusters(z_true, z_pred):
    """
    Align predicted cluster labels with true cluster labels using the
    Hungarian algorithm to maximize the number of correct assignments.
    
    Parameters
    ----------
    z_true : array-like
        True cluster assignments.
    z_pred : array-like
        Predicted cluster assignments.
    
    Returns
    -------
    tuple
        (pred_to_true, confusion_matrix, z_pred_aligned, accuracy)
        - pred_to_true: dict mapping predicted labels to true labels
        - confusion_matrix: (T, P) matrix of assignment counts
        - z_pred_aligned: predicted labels after alignment
        - accuracy: fraction of correctly assigned assessors
    """
    from scipy.optimize import linear_sum_assignment
    
    z_true = np.asarray(z_true)
    z_pred = np.asarray(z_pred)

    true_labels = np.unique(z_true)
    pred_labels = np.unique(z_pred)

    # map original labels to 0..T-1 and 0..P-1
    t2i = {t:i for i,t in enumerate(true_labels)}
    p2j = {p:j for j,p in enumerate(pred_labels)}
    T, P = len(true_labels), len(pred_labels)

    # contingency / confusion counts
    M = np.zeros((T, P), dtype=int)
    for t, p in zip(z_true, z_pred):
        M[t2i[t], p2j[p]] += 1

    # maximize matches -> minimize negative matches
    row_ind, col_ind = linear_sum_assignment(-M)

    # build mapping from predicted label -> matched true label
    pred_to_true = {pred_labels[j]: true_labels[i] for i, j in zip(row_ind, col_ind)}

    # relabel predictions (unmatched preds, if any, stay as-is or set to -1)
    z_pred_aligned = np.array([pred_to_true.get(p, -1) for p in z_pred])

    accuracy = (z_pred_aligned == z_true).mean()

    return pred_to_true, M, z_pred_aligned, accuracy
