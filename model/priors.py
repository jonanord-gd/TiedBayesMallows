"""Prior and normalizing-constant formulas."""

import math, time
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np
try:
    from scipy.special import gammaln as _vec_lgamma
except ImportError:
    _vec_lgamma = np.vectorize(math.lgamma)


def build_log_qfactorials(n: int, q: float) -> np.ndarray:
    """log([k]_q!) for k=0..n, returned as a numpy array of shape (n+1,)."""
    if n < 0:
        raise ValueError("n must be >= 0")
    out = np.zeros(n + 1)
    if n == 0 or q <= 0.0:
        return out
    if abs(q - 1.0) < 1e-12:
        out[1:] = np.cumsum(np.log(np.arange(1, n + 1, dtype=np.float64)))
        return out
    log_denom = math.log(1.0 - q)
    qpows = np.power(q, np.arange(1, n + 1, dtype=np.float64))
    out[1:] = np.cumsum(np.log1p(-qpows) - log_denom)
    return out


# internal cached computation when log_qfact is not provided
def _log_Z_star_core(sizes_tuple: Tuple[int, ...], theta: float) -> float:
    # builds qfactorials internally; sizes_tuple is tuple of ints
    if theta <= 0:
        return float("-inf")
    n = sum(sizes_tuple)
    q = math.exp(-1.0 * theta)
    sizes_arr = np.asarray(sizes_tuple, dtype=np.intp)
    logP = float(_vec_lgamma(sizes_arr + 1).sum())
    log_qfact = build_log_qfactorials(n, q)
    return logP + (float(log_qfact[n]) - float(log_qfact[sizes_arr].sum()))


def log_Z_star_from_sizes(sizes: List[int], theta: float, log_qfact=None) -> float:
    # if caller didn't supply precomputed log_qfactorials, use core
    if log_qfact is None:
        return _log_Z_star_core(tuple(sizes), theta)

    if theta <= 0:
        return float("-inf")
    n = sum(sizes)
    q = math.exp(-1.0 * theta)
    sizes_arr = np.asarray(sizes, dtype=np.intp)
    logP = float(_vec_lgamma(sizes_arr + 1).sum())
    log_qfact_arr = np.asarray(log_qfact)
    if len(log_qfact_arr) < n + 1:
        log_qfact_arr = build_log_qfactorials(n, q)
    return logP + (float(log_qfact_arr[n]) - float(log_qfact_arr[sizes_arr].sum()))


def log_Z_star(blocks: List[List[int]], theta: float) -> float:
    return log_Z_star_from_sizes([len(b) for b in blocks], theta)


def log_py_eppf_from_sizes(sizes: List[int], gamma: float, delta: float) -> float:
    """Delta is the discount, gamma the concentration. See Pitman-Yor EPPF formula."""
    if gamma <= -delta:
        return float("-inf")
    if not (0.0 <= delta < 1.0):
        return float("-inf")
    n = sum(sizes)
    K = len(sizes)
    if n == 0 or K == 0:
        return 0.0

    log_num_tables = 0.0
    for i in range(1, K):
        term = gamma + i * delta
        if term <= 0:
            return float("-inf")
        log_num_tables += math.log(term)

    log_num_sizes = 0.0
    log_gamma_1md = math.lgamma(1.0 - delta)
    for s in sizes:
        if s <= 0:
            return float("-inf")
        log_num_sizes += math.lgamma(s - delta) - log_gamma_1md

    log_denom = math.lgamma(gamma + n) - math.lgamma(gamma + 1.0)
    return log_num_tables + log_num_sizes - log_denom


def log_blocks_posterior(
    rankings_c: List[List[int]],
    blocks: List[List[int]],
    theta: float,
    gamma: float,
    delta: float,
    blocks_old: Optional[List[List[int]]] = None,
    distance_calculator=None,
    parallel: bool = False,
    use_py_prior: bool = True,
    include_order_prior: bool = True,
) -> float:
    """Compute log posterior of blocks given cluster rankings.

    The ``blocks_old``, ``distance_calculator``, and ``parallel`` parameters
    are accepted for backward compatibility with legacy code but are ignored;
    the full distance is always computed via ``total_distance_fast``.

    Note: the tie penalty p cancels analytically between the distance
    d(r, rho_B) and the normalisation constant Z*, so it does not appear
    in this function.
    """
    if not rankings_c:
        return float("-inf")
    sizes = [len(b) for b in blocks]
    K = len(sizes)
    from .distance import total_distance_fast
    from .profiling import get_profiler

    profiler = get_profiler()

    if profiler:
        t_start = time.time()
    S = total_distance_fast(rankings_c, blocks)
    if profiler:
        profiler.record_operation("distance_calculation", time.time() - t_start)

    if profiler:
        t_start = time.time()
    logZ = log_Z_star_from_sizes(sizes, theta)
    if profiler:
        profiler.record_operation("z_star_calculation", time.time() - t_start)

    if profiler:
        t_start = time.time()
    if use_py_prior:
        logpy = log_py_eppf_from_sizes(sizes, gamma, delta)
    else:
        logpy = 0.0
    if profiler:
        profiler.record_operation("py_prior_calculation", time.time() - t_start)

    log_ord = math.lgamma(K + 1) if include_order_prior else 0.0
    return (-theta * S) - (len(rankings_c) * logZ) + logpy - log_ord

