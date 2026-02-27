"""Prior and normalizing-constant formulas."""

import math, time
from functools import lru_cache
from typing import List, Optional, Tuple


def build_log_qfactorials(n: int, q: float) -> List[float]:
    """log([k]_q!) for k=0..n in O(n) for fixed q."""
    if n < 0:
        raise ValueError("n must be >= 0")
    out = [0.0] * (n + 1)
    if n == 0:
        return out
    if q <= 0.0:
        return out
    if abs(q - 1.0) < 1e-12:
        for k in range(1, n + 1):
            out[k] = math.lgamma(k + 1)
        return out

    log_denom = math.log(1.0 - q)
    qpow = q
    acc = 0.0
    for i in range(1, n + 1):
        acc += math.log(1.0 - qpow) - log_denom
        out[i] = acc
        qpow *= q
    return out


# internal cached computation when log_qfact is not provided
@lru_cache(maxsize=None)
def _log_Z_star_core(sizes_tuple: Tuple[int, ...], theta: float) -> float:
    # builds qfactorials internally; sizes_tuple is tuple of ints
    sizes = list(sizes_tuple)
    if theta <= 0:
        return float("-inf")
    n = sum(sizes)
    q = math.exp(-2.0 * theta)

    Tm = sum(s * (s - 1) // 2 for s in sizes)
    logP = sum(math.lgamma(s + 1) for s in sizes)

    log_qfact = build_log_qfactorials(n, q)
    return (-theta * Tm) + logP + (log_qfact[n] - sum(log_qfact[s] for s in sizes))


def log_Z_star_from_sizes(sizes: List[int], theta: float, log_qfact: Optional[List[float]] = None) -> float:
    # if caller didn't supply precomputed log_qfactorials, use cached core
    if log_qfact is None:
        return _log_Z_star_core(tuple(sizes), theta)

    if theta <= 0:
        return float("-inf")
    n = sum(sizes)
    q = math.exp(-2.0 * theta)

    Tm = sum(s * (s - 1) // 2 for s in sizes)
    logP = sum(math.lgamma(s + 1) for s in sizes)

    if len(log_qfact) < n + 1:
        log_qfact = build_log_qfactorials(n, q)

    return (-theta * Tm) + logP + (log_qfact[n] - sum(log_qfact[s] for s in sizes))


def log_Z_star(blocks: List[List[int]], theta: float) -> float:
    return log_Z_star_from_sizes([len(b) for b in blocks], theta, None)


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
) -> float:
    if not rankings_c:
        return float("-inf")
    sizes = [len(b) for b in blocks]
    K = len(sizes)
    from .distance import total_distance_fast
    from .profiling import get_profiler
    
    profiler = get_profiler()

    # Distance calculation
    if profiler:
        t_start = time.time()
    S = total_distance_fast(rankings_c, blocks)
    if profiler:
        profiler.record_operation("distance_calculation", time.time() - t_start)
    
    # Z* calculation
    if profiler:
        t_start = time.time()
    logZ = log_Z_star_from_sizes(sizes, theta, None)
    if profiler:
        profiler.record_operation("z_star_calculation", time.time() - t_start)
    
    # Pitman-Yor prior calculation
    if profiler:
        t_start = time.time()
    logpy = log_py_eppf_from_sizes(sizes, gamma, delta)
    if profiler:
        profiler.record_operation("py_prior_calculation", time.time() - t_start)
    
    return (-theta * S) - (len(rankings_c) * logZ) + logpy - math.lgamma(K + 1)

