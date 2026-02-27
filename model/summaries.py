"""Helpers for summarizing MCMC output and blocks."""

import math
from typing import Any, Dict, List, Tuple


def _canonicalize_blocks(blocks: List[List[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(sorted(block)) for block in blocks)


def estimate_z_from_frequency(z_samples: List[List[int]], *, C: int) -> Dict[str, Any]:
    T = len(z_samples)
    N = len(z_samples[0])
    counts = [[0] * C for _ in range(N)]
    for z in z_samples:
        for i, c in enumerate(z):
            counts[i][c] += 1
    p_ic = [[counts[i][c] / T for c in range(C)] for i in range(N)]
    z_hat = [max(range(C), key=lambda c: p_ic[i][c]) for i in range(N)]
    return {"p_ic": p_ic, "z_hat": z_hat}


def _posterior_mode_from_counts(counts: Dict[Any, int]) -> Tuple[Any, float, int]:
    total = sum(counts.values())
    mode_val = max(counts.keys(), key=lambda k: counts[k])
    mode_count = counts[mode_val]
    mode_prob = mode_count / total if total else float("nan")
    return mode_val, mode_prob, mode_count


def summarize_theta(theta_samples_c: List[float], *, ci: float = 0.95, map_bins: int = 50) -> Dict[str, float]:
    xs = sorted(theta_samples_c)
    n = len(xs)
    mean = sum(xs) / n
    median = xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    a = (1 - ci) / 2
    lo = xs[int(math.floor(a * (n - 1)))]
    hi = xs[int(math.floor((1 - a) * (n - 1)))]

    x_min, x_max = xs[0], xs[-1]
    if x_max == x_min:
        map_est = x_min
    else:
        binw = (x_max - x_min) / map_bins
        bins = [0] * map_bins
        for x in xs:
            k = min(map_bins - 1, int((x - x_min) / binw))
            bins[k] += 1
        kmax = max(range(map_bins), key=lambda k: bins[k])
        map_est = x_min + (kmax + 0.5) * binw
    return {"mean": mean, "median": median, "ci_lo": lo, "ci_hi": hi, "map": map_est}
