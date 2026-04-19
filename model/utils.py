"""Small utility routines used throughout the model.

These were originally defined in the monolithic ``TiedMallowsObject.py`` file.
They live here now so that the core model implementation can remain lean and
only import what it actually uses.
"""

import math
import random
from typing import List


def logsumexp(logw: List[float]) -> float:
    m = max(logw)
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(sum(math.exp(x - m) for x in logw))


def sample_categorical_from_logweights(logw: List[float], rng: random.Random) -> int:
    lse = logsumexp(logw)
    probs = [math.exp(x - lse) for x in logw]
    u = rng.random()
    s = 0.0
    for k, p in enumerate(probs):
        s += p
        if u <= s:
            return k
    return len(probs) - 1


def normalize_simplex(values: List[float], min_value: float = 1e-300) -> List[float]:
    """Return a strictly positive, normalized probability vector."""
    if not values:
        return []

    floor = float(min_value)
    cleaned = [floor if (not math.isfinite(v) or v <= 0.0) else float(v) for v in values]
    s = sum(cleaned)

    if not math.isfinite(s) or s <= 0.0:
        return [1.0 / len(cleaned)] * len(cleaned)

    probs = [max(v / s, floor) for v in cleaned]
    s_probs = sum(probs)
    return [p / s_probs for p in probs]


def dirichlet_sample(alpha: List[float], rng: random.Random) -> List[float]:
    if any((not math.isfinite(a)) or a <= 0.0 for a in alpha):
        raise ValueError("Dirichlet parameters must be finite and strictly positive")

    xs = [rng.gammavariate(a, 1.0) for a in alpha]

    # With very small concentration parameters, gamma draws can underflow to
    # exact zeros in floating point.  Floor and renormalise so the returned
    # simplex is always strictly positive.
    if not any(x > 0.0 for x in xs):
        return normalize_simplex(alpha)

    return normalize_simplex(xs)


def invert_perm(perm: List[int]) -> List[int]:
    inv = [0] * len(perm)
    for pos, item in enumerate(perm):
        inv[item] = pos
    return inv
