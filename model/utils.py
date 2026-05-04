"""Small utility routines used throughout the model.

These were originally defined in the monolithic ``TiedMallowsObject.py`` file.
They live here now so that the core model implementation can remain lean and
only import what it actually uses.
"""

import math
import random
from typing import List

import numpy as np


def logsumexp(logw) -> float:
    arr = np.asarray(logw, dtype=np.float64)
    m = float(arr.max())
    if m == float("-inf"):
        return float("-inf")
    return m + math.log(float(np.exp(arr - m).sum()))


def sample_categorical_from_logweights(logw, rng: random.Random) -> int:
    arr = np.asarray(logw, dtype=np.float64)
    m = arr.max()
    shifted = np.exp(arr - m)
    cumprobs = np.cumsum(shifted)
    u = rng.random() * float(cumprobs[-1])
    idx = int(np.searchsorted(cumprobs, u))
    return min(idx, len(arr) - 1)


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
