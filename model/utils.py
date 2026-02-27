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


def dirichlet_sample(alpha: List[float], rng: random.Random) -> List[float]:
    xs = [rng.gammavariate(a, 1.0) for a in alpha]
    s = sum(xs)
    return [x / s for x in xs]


def invert_perm(perm: List[int]) -> List[int]:
    inv = [0] * len(perm)
    for pos, item in enumerate(perm):
        inv[item] = pos
    return inv
