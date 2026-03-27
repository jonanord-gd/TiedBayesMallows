"""Data structures for MCMC state and results."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ClusterParams:
    blocks: List[List[int]]
    theta: float
    gamma: float
    delta: float


@dataclass
class MixtureState:
    clusters: List[ClusterParams]
    z: List[int]
    tau: List[float]


@dataclass
class MCMCSamples:
    z_samples: List[List[int]]
    blocks_samples: List[List[List[List[int]]]]  # [t][c][block][item]
    tau_samples: Optional[List[List[float]]] = None
    theta_samples: Optional[List[List[float]]] = None
    logp: Optional[List[float]] = None          # length T
    K: Optional[List[List[int]]] = None         # [t][c] number of blocks per cluster
    # Acceptance indicators saved per stored iteration: [t][c] 0/1
    theta_accepts: Optional[List[List[int]]] = None
    block_accepts: Optional[List[List[int]]] = None
    # Detailed per-proposal counters (optional, may be large)
    theta_proposals: Optional[List[List[int]]] = None
    theta_accept_counts: Optional[List[List[int]]] = None
    block_proposals: Optional[List[List[int]]] = None
    block_accept_counts: Optional[List[List[int]]] = None


@dataclass
class SamplerConfig:
    n_item_moves_per_cluster: int = 2   # number of Gibbs item moves to attempt per cluster each iteration

    # PY prior parameters. Clusters have their own `gamma`/`delta` fields, but
    # setting these here overrides all clusters (None = use each cluster's value).
    gamma: Optional[float] = None   # Pitman-Yor discount parameter (0 <= delta < 1)
    delta: Optional[float] = None   # Pitman-Yor strength/concentration parameter

    # Metropolis-Hastings parameters for updating theta
    a_theta: float = 2.0            # shape of Gamma prior on theta
    b_theta: float = 1.0            # rate of Gamma prior on theta
    theta_step: float = 0.25        # log-normal proposal stddev for theta updates

    # Tie penalty weight: the p in the K^(p) extended Kendall distance.
    # p=0.5 gives the standard Kemeny distance. Must be in (0, 1].
    tie_penalty: float = 0.5