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
    n_item_moves_per_cluster: int = 2   # number of item-level PY moves to attempt per cluster each iteration

    # PY prior parameters used by the various block moves.  Clusters have
    # their own `gamma`/`delta` fields, but setting these values here overrides
    # them for every cluster (with ``None`` meaning "use the cluster value").
    gamma: Optional[float] = None   # Pitman–Yor discount parameter (0<=delta<1)
    delta: Optional[float] = None   # Pitman–Yor strength/concentration parameter

    # mixture weights for the five block-update types; they are normalized
    # internally so only their ratios matter.
    p_gibbs_reassign: float = 0.0   # probability of Gibbs reassign-one-item move
    p_transfer: float = 0.4         # probability of adjacent-item-transfer move
    p_swapshift: float = 0.4        # probability of ordering swap/shift move
    p_splitmerge: float = 0.2       # probability of adjacent split/merge move
    p_reassign: float = 0.0         # probability of PY-prior MH single-item reassign move

    # parameters specific to the ordering swap/shift move
    ordering_p_short: float = 0.75   # chance of proposing a "short" swap vs long shift
    ordering_n_swap_steps: Optional[int] = None  # max number of random swap steps when using short move
    ordering_max_long_step: Optional[int] = None  # max displacement for long shift move

    # parameter for split/merge move probability of proposing merge given a split choice
    splitmerge_p_merge: float = 0.5

    # Metropolis–Hastings parameters for updating theta
    a_theta: float = 2.0            # shape of Gamma prior on theta
    b_theta: float = 1.0            # rate of Gamma prior on theta
    theta_step: float = 0.25        # log-normal proposal stddev for theta updates
