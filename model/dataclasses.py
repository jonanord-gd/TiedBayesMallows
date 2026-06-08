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
    saved_iterations: Optional[List[int]] = None  # actual MCMC iteration index for each stored draw
    theta_jump: int = 1
    # Acceptance indicators saved per stored iteration: [t][c] 0/1
    theta_accepts: Optional[List[List[int]]] = None
    block_accepts: Optional[List[List[int]]] = None
    # Detailed per-proposal counters (optional, may be large)
    theta_proposals: Optional[List[List[int]]] = None
    theta_accept_counts: Optional[List[List[int]]] = None
    block_proposals: Optional[List[List[int]]] = None
    block_accept_counts: Optional[List[List[int]]] = None
    # Per-observation marginal log-likelihoods for arviz WAIC/LOO.
    # Shape: [T][N] — one list of N floats per saved draw.
    # log p(r_i | tau, blocks, theta) = logsumexp_c[log(tau_c) - theta_c*D_ic - logZ_c]
    log_likelihood: Optional[List[List[float]]] = None


@dataclass
class SamplerConfig:
    n_item_moves_per_cluster: int = 2   # number of Gibbs item moves to attempt per cluster each iteration

    # PY prior parameters. Clusters have their own `gamma`/`delta` fields, but
    # setting these here overrides all clusters (None = use each cluster's value).
    gamma: Optional[float] = None   # Pitman-Yor discount parameter (0 <= delta < 1)
    delta: Optional[float] = None   # Pitman-Yor strength/concentration parameter

    # Whether to use Pitman-Yor prior on block structure.
    # When False, a flat (non-informative) prior on partitions is used instead.
    use_py_prior: bool = True

    # Metropolis-Hastings parameters for updating theta
    a_theta: float = 2.0            # shape of Gamma prior on theta
    b_theta: float = 1.0            # rate of Gamma prior on theta
    theta_step: float = 0.25        # log-normal proposal stddev for theta updates
    adapt_theta_step: bool = True   # adapt theta_step during burn-in to target acceptance rate
    target_theta_acceptance: float = 0.234  # target acceptance rate for theta proposals

    # Whether to include a uniform prior on block orderings: 1/K!
    # When True, more blocks are penalised factorially, favouring ties.
    # When False, only the likelihood and (optionally) the PY prior determine K.
    include_order_prior: bool = True