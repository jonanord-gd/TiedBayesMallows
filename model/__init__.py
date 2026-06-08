# Package initializer for the model subpackage.
# Re-export key classes and functions for convenience.

from .TiedMallowsModel import (
    MixtureRankingModel,
    ClusterParams,
    MixtureState,
    MCMCSamples,
    SamplerConfig,
    estimate_z_from_frequency,
    summarize_theta,
    init_spectral_with_z,
    init_clusters_default,
    init_blocks_spectral,
    fast_gibbs_reassign_one_item,
    icm_sweep_cluster,
    logsumexp,
    sample_categorical_from_logweights,
    dirichlet_sample,
    enable_profiling,
    disable_profiling,
    get_profiler,
    reset_profiling,
    MCMCProfiler,
    acceptance_probabilities,
    print_acceptance_summary,
)
from .augmentation import detect_missing, PartialRankingInfo
from .identifiability import (
    compute_dmin_plugin,
    compute_dmin_posterior_averaged,
    log_one_over_beta_pair,
)
from .summaries import (
    greedy_consensus_recovery,
    consensus_from_samples,
    medoid_from_samples,
    build_posterior_pairwise_matrix,
    merge_threshold,
)
from .theta_conditional_map import (
    theta_conditional_map_for_cluster,
    solve_theta_conditional_map,
)

__all__ = [
    "MixtureRankingModel",
    "ClusterParams",
    "MixtureState",
    "MCMCSamples",
    "SamplerConfig",
    "estimate_z_from_frequency",
    "summarize_theta",
    "init_spectral_with_z",
    "init_clusters_default",
    "init_blocks_spectral",
    "fast_gibbs_reassign_one_item",
    "icm_sweep_cluster",
    "logsumexp",
    "sample_categorical_from_logweights",
    "dirichlet_sample",
    "enable_profiling",
    "disable_profiling",
    "get_profiler",
    "reset_profiling",
    "MCMCProfiler",
    "acceptance_probabilities",
    "print_acceptance_summary",
    "detect_missing",
    "PartialRankingInfo",
    "greedy_consensus_recovery",
    "consensus_from_samples",
    "medoid_from_samples",
    "build_posterior_pairwise_matrix",
    "merge_threshold",
    "theta_conditional_map_for_cluster",
    "solve_theta_conditional_map",
]
