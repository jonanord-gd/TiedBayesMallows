"""
TiedMallows: Bayesian mixture of tied-ranking models via MCMC.

The main model class has been reorganized into submodules for better structure.
This file provides backwards-compatible re-exports.

Main API:
  - MixtureRankingModel: The primary model class
  - ClusterParams, MixtureState, MCMCSamples, SamplerConfig: Data structures
  - Utility functions for summarization, initialization, and moves
  - Profiling utilities for performance analysis

For the full documentation, import from model or see model.core.
"""

from .core import MixtureRankingModel
from .dataclasses import ClusterParams, MCMCSamples, MixtureState, SamplerConfig
from .summaries import estimate_z_from_frequency, summarize_theta
from .initialization import init_spectral_with_z, init_clusters_default, init_blocks_spectral
from .moves import fast_gibbs_reassign_one_item, icm_sweep_cluster
from .utils import logsumexp, sample_categorical_from_logweights, dirichlet_sample
from .profiling import enable_profiling, disable_profiling, get_profiler, reset_profiling, MCMCProfiler
from .acceptance import acceptance_probabilities, print_acceptance_summary

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
]

