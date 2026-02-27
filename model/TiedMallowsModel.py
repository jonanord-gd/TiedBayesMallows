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
from .initialization import init_blocks_borda_threshold
from .moves import (
    gibbs_reassign_one_item,
    mh_py_prior_reassign_one_item,
    mh_adjacent_split_merge,
    mh_adjacent_item_transfer,
    mh_ordering_swap_or_shift,
)
from .utils import logsumexp, sample_categorical_from_logweights, dirichlet_sample
from .profiling import enable_profiling, disable_profiling, get_profiler, reset_profiling

__all__ = [
    "MixtureRankingModel",
    "ClusterParams",
    "MixtureState",
    "MCMCSamples",
    "SamplerConfig",
    "estimate_z_from_frequency",
    "summarize_theta",
    "init_blocks_borda_threshold",
    "gibbs_reassign_one_item",
    "mh_py_prior_reassign_one_item",
    "mh_adjacent_split_merge",
    "mh_adjacent_item_transfer",
    "mh_ordering_swap_or_shift",
    "logsumexp",
    "sample_categorical_from_logweights",
    "dirichlet_sample",
    "enable_profiling",
    "disable_profiling",
    "get_profiler",
    "reset_profiling",
]

