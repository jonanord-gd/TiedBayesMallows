"""Backup of the original _compute_all_disagreements and _update_z methods
from model/core.py before the U_all matmul optimisation (March 2026).

These methods used per-ranking Fenwick-tree inversion counts:
O(N × C × n log K) per iteration.

To restore: copy these methods back into MixtureRankingModel in core.py,
and remove the U_all / fast-disagreement code.

NOTE: The inline relative imports (from .distance ...) have been changed to
absolute imports so this file is readable outside the model/ package.
"""


def _compute_all_disagreements_OLD(self):
    """
    OPTIMIZATION: Pre-compute all disagreement values in one pass.

    Returns:
        disagreements[i][c] = cross_block_disagreements_fast(ranking_i, cluster_c)
    """
    try:
        from model.distance import cross_block_disagreements_fast
    except ImportError:
        from TiedBayesMallows.model.distance import cross_block_disagreements_fast

    try:
        from joblib import Parallel, delayed
        _USE_JOBLIB = True
    except ImportError:
        _USE_JOBLIB = False

    cache = self._cache
    C = self.C
    N = self.N
    rankings = self.rankings

    use_parallel = _USE_JOBLIB and N >= self.parallel_threshold_n

    if use_parallel:
        disagreements_list = Parallel(n_jobs=-1, backend='threading')(
            delayed(self._compute_disagreements_for_assessor)(i)
            for i in range(N)
        )
        return disagreements_list
    else:
        disagreements = []
        for i, r_i in enumerate(rankings):
            disc_for_i = []
            for c in range(C):
                cc = cache[c]
                disc = cross_block_disagreements_fast(r_i, cc.block_idx, cc.K)
                disc_for_i.append(disc)
            disagreements.append(disc_for_i)
        return disagreements


def _compute_disagreements_for_assessor_OLD(self, i):
    """Helper for parallelized disagreement calculation."""
    try:
        from model.distance import cross_block_disagreements_fast
    except ImportError:
        from TiedBayesMallows.model.distance import cross_block_disagreements_fast

    cache = self._cache
    C = self.C
    r_i = self.rankings[i]

    disc_for_i = []
    for c in range(C):
        cc = cache[c]
        disc = cross_block_disagreements_fast(r_i, cc.block_idx, cc.K)
        disc_for_i.append(disc)
    return disc_for_i


def _update_z_OLD(self):
    """
    Uses cached calculations and vectorization to speed up cluster reassignment.
    Pre-matmul version: O(N × C × n log K) per iteration.
    """
    import math
    import numpy as np

    try:
        from model.priors import log_Z_star_from_sizes
        from model.utils import sample_categorical_from_logweights
    except ImportError:
        from TiedBayesMallows.model.priors import log_Z_star_from_sizes
        from TiedBayesMallows.model.utils import sample_categorical_from_logweights

    state = self.state
    rng = self.rng
    C = self.C
    cache = self._cache
    tie_penalty = self.cfg.tie_penalty

    log_tau = [math.log(t) for t in state.tau]
    logZ = [log_Z_star_from_sizes(cache[c].sizes, state.clusters[c].theta, None, tie_penalty)
            for c in range(C)]

    disagreements = self._compute_all_disagreements()

    thetas = [state.clusters[c].theta for c in range(C)]
    Tms = [cache[c].Tm for c in range(C)]

    try:
        disagreements_array = np.array(disagreements, dtype=np.float64)
        logweights_array = np.zeros_like(disagreements_array)

        for c in range(C):
            logweights_array[:, c] = (
                log_tau[c]
                - thetas[c] * (disagreements_array[:, c] + tie_penalty * Tms[c])
                - logZ[c]
            )

        for i in range(self.N):
            logw = logweights_array[i].tolist()
            state.z[i] = sample_categorical_from_logweights(logw, rng)
    except Exception:
        for i in range(self.N):
            logw = []
            for c in range(C):
                disc = disagreements[i][c]
                d_ic = disc + tie_penalty * Tms[c]
                logw.append(log_tau[c] - thetas[c] * d_ic - logZ[c])
            state.z[i] = sample_categorical_from_logweights(logw, rng)
