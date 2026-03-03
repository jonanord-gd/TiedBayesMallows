"""Core MCMC model class for Tied Mallows mixture ranking."""

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

from .dataclasses import ClusterParams, MCMCSamples, MixtureState, SamplerConfig
from .summaries import (
    _canonicalize_blocks,
    _posterior_mode_from_counts,
    estimate_z_from_frequency,
    summarize_theta,
)
from .utils import dirichlet_sample, sample_categorical_from_logweights
from .initialization import init_blocks_borda_threshold
from .blocks import T_of_sizes, blocks_to_block_index
from .distance import cross_block_disagreements_fast, total_distance_fast
from .incremental_distance import IncrementalDistanceCalculator
from .priors import log_Z_star_from_sizes, log_blocks_posterior
from .moves import (
    gibbs_reassign_one_item,
    mh_adjacent_item_transfer,
    mh_adjacent_split_merge,
    mh_ordering_swap_or_shift,
    mh_py_prior_reassign_one_item,
)
from .profiling import get_profiler

# Numba support
try:
    from numba import njit
    _USE_NUMBA = True
except ImportError:
    _USE_NUMBA = False


class MixtureRankingModel:
    """
    Single entry-point object for TiedMallows MCMC:
      - initialize(...)
      - run_mcmc(...)
      - estimate_map(...)
    Includes lightweight caches so we don't rebuild block indices / T(m) repeatedly.
    """

    def __init__(
        self,
        rankings: List[List[int]],
        *,
        init_clusters: Optional[List[ClusterParams]] = None,
        n_clusters: Optional[int] = None,
        init_mu: Optional[List[float]] = None,
        init_theta: Optional[float] = None,
        init_gamma: Optional[float] = None,
        init_delta: Optional[float] = None,
        borda_gap_threshold: float = 0.35,
        seed: int = 123,
        verbose: bool = False,
    ):
        """Create a mixture model from observed rankings.

        Parameters
        ----------
        rankings : list of permutations
            The observed strict rankings (each a permutation of 0..n-1).
        init_clusters : list of ClusterParams, optional
            Initial configuration for each cluster (blocks, theta, gamma, delta).
            If omitted, clusters are auto-generated via Borda consensus with
            per-cluster variation. The list length determines the number of 
            clusters C.
        n_clusters : int, optional
            Number of clusters to create (ignored if init_clusters is provided).
            Required if init_clusters is None.
        init_mu : list of float, optional
            Dirichlet prior parameters for the cluster weights. If omitted a
            symmetric prior with all entries 1.0 is used.
        init_theta : float, optional
            Initial theta value for all clusters. If given, overrides cluster values.
        init_gamma : float, optional
            Common PY gamma to use for all clusters.
        init_delta : float, optional
            Common PY delta to use for all clusters.
        borda_gap_threshold : float
            Threshold for tying adjacent items in Borda-derived rankings.
            Only used when init_clusters is None. Smaller => fewer ties.
        seed : int
            Random seed for reproducibility.
        verbose : bool
            Enable verbose logging of model setup and MCMC progress.
        """
        if not rankings:
            raise ValueError("rankings must be non-empty")
        self.rankings = rankings
        self.N = len(rankings)
        self.n = len(rankings[0])
        if any(len(r) != self.n for r in rankings):
            raise ValueError("All rankings must have same length")

        self.rng = random.Random(seed)

        # Auto-generate clusters if not provided
        if init_clusters is None:
            if n_clusters is None:
                raise ValueError("Either init_clusters or n_clusters must be provided")
            if n_clusters <= 0:
                raise ValueError("n_clusters must be positive")
            
            # Generate per-cluster block rankings via Borda consensus with variation
            cluster_blocks_list = init_blocks_borda_threshold(
                rankings,
                n_clusters,
                gap_threshold=borda_gap_threshold,
                rng=self.rng,
            )
            
            # Set default values for theta, gamma, delta
            theta_val = init_theta if init_theta is not None else 1.0
            gamma_val = init_gamma if init_gamma is not None else 1.0
            delta_val = init_delta if init_delta is not None else 0.5
            
            init_clusters = [
                ClusterParams(
                    blocks=cluster_blocks_list[c],
                    theta=theta_val,
                    gamma=gamma_val,
                    delta=delta_val,
                )
                for c in range(n_clusters)
            ]
        else:
            # Use provided clusters
            if n_clusters is not None:
                raise ValueError("Cannot specify both init_clusters and n_clusters")
            # Apply any overrides to the provided clusters
            if init_theta is not None or init_gamma is not None or init_delta is not None:
                for cl in init_clusters:
                    if init_theta is not None:
                        cl.theta = init_theta
                    if init_gamma is not None:
                        cl.gamma = init_gamma
                    if init_delta is not None:
                        cl.delta = init_delta

        self.C = len(init_clusters)
        if self.C <= 0:
            raise ValueError("Need at least one cluster")

        self.init_mu = init_mu if init_mu is not None else [1.0] * self.C
        if len(self.init_mu) != self.C:
            raise ValueError("init_mu must have length C")

        # state
        z0 = [self.rng.randrange(self.C) for _ in range(self.N)]
        tau0 = dirichlet_sample(self.init_mu, self.rng)
        self.state = MixtureState(clusters=init_clusters, z=z0, tau=tau0)

        # per-cluster caches (updated when blocks change)
        self._rebuild_all_cluster_caches()
        
        # Incremental distance calculator for efficient block updates
        from .incremental_distance import IncrementalDistanceCalculator
        self.dist_calculator = IncrementalDistanceCalculator(rankings)

        # Initiate samples from MCMC run
        self.samples: Optional[MCMCSamples] = None

        # Sampler config
        self.cfg = SamplerConfig()
        
        # Verbose logging flag
        self.verbose = verbose
        if self.verbose:
            print(f"[Model] Initialized ({self.N} assessors, {self.n} items, {self.C} clusters)")
            print(f"[Model] Numba JIT: {'enabled' if _USE_NUMBA else 'disabled'}")
            for c, cl in enumerate(self.state.clusters):
                K = len(cl.blocks)
                print(f"  Cluster {c}: {K} blocks, theta={cl.theta:.3f}, gamma={cl.gamma:.3f}, delta={cl.delta:.3f}")

    # -----------------------
    # caching
    # -----------------------
    class _ClusterCache:
        __slots__ = ("sizes", "K", "Tm", "block_idx")
        
        def __init__(self, sizes: List[int], K: int, Tm: int, block_idx: List[int]):
            self.sizes = sizes
            self.K = K
            self.Tm = Tm
            self.block_idx = block_idx

    def _rebuild_cluster_cache(self, c: int) -> None:
        cl = self.state.clusters[c]
        sizes = [len(b) for b in cl.blocks]
        K = len(sizes)
        Tm = T_of_sizes(sizes)
        blk = blocks_to_block_index(cl.blocks, self.n, validate=False)
        self._cache[c] = MixtureRankingModel._ClusterCache(
            sizes=sizes, K=K, Tm=Tm, block_idx=blk
        )

    def _rebuild_all_cluster_caches(self) -> None:
        self._cache: List[MixtureRankingModel._ClusterCache] = [None] * self.C  # type: ignore
        for c in range(self.C):
            self._rebuild_cluster_cache(c)
    
    @staticmethod
    def _compute_changed_items(blocks_old: List[List[int]], blocks_new: List[List[int]]) -> set:
        """
        Identify which items moved between blocks.
        
        Returns set of item IDs where block assignment changed.
        """
        # Build item -> block indices
        old_idx = blocks_to_block_index(blocks_old, 
                                       sum(len(b) for b in blocks_old), 
                                       validate=False)
        new_idx = blocks_to_block_index(blocks_new,
                                       sum(len(b) for b in blocks_new),
                                       validate=False)
        
        changed = set()
        for item in range(len(old_idx)):
            if old_idx[item] != new_idx[item]:
                changed.add(item)
        return changed
    
    def _compute_distance_incremental(
        self,
        rankings_c: List[List[int]],
        blocks_old: List[List[int]],
        blocks_new: List[List[int]]
    ) -> int:
        """
        Compute distance using incremental calculation if beneficial.
        
        Falls back to full computation if:
        - Many items changed (incremental overhead not worth it)
        - Block count changed significantly (cache invalidation)
        
        Returns total distance for the cluster.
        """
        changed_items = self._compute_changed_items(blocks_old, blocks_new)
        
        # Threshold: if more than 30% of items changed, do full recompute
        pct_changed = len(changed_items) / self.n if self.n > 0 else 0
        if pct_changed > 0.3:
            # Full recomputation
            from .blocks import T_of_blocks
            Tm_new = T_of_blocks(blocks_new)
            return self.dist_calculator.compute_distance(blocks_new, Tm_new)
        
        # Use incremental calculation
        from .blocks import T_of_blocks
        Tm_new = T_of_blocks(blocks_new)
        
        return self.dist_calculator.compute_distance_incremental(
            blocks_old,
            blocks_new,
            changed_items,
            Tm_new
        )
    
    def _log_blocks_posterior_incremental(
        self,
        rankings_c: List[List[int]],
        blocks_old: List[List[int]],
        blocks_new: List[List[int]],
        theta: float,
        gamma: float,
        delta: float,
    ) -> float:
        """
        Compute log posterior using incremental distance calculation.
        
        This is an optimized version of log_blocks_posterior that uses
        incremental distance updates when blocks change slightly.
        
        Parameters
        ----------
        rankings_c : list of lists
            Rankings for this cluster
        blocks_old : list of lists
            Previous block structure
        blocks_new : list of lists
            New proposed block structure
        theta, gamma, delta : float
            Cluster parameters
        
        Returns
        -------
        log_posterior : float
            Log probability of the move
        """
        if not rankings_c:
            return float("-inf")
        
        from .blocks import T_of_blocks
        profiler = get_profiler()
        
        # Use incremental distance calculation
        if profiler:
            t_start = time.time()
        S_new = self._compute_distance_incremental(rankings_c, blocks_old, blocks_new)
        if profiler:
            profiler.record_operation("distance_calculation_incremental", time.time() - t_start)
        
        # Z* calculation (standard)
        sizes_new = [len(b) for b in blocks_new]
        K_new = len(sizes_new)
        
        if profiler:
            t_start = time.time()
        logZ = log_Z_star_from_sizes(sizes_new, theta, None)
        if profiler:
            profiler.record_operation("z_star_calculation", time.time() - t_start)
        
        # Pitman-Yor prior calculation
        if profiler:
            t_start = time.time()
        logpy = log_Z_star_from_sizes(sizes_new, gamma, delta)
        if profiler:
            profiler.record_operation("py_prior_calculation", time.time() - t_start)
        
        return (-theta * S_new) - (len(rankings_c) * logZ) + logpy - math.lgamma(K_new + 1)

    # -----------------------
    # core updates
    # -----------------------
    def _update_z(self) -> None:
        """Uses cached block_idx, K, Tm; only logZ depends on theta each step."""
        # Cache self references to eliminate attribute lookup overhead in tight loop
        state = self.state
        rng = self.rng
        C = self.C
        cache = self._cache
        rankings = self.rankings
        
        log_tau = [math.log(t) for t in state.tau]
        logZ = [log_Z_star_from_sizes(cache[c].sizes, state.clusters[c].theta, None) for c in range(C)]

        for i, r_i in enumerate(rankings):
            logw = []
            for c in range(C):
                cc = cache[c]
                theta = state.clusters[c].theta
                disc = cross_block_disagreements_fast(r_i, cc.block_idx, cc.K)
                d_ic = 2 * disc + cc.Tm
                logw.append(log_tau[c] - theta * d_ic - logZ[c])
            state.z[i] = sample_categorical_from_logweights(logw, rng)

    def _update_tau(self) -> None:
        # Cache self references to eliminate attribute lookup overhead
        state = self.state
        rng = self.rng
        C = self.C
        init_mu = self.init_mu
        
        counts = [0] * C
        for zi in state.z:
            counts[zi] += 1
        post = [init_mu[c] + counts[c] for c in range(C)]
        state.tau = dirichlet_sample(post, rng)

    def _cluster_rankings(self, c: int) -> List[List[int]]:
        # Cache self references
        state_z = self.state.z
        rankings = self.rankings
        return [r for r, zi in zip(rankings, state_z) if zi == c]

    @staticmethod
    def _log_gamma_pdf(x: float, a: float, b: float) -> float:
        if x <= 0:
            return float("-inf")
        return (a - 1.0) * math.log(x) - b * x + a * math.log(b) - math.lgamma(a)

    def _update_cluster_theta(
        self,
        c: int,
        rankings_c: List[List[int]],
        *,
        a_theta: float = 2.0,
        b_theta: float = 1.0,
        step: float = 0.25
    ) -> Tuple[int, int]:
        # Cache self references
        state = self.state
        rng = self.rng
        cache = self._cache
        
        n_c = len(rankings_c)
        if n_c == 0:
            return 0, 0

        cl = state.clusters[c]
        cc = cache[c]

        theta_old = cl.theta
        theta_new = math.exp(math.log(theta_old) + rng.gauss(0.0, step))

        # S_c using cached block_idx/K/Tm; delegate to vectorized helper
        S_c = total_distance_fast(rankings_c, cl.blocks)

        lp_old = (-theta_old * S_c) - n_c * log_Z_star_from_sizes(cc.sizes, theta_old, None) + self._log_gamma_pdf(theta_old, a_theta, b_theta)
        lp_new = (-theta_new * S_c) - n_c * log_Z_star_from_sizes(cc.sizes, theta_new, None) + self._log_gamma_pdf(theta_new, a_theta, b_theta)

        log_hast = math.log(theta_new) - math.log(theta_old)
        log_acc = (lp_new - lp_old) + log_hast

        accepted = math.log(rng.random()) < min(0.0, log_acc)
        if accepted:
            cl.theta = theta_new
        # (theta changes do not invalidate cache, since cache is only blocks-derived)
        return 1, (1 if accepted else 0)

    def _update_cluster_blocks(
        self,
        c: int,
        rankings_c: List[List[int]],
        *,
        n_item_moves: int = 2,
        p_gibbs_reassign: float = 0.0,
        p_transfer: float = 0.25,
        p_swapshift: float = 0.25,
        p_splitmerge: float = 0.25,
        p_reassign: float = 0.25,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
        profiler=None,  # OPTIMIZATION: Accept profiler instead of calling get_profiler() in loop
    ) -> Tuple[int, int, bool]:
        if not rankings_c:
            return 0, 0, False
        
        # Cache self references to eliminate repeated attribute lookups in tight loop
        rng = self.rng
        cfg = self.cfg
        state = self.state
        cl = state.clusters[c]
        
        # Cache frequently-accessed cluster parameters
        cl_blocks = cl.blocks
        cl_theta = cl.theta

        # allow sampler-wide overrides for the PY hyperparameters
        if gamma is None:
            gamma = cl.gamma
        if delta is None:
            delta = cl.delta

        # Cache cfg parameters accessed in loop
        cfg_ordering_p_short = cfg.ordering_p_short
        cfg_ordering_n_swap_steps = cfg.ordering_n_swap_steps
        cfg_ordering_max_long_step = cfg.ordering_max_long_step
        cfg_splitmerge_p_merge = cfg.splitmerge_p_merge

        # OPTIMIZATION: Initialize posterior cache to avoid recalculation when theta unchanged
        # Maps (blocks_tuple, theta, gamma, delta) → posterior_value
        # Cache naturally stays small (~10-20 entries) due to acceptance dynamics.
        posterior_cache: dict = {}
        cache_max_size = 50  # Limit to prevent memory growth; reset if exceeded

# Normalize ALL move probabilities (including p_reassign for the else clause)
        # The 4 block moves + 1 reassign move form a complete set of options
        s = p_gibbs_reassign + p_transfer + p_swapshift + p_splitmerge + p_reassign
        if s <= 0:
            # Fallback if all probabilities are 0
            s = 1.0
        p_gibbs_reassign, p_transfer, p_swapshift, p_splitmerge, p_reassign= (
            p_gibbs_reassign / s,
            p_transfer / s,
            p_swapshift / s,
            p_splitmerge / s,
            p_reassign / s,
        )

        # record before state to detect whether any MH move was accepted
        before_key = _canonicalize_blocks(cl_blocks)

        proposals = 0
        accepts = 0

        for _ in range(n_item_moves):
            u = rng.random()
            move_name = None
            t_move_start = time.time()
            
            if u < p_gibbs_reassign:
                move_name = "gibbs_reassign"
                out_blocks, p, a = gibbs_reassign_one_item(
                    rankings=rankings_c,
                    blocks=cl_blocks,
                    theta=cl_theta,
                    gamma=gamma,
                    delta=delta,
                    rng=rng
                )
            elif u < p_gibbs_reassign + p_transfer:
                move_name = "mh_transfer"
                out_blocks, p, a = mh_adjacent_item_transfer(
                    rankings_c=rankings_c,
                    blocks=cl_blocks,
                    theta=cl_theta,
                    gamma=gamma,
                    delta=delta,
                    rng=rng,
                    blocks_old=cl_blocks,
                    distance_calculator=self.dist_calculator,
                    use_parallel=(self.N > 100),
                    posterior_cache=posterior_cache
                )
            elif u < p_gibbs_reassign + p_transfer + p_swapshift:
                move_name = "mh_swapshift"
                out_blocks, p, a = mh_ordering_swap_or_shift(
                    rankings_c=rankings_c,
                    blocks=cl_blocks,
                    theta=cl_theta,
                    gamma=gamma,
                    delta=delta,
                    rng=rng,
                    p_short=cfg_ordering_p_short,
                    n_swap_steps=cfg_ordering_n_swap_steps,
                    max_long_step=cfg_ordering_max_long_step,
                    blocks_old=cl_blocks,
                    distance_calculator=self.dist_calculator,
                    use_parallel=(self.N > 100),
                    posterior_cache=posterior_cache
                )
            elif u < p_gibbs_reassign + p_transfer + p_swapshift + p_splitmerge:
                move_name = "mh_splitmerge"
                out_blocks, p, a = mh_adjacent_split_merge(
                    rankings_c=rankings_c,
                    blocks=cl_blocks,
                    theta=cl_theta,
                    gamma=gamma,
                    delta=delta,
                    rng=rng,
                    p_merge=cfg_splitmerge_p_merge,
                )
            else:
                move_name = "mh_py_reassign"
                out_blocks, p, a = mh_py_prior_reassign_one_item(
                    rankings_c=rankings_c,
                    blocks=cl_blocks,
                    theta=cl_theta,
                    gamma=gamma,
                    delta=delta,
                    rng=rng,
                    blocks_old=cl_blocks,
                    distance_calculator=self.dist_calculator,
                    use_parallel=(self.N > 100),
                    posterior_cache=posterior_cache
                )
            
            t_move_elapsed = time.time() - t_move_start
            
            # OPTIMIZATION: Use passed-in profiler instead of calling get_profiler() in loop
            if profiler is not None and move_name:
                profiler.record_move(move_name, t_move_elapsed, accepted=False, proposals=p or 1, accepts=a or 0)

            proposals += (p if p is not None else 0)
            accepts += (a if a is not None else 0)
            cl_blocks = out_blocks

        # Update the actual cluster blocks
        cl.blocks = cl_blocks
        
        # blocks changed -> refresh cache
        after_key = _canonicalize_blocks(cl_blocks)
        self._rebuild_cluster_cache(c)
        
        # OPTIMIZATION: Limit posterior cache size to prevent unbounded growth
        # Cache is naturally small but reset if exceeds threshold (e.g., many parameter changes)
        if len(posterior_cache) > cache_max_size:
            posterior_cache.clear()

        return proposals, accepts, (before_key != after_key)

    # -----------------------
    # public API
    # -----------------------
    def set_sampler(self, **kwargs) -> None:
        """Configures the SamplerConfig parameters for the MCMC sampler."""
        for k, v in kwargs.items():
            if not hasattr(self.cfg, k):
                raise ValueError(f"Unknown sampler setting: {k}")
            setattr(self.cfg, k, v)

    def step(self, iteration: int = 0, theta_jump: int = 1, **overrides) -> Dict[str, Any]:
        """Performs one MCMC step, updating z, tau, and cluster blocks/theta.
        
        Parameters
        ----------
        iteration : int
            Current iteration number (0-indexed). Used with theta_jump to determine
            whether to update theta in this iteration.
        theta_jump : int, default=1
            Update theta every theta_jump iterations. Set to k > 1 to skip theta
            updates most of the time. E.g., theta_jump=5 updates theta every 5th
            iteration, skipping 4 iterations with only block updates (2-5x speedup).
            
        Examples
        --------
        Every theta update (baseline):
            model.step(iteration=it, theta_jump=1)
            
        Sparse theta updates (faster):
            model.step(iteration=it, theta_jump=10)  # Update theta every 10 iterations
        """
        cfg = self.cfg
        # OPTIMIZATION: Cache profiler once to avoid repeated get_profiler() calls
        profiler = get_profiler()
        
        # If user specifies any block-move probability, auto-zero the unspecified ones
        block_move_keys = {'p_gibbs_reassign', 'p_transfer', 'p_swapshift', 'p_splitmerge', 'p_reassign'}
        specified_keys = block_move_keys & set(overrides.keys())
        if specified_keys:
            for key in block_move_keys - specified_keys:
                overrides[key] = 0.0
        
        # allow per-call overrides
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown sampler setting: {k}")
            setattr(cfg, k, v)

        # Cache self references for tight loop
        C = self.C
        state_z = self.state.z
        rankings = self.rankings

        # Update z
        t_start = time.time()
        self._update_z()
        if profiler:
            profiler.record_stage("update_z", time.time() - t_start)

        # Update tau
        t_start = time.time()
        self._update_tau()
        if profiler:
            profiler.record_stage("update_tau", time.time() - t_start)

        theta_accepts: List[int] = [0] * C
        block_accepts: List[int] = [0] * C
        theta_proposals: List[int] = [0] * C
        theta_accept_counts: List[int] = [0] * C
        block_proposals: List[int] = [0] * C
        block_accept_counts: List[int] = [0] * C

        for c in range(C):
            # Use cached state_z and rankings instead of accessing self repeatedly
            Rc = [r for r, zi in zip(rankings, state_z) if zi == c]
            
            # Update cluster blocks
            t_start = time.time()
            bp, ba, block_changed = self._update_cluster_blocks(
                c, Rc,
                n_item_moves=cfg.n_item_moves_per_cluster,
                p_gibbs_reassign=cfg.p_gibbs_reassign,
                p_transfer=cfg.p_transfer,
                p_swapshift=cfg.p_swapshift,
                p_splitmerge=cfg.p_splitmerge,
                p_reassign=cfg.p_reassign,
                gamma=cfg.gamma,
                delta=cfg.delta,
                profiler=profiler,  # OPTIMIZATION: Pass cached profiler to avoid get_profiler() in loop
            )
            if profiler:
                profiler.record_stage("update_blocks", time.time() - t_start)
            
            block_proposals[c] = int(bp)
            block_accept_counts[c] = int(ba)
            block_accepts[c] = 1 if block_changed else 0

            # Update cluster theta - OPTIMIZATION: Only if we're on a theta_jump iteration
            t_start = time.time()
            if iteration % theta_jump == 0:
                # Update theta
                tp, ta = self._update_cluster_theta(
                    c, Rc,
                    a_theta=cfg.a_theta,
                    b_theta=cfg.b_theta,
                    step=cfg.theta_step,
                )
                theta_proposals[c] = int(tp)
                theta_accept_counts[c] = int(ta)
                theta_accepts[c] = 1 if ta > 0 else 0
            else:
                # Skip theta update this iteration
                theta_proposals[c] = 0
                theta_accept_counts[c] = 0
                theta_accepts[c] = 0
            
            if profiler:
                profiler.record_stage("update_theta", time.time() - t_start)

        return {
            "theta_accepts": theta_accepts,
            "block_accepts": block_accepts,
            "theta_proposals": theta_proposals,
            "theta_accept_counts": theta_accept_counts,
            "block_proposals": block_proposals,
            "block_accept_counts": block_accept_counts,
        }

    def run_mcmc(
        self,
        n_iter: int,
        *,
        burn_in: int = 0,
        thin: int = 1,
        theta_jump: int = 1,
        save_samples: bool = True,
        save_tau: bool = False,
        save_theta: bool = False,
        save_logp: bool = True,
        save_acceptance_details: bool = False,
        n_item_moves_per_cluster: int = 2,
        **sampler_kwargs,
    ) -> Tuple[MixtureState, Optional[MCMCSamples]]:
        """Runs MCMC for ``n_iter`` iterations, with optional burn-in and thinning.
        
        Parameters
        ----------
        theta_jump : int, default=1
            Update theta every theta_jump iterations. Set to k > 1 to skip theta
            updates most of the time for faster sampling. E.g., theta_jump=10 updates
            theta every 10 iterations. This can provide 2-5x speedup at the cost of
            longer autocorrelation in theta chains.
            
        Examples
        --------
        Normal MCMC with theta every iteration:
            model.run_mcmc(n_iter=5000)
            
        Faster MCMC with sparse theta updates:
            model.run_mcmc(n_iter=5000, theta_jump=10)
        """
        if n_iter <= 0:
            raise ValueError("n_iter must be positive")
        if thin <= 0:
            raise ValueError("thin must be >= 1")
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0")

        # Reset profiler at the start of a run if enabled
        profiler = get_profiler()
        if profiler is not None:
            profiler.reset()

        # apply any sampler-specific options upfront
        if sampler_kwargs:
            block_move_keys = {'p_gibbs_reassign', 'p_transfer', 'p_swapshift', 'p_splitmerge', 'p_reassign'}
            specified_keys = block_move_keys & set(sampler_kwargs.keys())
            if specified_keys:
                for key in block_move_keys - specified_keys:
                    sampler_kwargs[key] = 0.0
            
            if "n_item_moves_per_cluster" in sampler_kwargs:
                n_item_moves_per_cluster = sampler_kwargs.pop("n_item_moves_per_cluster")
            self.set_sampler(**sampler_kwargs)

        samples: Optional[MCMCSamples] = None
        if save_samples:
            samples = MCMCSamples(
                z_samples=[],
                blocks_samples=[],
                tau_samples=[] if save_tau else None,
                theta_samples=[] if save_theta else None,
                K = [],
                logp = [],
                theta_accepts = [],
                block_accepts = [],
                theta_proposals = [] if save_acceptance_details else None,
                theta_accept_counts = [] if save_acceptance_details else None,
                block_proposals = [] if save_acceptance_details else None,
                block_accept_counts = [] if save_acceptance_details else None,
            )

        def snapshot() -> None:
            assert samples is not None
            samples.z_samples.append(self.state.z[:])
            samples.blocks_samples.append([[b[:] for b in cl.blocks] for cl in self.state.clusters])
            if save_tau and samples.tau_samples is not None:
                samples.tau_samples.append(self.state.tau[:])
            if save_theta and samples.theta_samples is not None:
                samples.theta_samples.append([cl.theta for cl in self.state.clusters])
            samples.K.append([len(cl.blocks) for cl in self.state.clusters])
            if save_logp and samples.logp is not None:
                samples.logp.append(self.log_joint())

        if self.verbose:
            print(f"\n[MCMC] Starting run: n_iter={n_iter}, burn_in={burn_in}, thin={thin}, theta_jump={theta_jump}")
            print(f"[MCMC] Sampler config: p_gibbs_reassign={self.cfg.p_gibbs_reassign:.3f}, p_transfer={self.cfg.p_transfer:.3f}, p_swapshift={self.cfg.p_swapshift:.3f}, p_splitmerge={self.cfg.p_splitmerge:.3f}, p_reassign={self.cfg.p_reassign:.3f}")
            print(f"[MCMC] Item moves per cluster: {n_item_moves_per_cluster}")
            saved_iters = (n_iter - burn_in + thin - 1) // thin if save_samples else 0
            print(f"[MCMC] Will save {saved_iters} iterations after burn-in\n")

        t_start = time.time()
        theta_accepts_per_cluster = [0] * self.C
        block_accepts_per_cluster = [0] * self.C
        theta_proposals_per_cluster = [0] * self.C
        block_proposals_per_cluster = [0] * self.C

        for it in range(n_iter):
            info = self.step(iteration=it, theta_jump=theta_jump, n_item_moves_per_cluster=n_item_moves_per_cluster)
            
            for c in range(self.C):
                theta_accepts_per_cluster[c] += info["theta_accepts"][c]
                block_accepts_per_cluster[c] += info["block_accepts"][c]
                theta_proposals_per_cluster[c] += info.get("theta_proposals", [0]*self.C)[c]
                block_proposals_per_cluster[c] += info.get("block_proposals", [0]*self.C)[c]
            
            if save_samples and it >= burn_in and ((it - burn_in) % thin == 0):
                snapshot()
                assert samples is not None
                samples.theta_accepts.append(info["theta_accepts"])
                samples.block_accepts.append(info["block_accepts"])
                if save_acceptance_details:
                    if samples.theta_proposals is not None:
                        samples.theta_proposals.append(info.get("theta_proposals", [0]*self.C))
                    if samples.theta_accept_counts is not None:
                        samples.theta_accept_counts.append(info.get("theta_accept_counts", [0]*self.C))
                    if samples.block_proposals is not None:
                        samples.block_proposals.append(info.get("block_proposals", [0]*self.C))
                    if samples.block_accept_counts is not None:
                        samples.block_accept_counts.append(info.get("block_accept_counts", [0]*self.C))

            if self.verbose and it > 0:
                milestones = [int(0.1*n_iter), int(0.25*n_iter), int(0.5*n_iter), int(0.75*n_iter), int(0.9*n_iter), n_iter-1]
                if it in milestones:
                    elapsed = time.time() - t_start
                    pct = 100.0 * (it + 1) / n_iter
                    rate = (it + 1) / elapsed
                    eta_sec = (n_iter - it - 1) / rate if rate > 0 else 0
                    eta_min = eta_sec / 60
                    print(f"[MCMC] {pct:6.1f}% | iter {it+1:6d}/{n_iter} | {elapsed:7.1f}s elapsed | ETA {eta_min:5.1f} min")

        t_elapsed = time.time() - t_start

        if self.verbose:
            print(f"\n[MCMC] Run complete in {t_elapsed:.1f}s ({t_elapsed/60:.2f} min)")
            if samples and samples.logp:
                logp_vals = samples.logp
                def format_logp(x):
                    fixed = f"{x:.2f}"
                    return f"{x:.2e}" if len(fixed) > 15 else fixed
                final_str = format_logp(logp_vals[-1])
                min_str = format_logp(min(logp_vals))
                max_str = format_logp(max(logp_vals))
                print(f"[MCMC] Final logp: {final_str} (min={min_str}, max={max_str})")
            
            print(f"[MCMC] Acceptance rates (theta | blocks):")
            for c in range(self.C):
                theta_rate = theta_accepts_per_cluster[c] / max(1, theta_proposals_per_cluster[c]) if theta_proposals_per_cluster[c] > 0 else 0
                block_rate = block_accepts_per_cluster[c] / max(1, block_proposals_per_cluster[c]) if block_proposals_per_cluster[c] > 0 else 0
                print(f"  Cluster {c}: {theta_rate:.1%} | {block_rate:.1%}")

        self.samples = samples
        return self.state, samples

    def get_profiling_summary(self) -> Optional[str]:
        """Get profiling summary if profiling is enabled. Returns None otherwise."""
        profiler = get_profiler()
        if profiler is None:
            return None
        return profiler.get_full_summary()
    
    def print_profiling_summary(self) -> None:
        """Print profiling summary if profiling is enabled."""
        profiler = get_profiler()
        if profiler is None:
            print("[Profiling] Profiling is not enabled. Use enable_profiling() first.")
            return
        profiler.print_summary()

    def estimate_map(
        self,
        samples: Optional[MCMCSamples] = None,
        *,
        ci: float = 0.95
    ) -> Dict[str, Any]:
        """Frequency MAP-like summaries (no label switching handling)."""
        if samples is None:
            if self.samples is None:
                raise RuntimeError("No samples available. Run run_mcmc(...) first with save_samples=True.")
            samples = self.samples

        z_samples = samples.z_samples
        blocks_samples = samples.blocks_samples
        if not z_samples or not blocks_samples:
            raise ValueError("Need z_samples and blocks_samples")

        out: Dict[str, Any] = {}
        out["N"] = len(z_samples[0])
        out["C"] = self.C
        out["T"] = len(z_samples)

        out["labels"] = estimate_z_from_frequency(z_samples, C=self.C)

        consensus_blocks = []
        for c in range(self.C):
            counts: Dict[Tuple[Tuple[int, ...], ...], int] = {}
            for t in range(len(z_samples)):
                key = _canonicalize_blocks(blocks_samples[t][c])
                counts[key] = counts.get(key, 0) + 1
            mode_key, mode_prob, mode_count = _posterior_mode_from_counts(counts)
            blocks_hat = [list(block) for block in mode_key]
            consensus_blocks.append({
                "cluster": c,
                "blocks_hat": blocks_hat,
                "posterior_prob": mode_prob,
                "count": mode_count,
                "n_unique": len(counts),
            })
        out["consensus_blocks"] = consensus_blocks

        if samples.theta_samples is not None:
            theta_summary = []
            for c in range(self.C):
                theta_c = [samples.theta_samples[t][c] for t in range(len(z_samples))]
                theta_summary.append({"cluster": c, **summarize_theta(theta_c, ci=ci)})
            out["theta_summary"] = theta_summary
        else:
            out["theta_summary"] = None

        return out
    
    def acceptance_statistics(self, parameter: Optional[str] = None) -> Dict[str, Any]:
        """Compute acceptance statistics after a run."""
        from AcceptanceProbabilities import acceptance_probabilities

        if self.samples is None:
            raise RuntimeError("No samples available. Run run_mcmc() first with save_samples=True.")

        return acceptance_probabilities(self.samples, self.C, parameter=parameter)

    def print_acceptance_summary(self) -> None:
        """Print a human-readable acceptance summary to stdout."""
        from AcceptanceProbabilities import print_acceptance_summary

        if self.samples is None:
            raise RuntimeError("No samples available. Run run_mcmc() first with save_samples=True.")

        print_acceptance_summary(self.samples, self.C)
    
    def log_joint(self) -> float:
        """Unnormalized log posterior (up to constants) of current state."""
        # Cache self references to eliminate repeated attribute lookups
        state = self.state
        init_mu = self.init_mu
        C = self.C
        rankings = self.rankings
        state_z = state.z
        
        lp = 0.0

        # tau prior: Dirichlet(mu) has log p(tau) = const + sum((mu_c-1)log tau_c)
        state_tau = state.tau
        for c, t in enumerate(state_tau):
            if t <= 0:
                return float("-inf")
            lp += (init_mu[c] - 1.0) * math.log(t)

        # z likelihood part
        for zi in state_z:
            lp += math.log(state_tau[zi])

        # cluster blocks + theta likelihood + priors
        for c, cl in enumerate(state.clusters):
            Rc = [r for r, zi in zip(rankings, state_z) if zi == c]
            if Rc:
                lp += log_blocks_posterior(Rc, cl.blocks, cl.theta, cl.gamma, cl.delta)
                lp += self._log_gamma_pdf(cl.theta, 2.0, 1.0)

        return lp
    
    def _require_samples(self) -> None:
        if self.samples is None:
            raise RuntimeError("Run run_mcmc(..., save_samples=True) first.")

    def plot_trace_theta(self, *, burn: int = 0, combined: bool = True) -> None:
        """Plot theta trace for all clusters.
        
        Parameters
        ----------
        burn : int
            Number of iterations to skip from the beginning
        combined : bool
            If True (default), plot all clusters on one figure with different colors.
            If False, create separate figure for each cluster (old behavior).
        """
        self._require_samples()
        import matplotlib.pyplot as plt

        assert self.samples is not None
        assert self.samples.theta_samples is not None, "Run with save_theta=True"

        T = len(self.samples.theta_samples)
        xs = list(range(burn, T))

        if combined:
            plt.figure(figsize=(12, 5))
            for c in range(self.C):
                ys = [self.samples.theta_samples[t][c] for t in xs]
                plt.plot(xs, ys, label=f"Cluster {c}", alpha=0.8)
            plt.title("Theta trace (all clusters)")
            plt.xlabel("Saved iteration")
            plt.ylabel("theta")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            for c in range(self.C):
                ys = [self.samples.theta_samples[t][c] for t in xs]
                plt.figure()
                plt.plot(xs, ys)
                plt.title(f"Theta trace (cluster {c})")
                plt.xlabel("Saved iteration")
                plt.ylabel("theta")
                plt.show()

    def plot_trace_tau(self, *, burn: int = 0, combined: bool = True) -> None:
        """Plot tau trace for all clusters.
        
        Parameters
        ----------
        burn : int
            Number of iterations to skip from the beginning
        combined : bool
            If True (default), plot all clusters on one figure with different colors.
            If False, create separate figure for each cluster (old behavior).
        """
        self._require_samples()
        import matplotlib.pyplot as plt

        assert self.samples is not None
        assert self.samples.tau_samples is not None, "Run with save_tau=True"

        T = len(self.samples.tau_samples)
        xs = list(range(burn, T))

        if combined:
            plt.figure(figsize=(12, 5))
            for c in range(self.C):
                ys = [self.samples.tau_samples[t][c] for t in xs]
                plt.plot(xs, ys, label=f"Cluster {c}", alpha=0.8)
            plt.title("Tau trace (all clusters)")
            plt.xlabel("Saved iteration")
            plt.ylabel("tau")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            for c in range(self.C):
                ys = [self.samples.tau_samples[t][c] for t in xs]
                plt.figure()
                plt.plot(xs, ys)
                plt.title(f"Tau trace (cluster {c})")
                plt.xlabel("Saved iteration")
                plt.ylabel("tau")
                plt.show()

    def plot_trace_K(self, *, burn: int = 0, combined: bool = True) -> None:
        """Plot number of blocks K trace for all clusters.
        
        Parameters
        ----------
        burn : int
            Number of iterations to skip from the beginning
        combined : bool
            If True (default), plot all clusters on one figure with different colors.
            If False, create separate figure for each cluster (old behavior).
        """
        self._require_samples()
        import matplotlib.pyplot as plt

        assert self.samples.K is not None

        T = len(self.samples.K)
        xs = list(range(burn, T))

        if combined:
            plt.figure(figsize=(12, 5))
            for c in range(self.C):
                ys = [self.samples.K[t][c] for t in xs]
                plt.plot(xs, ys, label=f"Cluster {c}", alpha=0.8, marker='o', markersize=3)
            plt.title("Number of blocks K trace (all clusters)")
            plt.xlabel("Saved iteration")
            plt.ylabel("K")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()
        else:
            for c in range(self.C):
                ys = [self.samples.K[t][c] for t in xs]
                plt.figure()
                plt.plot(xs, ys)
                plt.title(f"#Blocks K trace (cluster {c})")
                plt.xlabel("Saved iteration")
                plt.ylabel("K")
                plt.show()

    def plot_trace_logp(self, *, burn: int = 0) -> None:
        """Plot log-joint probability trace."""
        self._require_samples()
        import matplotlib.pyplot as plt

        assert self.samples.logp is not None

        xs = list(range(burn, len(self.samples.logp)))
        ys = self.samples.logp[burn:]

        plt.figure(figsize=(12, 5))
        plt.plot(xs, ys, color='darkred', linewidth=1.5)
        plt.title("Log-joint density trace")
        plt.xlabel("Saved iteration")
        plt.ylabel("log p(data, z, tau, theta | ...)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_traces(self, *, burn: int = 0, include_logp: bool = True) -> None:
        """Plot all traces together in a combined multi-panel figure.
        
        Creates a figure with:
        - Row 1: Theta trace (all clusters)
        - Row 2: Tau trace (all clusters) 
        - Row 3: K trace (all clusters)
        - Row 4: Log-joint trace (optional)
        
        Parameters
        ----------
        burn : int
            Number of iterations to skip from the beginning
        include_logp : bool
            If True, include log-joint trace at bottom
        """
        self._require_samples()
        import matplotlib.pyplot as plt

        assert self.samples is not None
        T_samples = len(self.samples.K) if self.samples.K else len(self.samples.theta_samples or [])
        
        n_rows = 4 if include_logp else 3
        if self.samples.theta_samples is None:
            n_rows -= 1
        if self.samples.tau_samples is None:
            n_rows -= 1

        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3*n_rows))
        if n_rows == 1:
            axes = [axes]

        xs = list(range(burn, T_samples))
        row = 0

        # Theta trace
        if self.samples.theta_samples is not None:
            ax = axes[row]
            for c in range(self.C):
                ys = [self.samples.theta_samples[t][c] for t in xs]
                ax.plot(xs, ys, label=f"Cluster {c}", alpha=0.8)
            ax.set_title("Theta trace", fontsize=12, fontweight='bold')
            ax.set_ylabel("theta")
            ax.legend(loc='best', fontsize=9)
            ax.grid(alpha=0.3)
            row += 1

        # Tau trace
        if self.samples.tau_samples is not None:
            ax = axes[row]
            for c in range(self.C):
                ys = [self.samples.tau_samples[t][c] for t in xs]
                ax.plot(xs, ys, label=f"Cluster {c}", alpha=0.8)
            ax.set_title("Tau trace", fontsize=12, fontweight='bold')
            ax.set_ylabel("tau")
            ax.legend(loc='best', fontsize=9)
            ax.grid(alpha=0.3)
            row += 1

        # K trace
        ax = axes[row]
        for c in range(self.C):
            ys = [self.samples.K[t][c] for t in xs]
            ax.plot(xs, ys, label=f"Cluster {c}", alpha=0.8, marker='o', markersize=3)
        ax.set_title("Number of blocks K trace", fontsize=12, fontweight='bold')
        ax.set_ylabel("K")
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
        row += 1

        # Log-joint trace
        if include_logp and self.samples.logp is not None:
            ax = axes[row]
            ys = self.samples.logp[burn:]
            ax.plot(xs, ys, color='darkred', linewidth=1.5)
            ax.set_title("Log-joint density", fontsize=12, fontweight='bold')
            ax.set_ylabel("log p")
            ax.grid(alpha=0.3)
            row += 1

        axes[-1].set_xlabel("Saved iteration")
        plt.tight_layout()
        plt.show()

