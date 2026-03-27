"""Core MCMC model class for Tied Mallows mixture ranking."""

import math
import random
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
    _USE_NUMPY = True
except ImportError:
    _USE_NUMPY = False

try:
    from joblib import Parallel, delayed
    _USE_JOBLIB = True
except ImportError:
    _USE_JOBLIB = False

from .dataclasses import ClusterParams, MCMCSamples, MixtureState, SamplerConfig
from .summaries import (
    _canonicalize_blocks,
    _posterior_mode_from_counts,
    estimate_z_from_frequency,
    summarize_theta,
)
from .utils import dirichlet_sample, sample_categorical_from_logweights
from .blocks import T_of_sizes, blocks_to_block_index
from .distance import total_distance_fast
from .priors import log_Z_star_from_sizes, log_py_eppf_from_sizes, log_blocks_posterior
from .profiling import get_profiler
from .moves import (
    compute_U_all,
    build_cluster_pair_masks,
    compute_all_disagreements_fast,
    fast_gibbs_reassign_one_item,
    icm_sweep_cluster,
)
from .priors import build_log_qfactorials

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
        init_z: Optional[List[int]] = None,
        init_theta: Optional[float] = None,
        init_gamma: Optional[float] = None,
        init_delta: Optional[float] = None,
        use_spectral_init: bool = False,
        seed: int = 123,
        verbose: bool = False,
        parallel_threshold_n: Optional[int] = None,
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
        init_z : list of int, optional
            Initial cluster assignments for each ranking (z[i] = cluster for ranking i).
            If provided, tau is derived from the frequency of assignments in z.
            If omitted, z is randomly assigned and tau is sampled from Dirichlet.
            
            Use with init_spectral_with_z() to intelligently initialize from spectral clustering:
            ```
            clusters, z = init_spectral_with_z(rankings, n_clusters=C, seed=123)
            model = MixtureRankingModel(rankings, init_clusters=clusters, init_z=z)
            ```
        init_theta : float, optional
            Initial theta value for all clusters. If given, overrides cluster values.
        init_gamma : float, optional
            Common PY gamma to use for all clusters.
        init_delta : float, optional
            Common PY delta to use for all clusters.
        use_spectral_init : bool, default=False
            When ``init_clusters`` is not provided and ``n_clusters`` is given,
            choose the auto-initialization strategy:

            * ``False`` (default): trivial init — each cluster starts with all
              items in one block. Fast and cheap; the Gibbs moves will build
              structure from scratch. Good for exploratory runs.
            * ``True``: run spectral clustering on the rankings to form initial
              clusters. Slower startup but produces better-structured initial
              blocks. Recommended for production runs.

            In both cases a ``UserWarning`` is raised to remind you to consider
            providing ``init_clusters`` explicitly via ``init_spectral_with_z()``.
        seed : int
            Random seed for reproducibility.
        verbose : bool
            Enable verbose logging of model setup and MCMC progress.
        parallel_threshold_n : int, optional
            If specified, enable parallelization of disagreement calculations when N >= parallel_threshold_n.
            Default (None): Parallelization is disabled. 
            
            Use this for large-scale problems:
            - parallel_threshold_n=200: Enable parallelization for N >= 200 assessors
            - parallel_threshold_n=500: Enable parallelization for N >= 500 assessors
            
            Parallelization has ~1-2ms overhead per call but benefits large problems.
            Recommended thresholds based on problem size:
            - N <= 100: Leave disabled (default, None)
            - 100 < N <= 300: parallel_threshold_n=200-250
            - N > 300: parallel_threshold_n=300
            
        """
        if len(rankings) == 0:
            raise ValueError("rankings must be non-empty")
        self.rankings = rankings
        self.N = len(rankings)
        self.n = len(rankings[0])
        if any(len(r) != self.n for r in rankings):
            raise ValueError("All rankings must have same length")

        self.rng = random.Random(seed)

        # Auto-generate clusters if not provided
        if init_clusters is None:
            import warnings
            if n_clusters is None:
                raise ValueError(
                    "n_clusters must be provided when init_clusters is not given."
                )
            warnings.warn(
                "init_clusters not provided. For best results supply them via "
                "init_spectral_with_z(rankings, n_clusters=C, seed=...). "
                "Falling back to " + ("spectral" if use_spectral_init else "trivial") + " initialization.",
                UserWarning,
                stacklevel=2,
            )
            if use_spectral_init:
                from .initialization import init_spectral_with_z
                init_clusters, _auto_z = init_spectral_with_z(
                    rankings, n_clusters=n_clusters, seed=seed)
            else:
                from .initialization import init_clusters_default
                init_clusters, _auto_z = init_clusters_default(
                    rankings, n_clusters,
                    init_theta=init_theta or 1.0,
                    init_gamma=init_gamma or 1.0,
                    init_delta=init_delta or 0.5,
                    seed=seed,
                )
            if init_z is None:
                init_z = _auto_z
        else:
            if n_clusters is not None:
                raise ValueError("Cannot specify both init_clusters and n_clusters")

        # Apply any parameter overrides to provided clusters
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

        # Initialize z and tau
        if init_z is not None:
            # Use provided cluster assignments
            if len(init_z) != self.N:
                raise ValueError(f"init_z must have length N={self.N}, got {len(init_z)}")
            if any(z < 0 or z >= self.C for z in init_z):
                raise ValueError(f"init_z values must be in range [0, {self.C-1}]")
            
            z0 = list(init_z)  # Copy the provided z
            
            # Compute tau from frequency of assignments in z
            # tau[c] = (count of c in z) / N
            tau0 = [0.0] * self.C
            for z_i in z0:
                tau0[z_i] += 1.0
            tau0 = [t / self.N for t in tau0]
        else:
            # Random initialization
            z0 = [self.rng.randrange(self.C) for _ in range(self.N)]
            tau0 = dirichlet_sample(self.init_mu, self.rng)
        
        self.state = MixtureState(clusters=init_clusters, z=z0, tau=tau0)

        # per-cluster caches (updated when blocks change)
        self._rebuild_all_cluster_caches()

        # Precompute per-assessor pairwise preference matrix (O(N n^2), done once)
        self._U_all = compute_U_all(rankings, self.n)

        # Precompute initial M / offsets for all clusters (cached across iterations)
        block_idx_list = [self._cache[c].block_idx for c in range(self.C)]
        self._M, self._offsets = build_cluster_pair_masks(block_idx_list, self.n)
        self._M_dirty = [False] * self.C
        self._triu_indices = np.triu_indices(self.n, k=1)  # reused every rebuild

        # Initiate samples from MCMC run
        self.samples: Optional[MCMCSamples] = None

        # Sampler config
        self.cfg = SamplerConfig()
        
        # Parallelization threshold: disable by default to avoid overhead
        # For large problems (N > 300), set this to enable parallelization
        self.parallel_threshold_n = parallel_threshold_n if parallel_threshold_n is not None else float('inf')
        
        # Verbose logging flag
        self.verbose = verbose
        if self.verbose:
            print(f"[Model] Initialized ({self.N} assessors, {self.n} items, {self.C} clusters)")
            print(f"[Model] Numba JIT: {'enabled' if _USE_NUMBA else 'disabled'}")
            print(f"[Model] Parallelization threshold: N >= {self.parallel_threshold_n if self.parallel_threshold_n != float('inf') else 'disabled'}")
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
        # Mark M row as dirty so _compute_all_disagreements rebuilds it
        if hasattr(self, '_M_dirty'):
            self._M_dirty[c] = True

    def _rebuild_all_cluster_caches(self) -> None:
        self._cache: List[MixtureRankingModel._ClusterCache] = [None] * self.C  # type: ignore
        for c in range(self.C):
            self._rebuild_cluster_cache(c)
    
    # -----------------------
    # core updates
    # -----------------------
    
    def _compute_all_disagreements(self) -> np.ndarray:
        """Compute disagreements[i][c] for all assessors × clusters via matmul.

        Uses precomputed U_all and cached M/offsets.  Only rebuilds M rows
        for clusters whose blocks changed (flagged dirty by
        ``_rebuild_cluster_cache``).

        The result is also stored in ``self._D_cache`` for reuse by
        ``_update_cluster_theta``.

        Returns ndarray of shape (N, C).
        """
        a_idx, b_idx = self._triu_indices

        for c in range(self.C):
            if self._M_dirty[c]:
                bidx_arr = np.asarray(self._cache[c].block_idx, dtype=np.intp)
                diff = bidx_arr[a_idx] - bidx_arr[b_idx]
                self._M[c] = np.sign(-diff)
                self._offsets[c] = np.count_nonzero(diff > 0)
                self._M_dirty[c] = False

        D = compute_all_disagreements_fast(self._U_all, self._M, self._offsets)
        self._D_cache = D
        return D

    def _update_z(self) -> None:
        """Update cluster assignments using matmul-based disagreements.

        All N×C disagreements are computed in a single BLAS matmul call
        via ``_compute_all_disagreements`` (which uses precomputed U_all).
        Log-weights are built with vectorised numpy, and sampling uses the
        Gumbel-max trick for a single vectorised draw.
        """
        state = self.state
        C = self.C
        cache = self._cache
        tie_penalty = self.cfg.tie_penalty

        # Per-cluster scalars (computed once per iteration)
        log_tau = np.array([math.log(t) for t in state.tau])
        thetas = np.array([state.clusters[c].theta for c in range(C)])

        # Cache q-factorials: rebuild only when theta changes
        if not hasattr(self, '_qfact_cache'):
            self._qfact_cache = {}
        logZ = np.empty(C)
        for c in range(C):
            theta_c = state.clusters[c].theta
            q_c = math.exp(-theta_c)
            # Cache key: (n, q) rounded to avoid float comparison issues
            qkey = round(q_c, 15)
            if qkey not in self._qfact_cache:
                self._qfact_cache[qkey] = build_log_qfactorials(self.n, q_c)
            logZ[c] = log_Z_star_from_sizes(
                cache[c].sizes, theta_c, self._qfact_cache[qkey], tie_penalty)

        Tms = np.array([cache[c].Tm for c in range(C)], dtype=np.float64)

        # N×C disagreement matrix (single matmul)
        D = self._compute_all_disagreements()  # ndarray (N, C)

        # Vectorised log-weights: shape (N, C)
        logweights = (log_tau[np.newaxis, :]
                      - thetas[np.newaxis, :] * (D + tie_penalty * Tms[np.newaxis, :])
                      - logZ[np.newaxis, :])

        # Gumbel-max trick: sample all N assignments in one vectorised op
        gumbels = -np.log(-np.log(
            np.random.default_rng(self.rng.randint(0, 2**31)).random((self.N, C))
        ))
        z_new = np.argmax(logweights + gumbels, axis=1)
        state.z = z_new.tolist()

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
        tie_penalty = self.cfg.tie_penalty
        
        n_c = len(rankings_c)
        if n_c == 0:
            return 0, 0

        cl = state.clusters[c]
        cc = cache[c]

        theta_old = cl.theta
        theta_new = math.exp(math.log(theta_old) + rng.gauss(0.0, step))

        # S_c from cached D matrix (computed during _update_z): column sum
        # D[i,c] = cross-block disagreements; add tie penalty for full distance
        if hasattr(self, '_D_cache') and self._D_cache is not None:
            cluster_mask = np.array([zi == c for zi in state.z], dtype=bool)
            S_c = float(self._D_cache[cluster_mask, c].sum()) + tie_penalty * cc.Tm * n_c
        else:
            S_c = total_distance_fast(rankings_c, cl.blocks, tie_penalty)

        # Reuse cached q-factorials where possible
        qfact_cache = getattr(self, '_qfact_cache', {})
        def _logZ(theta):
            q = math.exp(-theta)
            qkey = round(q, 15)
            lqf = qfact_cache.get(qkey)
            if lqf is None:
                lqf = build_log_qfactorials(self.n, q)
                qfact_cache[qkey] = lqf
            return log_Z_star_from_sizes(cc.sizes, theta, lqf, tie_penalty)

        lp_old = (-theta_old * S_c) - n_c * _logZ(theta_old) + self._log_gamma_pdf(theta_old, a_theta, b_theta)
        lp_new = (-theta_new * S_c) - n_c * _logZ(theta_new) + self._log_gamma_pdf(theta_new, a_theta, b_theta)

        log_hast = math.log(theta_new) - math.log(theta_old)
        log_acc = (lp_new - lp_old) + log_hast

        accepted = math.log(rng.random()) < min(0.0, log_acc)
        if accepted:
            cl.theta = theta_new
        return 1, (1 if accepted else 0)

    def _update_cluster_blocks(
        self,
        c: int,
        rankings_c: List[List[int]],
        *,
        n_item_moves: int = 2,
        gamma: Optional[float] = None,
        delta: Optional[float] = None,
        profiler=None,
    ) -> Tuple[int, int, bool]:
        if not rankings_c:
            return 0, 0, False

        rng = self.rng
        cfg = self.cfg
        state = self.state
        cl = state.clusters[c]
        cl_blocks = cl.blocks
        cl_theta = cl.theta

        if gamma is None:
            gamma = cl.gamma
        if delta is None:
            delta = cl.delta

        before_key = _canonicalize_blocks(cl_blocks)

        proposals = 0
        accepts = 0

        cluster_mask = np.array([zi == c for zi in state.z], dtype=bool)
        _fast_H = self._U_all[cluster_mask].sum(axis=0)

        for _ in range(n_item_moves):
            t_move_start = time.time()
            out_blocks, p, a = fast_gibbs_reassign_one_item(
                rankings=rankings_c,
                blocks=cl_blocks,
                theta=cl_theta,
                gamma=gamma,
                delta=delta,
                H=_fast_H,
                rng=rng,
                tie_penalty=cfg.tie_penalty,
            )
            t_move_elapsed = time.time() - t_move_start

            if profiler is not None:
                profiler.record_move("gibbs_reassign", t_move_elapsed, accepted=False, proposals=p or 1, accepts=a or 0)

            proposals += (p if p is not None else 0)
            accepts += (a if a is not None else 0)
            cl_blocks = out_blocks

        cl.blocks = cl_blocks
        after_key = _canonicalize_blocks(cl_blocks)
        self._rebuild_cluster_cache(c)

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
        profiler = get_profiler()

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
                gamma=cfg.gamma,
                delta=cfg.delta,
                profiler=profiler,
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
        tie_penalty: float = 0.5,
        use_annealing: bool = False,
        annealing_schedule: Optional[List[float]] = None,
        annealing_schedule_type: str = "linear",
        temp_min: float = 0.5,
        temp_max: float = 1.0,
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
        tie_penalty : float, default=0.5
            Weight for the within-block penalty term (T_m) in distance calculations.
            The p in the K^(p) extended Kendall distance, where p=0.5 recovers the
            Kemeny distance. Controls how much ties in rankings are penalized. p should be a value between (0, 1]
            Must be > 0.
            - Values < 0.5: De-emphasize ties relative to inversions (Only a near metric)
            - Values = 0.5: Kemeny/standard Mallows model (default)
            - Values > 0.5: Emphasize ties relative to inversions 
        use_annealing : bool, default=False
            If True, apply temperature annealing during burn-in to prevent cluster collapse.
            Starts with a soft likelihood (low theta) to explore cluster configurations,
            then gradually sharpens the likelihood (high theta) to focus on structure.
        annealing_schedule : list of float, optional
            Explicit temperature schedule. If provided, use_annealing is ignored.
            Each element is multiplied with the base theta value at that iteration.
            E.g., [0.5, 1.0, 2.0, 5.0] creates a 4-phase annealing schedule.
        annealing_schedule_type : str, default="linear"
            How to generate the annealing schedule if annealing_schedule is None.
            Options: "linear" (linear interpolation), "exponential" (exponential growth).
            Only used if use_annealing=True and annealing_schedule is None.
        temp_min : float, default=0.5
            Starting temperature (multiplier for theta) during annealing.
            Lower values (0.1-0.5) = softer likelihood = better exploration.
        temp_max : float, default=1.0
            Final temperature (multiplier for theta) after annealing completes.
            At temp_max=1.0, theta returns to its true posterior value.
            
        Examples
        --------
        Normal MCMC without annealing:
            model.run_mcmc(n_iter=5000)
            
        With temperature annealing (auto-generated schedule):
            model.run_mcmc(n_iter=5000, use_annealing=True, temp_min=0.5, burn_in=500)
            
        With custom temperature schedule:
            model.run_mcmc(n_iter=5000, annealing_schedule=[0.5, 1.0, 2.0, 5.0])
            
        Faster MCMC with sparse theta updates and annealing:
            model.run_mcmc(n_iter=5000, theta_jump=10, use_annealing=True, burn_in=500)
        """
        if n_iter <= 0:
            raise ValueError("n_iter must be positive")
        if thin <= 0:
            raise ValueError("thin must be >= 1")
        if burn_in < 0:
            raise ValueError("burn_in must be >= 0")
        if tie_penalty <= 0:
            raise ValueError("tie_penalty must be > 0")

        # Setup temperature annealing schedule
        temperature_schedule: Optional[List[float]] = None
        if annealing_schedule is not None:
            # Use explicit schedule
            temperature_schedule = annealing_schedule
        elif use_annealing:
            # Generate automatic schedule covering burn-in period
            n_anneal_iters = min(burn_in, max(50, n_iter // 4))  # Cool over burn-in or 25% of main run
            if n_anneal_iters > 0:
                if annealing_schedule_type == "exponential":
                    # Exponential schedule: temp(t) = temp_min * (temp_max/temp_min)^(t/n_anneal)
                    ratio = temp_max / temp_min if temp_min > 0 else 1.0
                    temperature_schedule = [
                        temp_min * (ratio ** (t / max(1, n_anneal_iters - 1)))
                        for t in range(n_anneal_iters)
                    ]
                else:  # linear (default)
                    # Linear schedule: temp(t) = temp_min + (temp_max - temp_min) * (t / n_anneal)
                    temperature_schedule = [
                        temp_min + (temp_max - temp_min) * (t / max(1, n_anneal_iters - 1))
                        for t in range(n_anneal_iters)
                    ]

        # Reset profiler at the start of a run if enabled
        profiler = get_profiler()
        if profiler is not None:
            profiler.reset()

        # apply any sampler-specific options upfront
        if sampler_kwargs:
            if "n_item_moves_per_cluster" in sampler_kwargs:
                n_item_moves_per_cluster = sampler_kwargs.pop("n_item_moves_per_cluster")
            self.set_sampler(**sampler_kwargs)

        # Apply tie penalty weight
        if tie_penalty != 0.5:
            self.cfg.tie_penalty = tie_penalty

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
            print(f"[MCMC] Item moves per cluster: {n_item_moves_per_cluster}")
            saved_iters = (n_iter - burn_in + thin - 1) // thin if save_samples else 0
            print(f"[MCMC] Will save {saved_iters} iterations after burn-in\n")

        t_start = time.time()
        theta_accepts_per_cluster = [0] * self.C
        block_accepts_per_cluster = [0] * self.C
        theta_proposals_per_cluster = [0] * self.C
        block_proposals_per_cluster = [0] * self.C

        for it in range(n_iter):
            # Apply temperature annealing if active during burn-in
            if temperature_schedule is not None and it < len(temperature_schedule):
                temp_multiplier = temperature_schedule[it]
                # Scale thetas for this iteration
                original_thetas = [cl.theta for cl in self.state.clusters]
                for c in range(self.C):
                    self.state.clusters[c].theta = original_thetas[c] * temp_multiplier
            
            info = self.step(iteration=it, theta_jump=theta_jump, n_item_moves_per_cluster=n_item_moves_per_cluster)
            
            # Restore original thetas after step if annealing was applied
            if temperature_schedule is not None and it < len(temperature_schedule):
                for c in range(self.C):
                    self.state.clusters[c].theta = original_thetas[c]
            
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

    def find_map(
        self,
        samples: Optional[MCMCSamples] = None,
        *,
        refine: bool = True,
        max_sweeps: int = 50,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Find the MAP state from the MCMC chain and optionally refine with ICM.

        1. Recover the state with the highest ``log_joint`` from saved samples.
        2. (Optional) Run Iterated Conditional Modes (ICM) sweeps on the block
           structures — deterministic hill-climbing that reuses the fast
           H-vector + prefix-sum machinery from the Gibbs sampler.

        Parameters
        ----------
        samples : MCMCSamples, optional
            If None, uses ``self.samples``.
        refine : bool
            If True (default), run ICM sweeps after recovering the best sample.
        max_sweeps : int
            Maximum number of ICM sweeps per cluster (stops early on convergence).
        verbose : bool
            Print progress information.

        Returns
        -------
        dict with keys:
            ``best_t``          – index of the best sample in the chain
            ``logp_chain``      – log-joint of the best sample (before refinement)
            ``logp_refined``    – log-joint after ICM refinement (if refine=True)
            ``z``               – cluster assignments
            ``tau``             – mixture weights
            ``clusters``        – list of dicts per cluster with ``blocks``, ``theta``,
                                  ``icm_moves``, ``icm_sweeps``
        """
        if samples is None:
            if self.samples is None:
                raise RuntimeError("No samples. Run run_mcmc(save_samples=True) first.")
            samples = self.samples

        if samples.logp is None or not samples.logp:
            raise ValueError("No logp saved. Run with save_samples=True and ensure logp is recorded.")

        # ── Step 1: find the best sample in the chain ──
        logp_arr = np.asarray(samples.logp)
        best_t = int(np.argmax(logp_arr))
        logp_chain = float(logp_arr[best_t])

        if verbose:
            print(f"[MAP] Best sample at t={best_t} with logp={logp_chain:.4f}")

        # Recover state
        z_best = samples.z_samples[best_t][:]
        blocks_best = [[b[:] for b in samples.blocks_samples[best_t][c]] for c in range(self.C)]

        tau_best = None
        if samples.tau_samples is not None:
            tau_best = samples.tau_samples[best_t][:]

        theta_best = []
        for c in range(self.C):
            if samples.theta_samples is not None:
                theta_best.append(samples.theta_samples[best_t][c])
            else:
                theta_best.append(self.state.clusters[c].theta)

        # Temporarily install this state into the model for log_joint evaluation
        old_state = self.state
        self.state = MixtureState(
            clusters=[
                ClusterParams(
                    blocks=blocks_best[c],
                    theta=theta_best[c],
                    gamma=old_state.clusters[c].gamma,
                    delta=old_state.clusters[c].delta,
                )
                for c in range(self.C)
            ],
            z=z_best,
            tau=tau_best if tau_best else old_state.tau[:],
        )
        # Rebuild caches for current state
        for c in range(self.C):
            self._rebuild_cluster_cache(c)
        self._M_dirty = [True] * self.C
        self._compute_all_disagreements()

        # ── Step 2: ICM refinement ──
        result_clusters = []
        if refine:
            z_arr = np.asarray(self.state.z, dtype=np.intp)

            for c in range(self.C):
                mask = z_arr == c
                N_c = int(mask.sum())
                if N_c == 0:
                    result_clusters.append({
                        "blocks": blocks_best[c],
                        "theta": theta_best[c],
                        "icm_moves": 0,
                        "icm_sweeps": 0,
                    })
                    continue

                H_c = self._U_all[mask].sum(axis=0)
                cl = self.state.clusters[c]

                new_blocks, total_moves, sweeps = icm_sweep_cluster(
                    blocks=cl.blocks,
                    theta=cl.theta,
                    gamma=cl.gamma,
                    delta=cl.delta,
                    H=H_c,
                    N=N_c,
                    n=self.n,
                    tie_penalty=self.cfg.tie_penalty,
                    max_sweeps=max_sweeps,
                )

                cl.blocks = new_blocks
                self._rebuild_cluster_cache(c)
                self._M_dirty[c] = True

                if verbose:
                    print(f"  Cluster {c}: {total_moves} moves in {sweeps} sweeps, "
                          f"K={len(new_blocks)} blocks")

                result_clusters.append({
                    "blocks": new_blocks,
                    "theta": theta_best[c],
                    "icm_moves": total_moves,
                    "icm_sweeps": sweeps,
                })

            # Recompute logp after refinement
            self._compute_all_disagreements()
            logp_refined = self.log_joint()
        else:
            logp_refined = logp_chain
            for c in range(self.C):
                result_clusters.append({
                    "blocks": blocks_best[c],
                    "theta": theta_best[c],
                    "icm_moves": 0,
                    "icm_sweeps": 0,
                })

        if verbose:
            delta_lp = logp_refined - logp_chain
            print(f"[MAP] logp: {logp_chain:.4f} → {logp_refined:.4f} "
                  f"(Δ={delta_lp:+.4f})")

        out = {
            "best_t": best_t,
            "logp_chain": logp_chain,
            "logp_refined": logp_refined,
            "z": self.state.z[:],
            "tau": self.state.tau[:],
            "clusters": result_clusters,
        }

        # Restore original state
        self.state = old_state
        for c in range(self.C):
            self._rebuild_cluster_cache(c)
        self._M_dirty = [True] * self.C

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
        """Unnormalized log posterior (up to constants) of current state.

        Uses the cached D matrix (from ``_compute_all_disagreements``) when
        available, avoiding an expensive O(N n²) recomputation per call.
        """
        state = self.state
        init_mu = self.init_mu
        C = self.C
        tie_penalty = self.cfg.tie_penalty

        lp = 0.0

        # tau prior: Dirichlet(mu)
        state_tau = state.tau
        for c, t in enumerate(state_tau):
            if t <= 0:
                return float("-inf")
            lp += (init_mu[c] - 1.0) * math.log(t)

        # z likelihood
        for zi in state.z:
            lp += math.log(state_tau[zi])

        # cluster blocks + theta + priors — via cached D matrix
        use_cache = hasattr(self, '_D_cache') and self._D_cache is not None
        if use_cache:
            z_arr = np.asarray(state.z, dtype=np.intp)
            qfact_cache = getattr(self, '_qfact_cache', {})

        for c, cl in enumerate(state.clusters):
            if use_cache:
                mask = z_arr == c
                n_c = int(mask.sum())
                if n_c == 0:
                    continue
                # S_c = sum of (cross-block disagreements + tie_penalty * Tm)
                S_c = float(self._D_cache[mask, c].sum()) + tie_penalty * self._cache[c].Tm * n_c

                # logZ via cached q-factorials
                q_c = math.exp(-cl.theta)
                qkey = round(q_c, 15)
                if qkey not in qfact_cache:
                    qfact_cache[qkey] = build_log_qfactorials(self.n, q_c)
                logZ_c = log_Z_star_from_sizes(
                    self._cache[c].sizes, cl.theta, qfact_cache[qkey], tie_penalty)

                # Pitman-Yor prior
                logpy = log_py_eppf_from_sizes(self._cache[c].sizes, cl.gamma, cl.delta)
                K = len(cl.blocks)

                lp += (-cl.theta * S_c) - (n_c * logZ_c) + logpy - math.lgamma(K + 1)
            else:
                Rc = [r for r, zi in zip(self.rankings, state.z) if zi == c]
                if not Rc:
                    continue
                lp += log_blocks_posterior(Rc, cl.blocks, cl.theta, cl.gamma, cl.delta, tie_penalty=tie_penalty)

            lp += self._log_gamma_pdf(cl.theta, self.cfg.a_theta, self.cfg.b_theta)

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

