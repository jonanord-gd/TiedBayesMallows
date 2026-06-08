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
from .utils import dirichlet_sample, normalize_simplex, sample_categorical_from_logweights
from .blocks import blocks_to_block_index
from .distance import total_distance_fast
from .priors import log_Z_star_from_sizes, log_py_eppf_from_sizes, log_blocks_posterior
from .profiling import get_profiler
from .moves import (
    compute_U_all,
    build_cluster_pair_masks,
    build_pair_cache,
    compute_all_disagreements_fast,
    fast_gibbs_reassign_one_item,
    icm_sweep_cluster,
)
from .priors import build_log_qfactorials
from .augmentation import (
    PartialRankingInfo,
    detect_missing,
    complete_rankings,
    update_U_rows,
    augmentation_mh_step,
)

# Optional PyTorch GPU support — gracefully absent if not installed
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore
    _TORCH_AVAILABLE = False


class MixtureRankingModel:
    """
    Single entry-point object for TiedMallows MCMC:
      - initialize(...)
      - run_mcmc(...)
      - find_map(...)
    Includes lightweight caches so we don't rebuild block indices / T(m) repeatedly.
    """

    def __init__(
        self,
        rankings: List[List[int]],
        *,
        n_items: Optional[int] = None,
        partial_mode: str = "subset",
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
                partial_mode : {"subset", "top_k"}, default="subset"
                        Semantics used when a ranking omits some items.

                        * ``"subset"`` preserves only the observed relative order. The
                            latent completion may interleave missing items anywhere.
                        * ``"top_k"`` treats the observed items as a fixed prefix, with
                            all missing items constrained to appear below them.
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
            Initial cluster assignments for each ranking (z[i] = cluster for assessor i).
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
        self.N = len(rankings)

        # Determine n (total items).  With partial rankings the lists may
        # differ in length, so the caller can pass n_items explicitly.
        if n_items is not None:
            self.n = n_items
        else:
            self.n = len(rankings[0])
            if any(len(r) != self.n for r in rankings):
                raise ValueError(
                    "Rankings have different lengths.  When passing partial "
                    "rankings, set n_items to the total number of items."
                )

        self.rng = random.Random(seed)
        if partial_mode not in {"subset", "top_k"}:
            raise ValueError("partial_mode must be either 'subset' or 'top_k'")
        self.partial_mode = partial_mode

        # ── Partial-ranking detection & completion ────────────────────────
        self._partial_info: PartialRankingInfo = detect_missing(rankings, self.n)
        if self._partial_info.has_missing:
            # Store the original (observed) rankings and create completed copies
            self._original_rankings = [list(r) for r in rankings]
            self.rankings = complete_rankings(
                rankings,
                self._partial_info,
                self.n,
                self.rng,
                partial_mode=self.partial_mode,
            )
        else:
            self._original_rankings = None
            self.rankings = rankings

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

        if init_mu is None:
            self.init_mu = [1.0] * self.C
        elif len(init_mu) == 1:
            self.init_mu = [float(init_mu[0])] * self.C
        else:
            self.init_mu = list(init_mu)

        if len(self.init_mu) != self.C:
            raise ValueError("init_mu must have length 1 or C")

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

        tau0 = normalize_simplex(tau0)

        self.state = MixtureState(clusters=init_clusters, z=np.array(z0, dtype=np.intp), tau=tau0)

        # per-cluster caches (updated when blocks change)
        self._rebuild_all_cluster_caches()

        # Precompute per-assessor pairwise preference matrix (O(N n^2), done once).
        # Stored as float32: exact for 0/1 values, avoids per-iteration cast in matmul.
        # Uses self.rankings (which are completed if partial data was detected).
        self._U_all = compute_U_all(self.rankings, self.n).astype(np.float32)
        self._pair_cache = build_pair_cache(self.n)  # (n, n-1) pair indices

        # Precompute initial M / offsets for all clusters (cached across iterations)
        block_idx_list = [self._cache[c].block_idx for c in range(self.C)]
        self._M, self._offsets = build_cluster_pair_masks(block_idx_list, self.n)
        # _M_f32_T: contiguous (n_pairs, C) float32 transpose — reused every matmul
        self._M_f32_T: np.ndarray = np.ascontiguousarray(self._M.T, dtype=np.float32)
        self._M_dirty = [False] * self.C
        self._triu_indices = np.triu_indices(self.n, k=1)  # reused every rebuild

        # D_cache: (N, C) disagreement matrix maintained incrementally.
        # Initialised to None; the first call to _compute_all_disagreements()
        # does a full matmul and populates this.  Subsequent iterations use
        # sparse O(N·n) updates per block change (see _apply_incremental_D_update).
        self._D_cache: Optional[np.ndarray] = None

        # H_cache: (C, n_pairs) array — per-cluster pairwise preference sums
        # H_cache[c] = U_all[z==c].sum(0).  Maintained incrementally via
        # vectorised matmul after z updates.  Initialised to None; built from
        # scratch on the first z-update.
        self._H_cache: Optional[np.ndarray] = None

        # ── GPU acceleration ──────────────────────────────────────────────────
        # When a CUDA device is found, U_all and M are stored exclusively on GPU.
        # self._U_all is set to None to release CPU RAM (can be several GB for
        # large n).  All matmul and H-sum operations go through _U_row_sum() and
        # _compute_all_disagreements(), which dispatch to GPU or CPU automatically.
        self._gpu: Optional[Any] = None  # torch.device, or None when CPU-only
        self._U_all_t: Optional[Any] = None   # GPU tensor (N, n_pairs) float32
        self._M_t: Optional[Any] = None       # GPU tensor (n_pairs, C) float32
        self._offsets_t: Optional[Any] = None # GPU tensor (C,) float32
        if _TORCH_AVAILABLE and _torch.cuda.is_available():
            self._gpu = _torch.device("cuda")
            self._U_all_t = _torch.from_numpy(self._U_all).to(self._gpu)
            self._M_t = _torch.from_numpy(self._M_f32_T).to(self._gpu)
            self._offsets_t = _torch.from_numpy(
                self._offsets.astype(np.float32)).to(self._gpu)
            # Free the CPU copy — GPU tensor is the sole owner from here on
            self._U_all = None  # type: ignore

        # Initiate samples from MCMC run
        self.samples: Optional[MCMCSamples] = None

        # Sampler config
        self.cfg = SamplerConfig()

        # Per-cluster adaptive step sizes for theta proposals
        self._theta_steps: List[float] = [self.cfg.theta_step] * self.C

        # ── Frozen-cluster tracking ───────────────────────────────────────────────
        # A cluster is frozen immediately when it has 0 assessors: its blocks and
        # theta are not updated that iteration.  Frozen clusters remain eligible
        # for z-sampling (their weight comes from the Dirichlet prior on tau),
        # so they can naturally resurrect when the sampler reassigns an assessor.
        # _zero_streak records how many consecutive iterations a cluster has been
        # empty; _dead_clusters is the set of currently-frozen clusters.
        # COLLAPSE_PATIENCE is retained as an attribute for external inspection
        # but no longer controls permanent death.
        self.COLLAPSE_PATIENCE: int = 20   # kept for back-compat (no longer kills clusters)
        self._zero_streak: List[int] = [0] * self.C   # consecutive-empty counter
        self._dead_clusters: set = set()               # currently frozen (empty) clusters

        # Parallelization threshold: disable by default to avoid overhead
        # For large problems (N > 300), set this to enable parallelization
        self.parallel_threshold_n = parallel_threshold_n if parallel_threshold_n is not None else float('inf')
        
        # Verbose logging flag
        self.verbose = verbose
        if self.verbose:
            n_pairs = self.n * (self.n - 1) // 2
            u_bytes = self.N * n_pairs * 4  # float32 = 4 bytes
            u_mb    = u_bytes / 2**20
            print(f"[Model] Initialized: N={self.N} assessors, n={self.n} items, "
                  f"C={self.C} clusters, n_pairs={n_pairs:,}")
            if self._gpu is not None:
                dev_name = _torch.cuda.get_device_name(0)
                vram_mb  = _torch.cuda.get_device_properties(0).total_memory // 2**20
                print(f"[Model] Compute:  GPU — {dev_name} ({vram_mb:,} MB VRAM)")
                print(f"[Model] U_all:    {self.N}×{n_pairs:,}  ({u_mb:.1f} MB float32, on GPU)")
            else:
                if not _TORCH_AVAILABLE:
                    _reason = "torch not installed"
                elif not _torch.cuda.is_available():
                    _reason = "no CUDA device found"
                else:
                    _reason = "unknown"
                print(f"[Model] Compute:  CPU  (GPU unavailable — {_reason})")
                print(f"[Model] U_all:    {self.N}×{n_pairs:,}  ({u_mb:.1f} MB float32, on CPU)")
            print(f"[Model] Parallel: N threshold = "
                  f"{self.parallel_threshold_n if self.parallel_threshold_n != float('inf') else 'disabled'}")
            if self._partial_info.has_missing:
                print(f"[Model] Partial rankings: {self._partial_info.n_partial}/{self.N} "
                      f"assessors have missing items")
                print(f"[Model] Partial ranking semantics: {self.partial_mode}")
                miss_counts = [len(m) for m in self._partial_info.missing_items if m]
                print(f"[Model]   Missing items per assessor: "
                      f"min={min(miss_counts)}, max={max(miss_counts)}, "
                      f"mean={sum(miss_counts)/len(miss_counts):.1f}")
            for c, cl in enumerate(self.state.clusters):
                K = len(cl.blocks)
                print(f"  Cluster {c}: {K} blocks, theta={cl.theta:.3f}, gamma={cl.gamma:.3f}, delta={cl.delta:.3f}")

    # -----------------------
    # caching
    # -----------------------
    class _ClusterCache:
        __slots__ = ("sizes", "K", "block_idx")
        
        def __init__(self, sizes: List[int], K: int, block_idx: List[int]):
            self.sizes = sizes
            self.K = K
            self.block_idx = block_idx

    def _rebuild_cluster_cache(self, c: int) -> None:
        cl = self.state.clusters[c]
        sizes = [len(b) for b in cl.blocks]
        K = len(sizes)
        blk = blocks_to_block_index(cl.blocks, self.n, validate=False)
        self._cache[c] = MixtureRankingModel._ClusterCache(
            sizes=sizes, K=K, block_idx=blk
        )
        # Mark M row as dirty so _compute_all_disagreements rebuilds it
        if hasattr(self, '_M_dirty'):
            self._M_dirty[c] = True

    def _rebuild_all_cluster_caches(self) -> None:
        self._cache: List[MixtureRankingModel._ClusterCache] = [None] * self.C  # type: ignore
        for c in range(self.C):
            self._rebuild_cluster_cache(c)

    def _U_row_sum(self, mask: np.ndarray) -> np.ndarray:
        """Sum U_all rows selected by a boolean mask. Returns float64 (n_pairs,).

        Dispatches to GPU when available; otherwise uses CPU numpy. The result
        is always returned as a CPU numpy array for compatibility with downstream
        Python / scipy code.
        """
        if self._gpu is not None:
            mask_t = _torch.from_numpy(mask).to(self._gpu)
            return self._U_all_t[mask_t].sum(dim=0).cpu().numpy().astype(np.float64)
        return self._U_all[mask].sum(axis=0)

    # -----------------------
    # core updates
    # -----------------------
    
    def _compute_all_disagreements(self) -> np.ndarray:
        """Compute disagreements[i][c] for all assessors × clusters.

        **Fast path** (typical after iteration 1): if ``_D_cache`` is valid
        and no clusters are dirty, returns the cached matrix in O(1).
        The cache is maintained by ``_apply_incremental_D_update`` which
        performs O(N·n) sparse updates per block change instead of the
        O(N·C·n²) full matmul.

        **Slow path** (first call, or after U_all invalidation): full matmul
        ``U_all @ M.T + offsets``.  Sets ``_D_cache`` for future re-use.

        Returns ndarray of shape (N, C).
        """
        # ── fast path: D_cache already up-to-date ─────────────────────────
        if self._D_cache is not None and not any(self._M_dirty):
            return self._D_cache

        # ── slow path: full matmul rebuild ────────────────────────────────
        a_idx, b_idx = self._triu_indices

        for c in range(self.C):
            if self._M_dirty[c]:
                bidx_arr = np.asarray(self._cache[c].block_idx, dtype=np.intp)
                diff = bidx_arr[a_idx] - bidx_arr[b_idx]
                self._M[c] = np.sign(-diff)
                self._offsets[c] = np.count_nonzero(diff > 0)
                self._M_dirty[c] = False

        # Always rebuild the transpose — M may have been modified by
        # incremental updates that didn't touch _M_f32_T.
        np.copyto(self._M_f32_T, self._M.T)
        if self._gpu is not None:
            self._M_t.copy_(_torch.from_numpy(self._M_f32_T))
            self._offsets_t.copy_(_torch.from_numpy(
                self._offsets.astype(np.float32)))

        if self._gpu is not None:
            D_t = _torch.mm(self._U_all_t, self._M_t)
            D = D_t.cpu().numpy().astype(np.float64) + self._offsets[np.newaxis, :]
        else:
            D = self._U_all @ self._M_f32_T + self._offsets[np.newaxis, :]
        self._D_cache = D
        return D

    def _apply_incremental_D_update(
        self, c: int, moved_item: int,
        new_block_idx,
    ) -> None:
        """Incrementally update ``D_cache[:, c]`` and ``M[c]`` after one item moved.

        Only the n−1 pairs involving *moved_item* can change their M sign.
        Cost: O(N · n_changed) where n_changed ≤ n−1.  Compare to the full
        matmul at O(N · C · n²).
        """
        n = self.n
        l = moved_item

        # Pair indices for all (l, x) pairs
        xs = np.arange(n)
        xs = xs[xs != l]
        a_arr = np.minimum(l, xs)
        b_arr = np.maximum(l, xs)
        pidx_arr = (a_arr * (2 * n - a_arr - 1) // 2 + (b_arr - a_arr - 1)).astype(np.intp)

        # Old signs from current M[c] (still reflects pre-move state)
        old_signs = self._M[c, pidx_arr].copy()

        # New signs from new block_idx
        new_bi = np.asarray(new_block_idx, dtype=np.intp)
        new_signs = np.sign(new_bi[b_arr] - new_bi[a_arr]).astype(np.float64)

        # Only process pairs whose sign actually changed
        changed_mask = old_signs != new_signs
        if not changed_mask.any():
            return

        changed_pidx = pidx_arr[changed_mask]
        old_ch = old_signs[changed_mask]
        new_ch = new_signs[changed_mask]
        delta = (new_ch - old_ch).astype(np.float32)

        # Update M[c] in-place
        self._M[c, changed_pidx] = new_ch

        # Update offset[c]
        offset_delta = float((new_ch == -1.0).sum() - (old_ch == -1.0).sum())
        self._offsets[c] += offset_delta

        # Sparse D update:  D[:, c] += U_all[:, changed] @ delta + offset_delta
        if self._gpu is not None:
            pidx_t = _torch.from_numpy(changed_pidx.copy()).to(
                self._gpu, dtype=_torch.long)
            delta_t = _torch.from_numpy(delta.copy()).to(self._gpu)
            update = _torch.mv(self._U_all_t[:, pidx_t], delta_t)
            self._D_cache[:, c] += (
                update.cpu().numpy().astype(np.float64) + offset_delta
            )
        else:
            U_sub = self._U_all[:, changed_pidx]   # (N, n_changed) float32
            self._D_cache[:, c] += (
                (U_sub @ delta).astype(np.float64) + offset_delta
            )

    def _update_z(self) -> None:
        """Update cluster assignments using matmul-based disagreements.

        All N×C disagreements are computed in a single BLAS matmul call
        via ``_compute_all_disagreements`` (which uses precomputed U_all).
        Log-weights are built with vectorised numpy, and sampling uses the
        Gumbel-max trick for a single vectorised draw.

        After z changes, the per-cluster H_cache is updated incrementally:
        only rows that changed cluster have their U row subtracted from the
        old cluster's H and added to the new cluster's H.
        """
        state = self.state
        C = self.C
        cache = self._cache

        # Per-cluster scalars (computed once per iteration)
        tau_arr = np.asarray(normalize_simplex(state.tau), dtype=np.float64)
        state.tau = tau_arr.tolist()
        log_tau = np.log(np.clip(tau_arr, np.finfo(np.float64).tiny, None))
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
                cache[c].sizes, theta_c, self._qfact_cache[qkey])

        # N×C disagreement matrix (single matmul)
        D = self._compute_all_disagreements()  # ndarray (N, C)

        # Vectorised log-weights: shape (N, C)
        logweights = (log_tau[np.newaxis, :]
                      - thetas[np.newaxis, :] * D
                      - logZ[np.newaxis, :])

        # Frozen clusters (currently empty) are still eligible for z-assignment —
        # their log-weight is driven purely by the Dirichlet prior on tau, which
        # is small but positive, allowing natural resurrection when the sampler
        # finds it beneficial to reassign an assessor to them.

        # Save old z for incremental H update
        old_z = state.z.copy()

        # Gumbel-max trick: sample all N assignments in one vectorised op
        gumbels = -np.log(-np.log(
            np.random.default_rng(self.rng.randint(0, 2**31)).random((self.N, C))
        ))
        z_new = np.argmax(logweights + gumbels, axis=1)
        # Keep as ndarray — avoids .tolist() here and np.array() in every cluster_mask
        state.z = z_new

        # ── Incremental H_cache update ────────────────────────────────────
        self._update_H_cache_after_z_change(old_z, z_new)

    def _update_H_cache_after_z_change(
        self, old_z: np.ndarray, new_z: np.ndarray
    ) -> None:
        """Incrementally update H_cache after z assignments changed.

        If H_cache doesn't exist yet (first iteration), builds from scratch
        via a single indicator-matrix matmul:  H = Ind.T @ U_all.

        Otherwise, builds a (k, C) delta matrix with -1 at old cluster and
        +1 at new cluster for each changed assessor, then computes the
        update in one BLAS matmul:  H += delta.T @ U_changed.

        Both paths are loop-free and use multi-threaded BLAS.
        """
        C = self.C
        N = self.N

        if self._H_cache is None:
            # First time: build from scratch via indicator matmul
            # indicators: (N, C) one-hot, U_all: (N, n_pairs)
            # H = indicators.T @ U_all → (C, n_pairs)
            indicators = np.zeros((N, C), dtype=np.float32)
            indicators[np.arange(N), new_z] = 1.0
            if self._gpu is not None:
                ind_t = _torch.from_numpy(indicators).to(self._gpu)
                self._H_cache = (
                    (ind_t.T @ self._U_all_t)
                    .cpu().numpy().astype(np.float64)
                )
            else:
                self._H_cache = (indicators.T @ self._U_all).astype(
                    np.float64
                )
            return

        # Incremental path: find assessors that changed cluster
        changed_mask = old_z != new_z
        if not changed_mask.any():
            return

        changed_idx = np.where(changed_mask)[0]
        old_c = old_z[changed_idx]
        new_c = new_z[changed_idx]
        k = len(changed_idx)

        if self._gpu is not None:
            # delta: (k, C) with -1 at old cluster, +1 at new cluster
            delta = np.zeros((k, C), dtype=np.float32)
            delta[np.arange(k), old_c] = -1.0
            delta[np.arange(k), new_c] = 1.0

            idx_t = _torch.from_numpy(changed_idx.copy()).to(
                self._gpu, dtype=_torch.long
            )
            delta_t = _torch.from_numpy(delta).to(self._gpu)
            U_changed = self._U_all_t[idx_t]  # (k, n_pairs)
            # (C, k) @ (k, n_pairs) → (C, n_pairs)
            self._H_cache += (
                (delta_t.T @ U_changed)
                .cpu().numpy().astype(np.float64)
            )
        elif k <= 128:
            # Small k: per-cluster accumulation avoids creating a large
            # (k, n_pairs) temporary and the delta matrix entirely.
            U = self._U_all
            H = self._H_cache
            for c in range(C):
                leaving = changed_idx[old_c == c]
                if len(leaving) > 0:
                    H[c] -= U[leaving].sum(axis=0)
                joining = changed_idx[new_c == c]
                if len(joining) > 0:
                    H[c] += U[joining].sum(axis=0)
        else:
            # Large k: single BLAS matmul is more efficient.
            delta = np.zeros((k, C), dtype=np.float32)
            delta[np.arange(k), old_c] = -1.0
            delta[np.arange(k), new_c] = 1.0
            U_changed = self._U_all[changed_idx]  # (k, n_pairs) float32
            self._H_cache += (delta.T @ U_changed).astype(np.float64)

    def _update_tau(self) -> None:
        # Cache self references to eliminate attribute lookup overhead
        state = self.state
        rng = self.rng
        C = self.C
        init_mu = self.init_mu

        counts = np.bincount(state.z, minlength=C)
        post = [init_mu[c] + int(counts[c]) for c in range(C)]
        state.tau = normalize_simplex(dirichlet_sample(post, rng))

        # ── Freeze / resurrection tracking ────────────────────────────────────
        # Clusters are frozen immediately when they have 0 assessors: their
        # blocks and theta are not updated (handled by the empty-Rc guard in
        # step()).  A frozen cluster is NOT permanently excluded from z-sampling,
        # so it can be resurrected the moment at least one assessor enters it.
        newly_frozen: List[int] = []
        resurrected: List[int] = []
        for c in range(C):
            if counts[c] == 0:
                self._zero_streak[c] += 1
                if c not in self._dead_clusters:
                    self._dead_clusters.add(c)
                    newly_frozen.append(c)
            else:
                self._zero_streak[c] = 0
                if c in self._dead_clusters:
                    self._dead_clusters.discard(c)
                    resurrected.append(c)
        if newly_frozen and self.verbose:
            print(f"[Model] Clusters frozen (no assessors): {newly_frozen}  "
                  f"(total frozen: {len(self._dead_clusters)}/{C})")
        if resurrected and self.verbose:
            print(f"[Model] Clusters resurrected: {resurrected}")

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

        # S_c from cached D matrix (computed during _update_z): column sum
        # D[i,c] = cross-block disagreements (tie penalty cancels with Z*)
        if hasattr(self, '_D_cache') and self._D_cache is not None:
            cluster_mask = state.z == c
            S_c = float(self._D_cache[cluster_mask, c].sum())
        else:
            S_c = total_distance_fast(rankings_c, cl.blocks)

        # Reuse cached q-factorials where possible
        qfact_cache = getattr(self, '_qfact_cache', {})
        def _logZ(theta):
            q = math.exp(-theta)
            qkey = round(q, 15)
            lqf = qfact_cache.get(qkey)
            if lqf is None:
                lqf = build_log_qfactorials(self.n, q)
                qfact_cache[qkey] = lqf
            return log_Z_star_from_sizes(cc.sizes, theta, lqf)

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

        # Use cached H if available (maintained incrementally after z updates),
        # otherwise fall back to full sum.
        if self._H_cache is not None:
            _fast_H = self._H_cache[c]
        else:
            cluster_mask = state.z == c
            _fast_H = self._U_row_sum(cluster_mask)

        # Track block_idx for incremental D updates between moves
        current_block_idx = self._cache[c].block_idx

        # Precompute values constant across the n_item_moves loop:
        _log_qfact = build_log_qfactorials(self.n, math.exp(-cl_theta))
        _pair_cache = self._pair_cache

        for _ in range(n_item_moves):
            t_move_start = time.time()
            out_blocks, p, a, moved_item = fast_gibbs_reassign_one_item(
                rankings=rankings_c,
                blocks=cl_blocks,
                theta=cl_theta,
                gamma=gamma,
                delta=delta,
                H=_fast_H,
                rng=rng,
                log_qfact=_log_qfact,
                block_index=current_block_idx,
                pair_cache=_pair_cache,
                use_py_prior=cfg.use_py_prior,
            )
            t_move_elapsed = time.time() - t_move_start

            # Incremental D update: O(N·n) instead of full O(N·C·n²) matmul
            if self._D_cache is not None and moved_item >= 0:
                new_block_idx = blocks_to_block_index(
                    out_blocks, self.n, validate=False)
                self._apply_incremental_D_update(
                    c, moved_item, new_block_idx)
                current_block_idx = new_block_idx
            else:
                # Blocks may have changed even when D_cache is None;
                # recompute block_index for the next iteration.
                current_block_idx = blocks_to_block_index(
                    out_blocks, self.n, validate=False)

            if profiler is not None:
                profiler.record_move("gibbs_reassign", t_move_elapsed, accepted=False, proposals=p or 1, accepts=a or 0)

            proposals += (p if p is not None else 0)
            accepts += (a if a is not None else 0)
            cl_blocks = out_blocks

        cl.blocks = cl_blocks
        after_key = _canonicalize_blocks(cl_blocks)
        self._rebuild_cluster_cache(c)
        # M[c] was already updated incrementally — clear the dirty flag
        # that _rebuild_cluster_cache sets.
        if self._D_cache is not None:
            self._M_dirty[c] = False

        return proposals, accepts, (before_key != after_key)

    def _update_augmented_rankings(self) -> Tuple[int, int]:
        """MH augmentation step for partial rankings.

        For each assessor with missing items, propose a move that respects the
        configured partial-ranking semantics and accept/reject based on the
        Mallows likelihood under the assessor's current cluster. Accepted swaps modify
        ``self.rankings`` in-place and the corresponding U_all rows are
        recomputed.

        Returns ``(n_proposals, n_accepts)``.
        """
        if not self._partial_info.has_missing:
            return 0, 0

        n_prop, n_acc = augmentation_mh_step(
            rankings=self.rankings,
            info=self._partial_info,
            z=self.state.z,
            clusters=self.state.clusters,
            cache=self._cache,
            rng=self.rng,
            n=self.n,
            partial_mode=self.partial_mode,
        )

        if n_acc > 0:
            # Recompute U_all rows for assessors that were modified
            changed_idx = np.where(self._partial_info.partial_mask)[0]
            if self._gpu is not None:
                # CPU-side U_all was freed; work on a temporary then push to GPU
                U_cpu = self._U_all_t.cpu().numpy()
                update_U_rows(U_cpu, self.rankings, changed_idx, self.n)
                self._U_all_t.copy_(_torch.from_numpy(U_cpu))
            else:
                update_U_rows(self._U_all, self.rankings, changed_idx, self.n)

            # U_all changed → D_cache and H_cache are stale
            self._D_cache = None
            self._H_cache = None

        return n_prop, n_acc

    # -----------------------
    # public API
    # -----------------------
    def set_sampler(self, **kwargs) -> None:
        """Configures the SamplerConfig parameters for the MCMC sampler."""
        for k, v in kwargs.items():
            if not hasattr(self.cfg, k):
                raise ValueError(f"Unknown sampler setting: {k}")
            setattr(self.cfg, k, v)

    def step(self, iteration: int = 0, theta_jump: int = 1, ranking_jump: int = 1, **overrides) -> Dict[str, Any]:
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
        ranking_jump : int, default=1
            Update augmented (missing) rankings every ranking_jump iterations.
            Only takes effect when the data contains partial rankings.  Set to
            k > 1 to skip expensive augmentation steps most of the time.
            
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

        # Update augmented rankings for partial data (MH step)
        aug_proposals = 0
        aug_accepts = 0
        if self._partial_info.has_missing and iteration % ranking_jump == 0:
            t_start = time.time()
            aug_proposals, aug_accepts = self._update_augmented_rankings()
            if profiler:
                profiler.record_stage("update_augmented_rankings", time.time() - t_start)

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
            # Compute cluster rankings first; freeze immediately if the cluster
            # has no assessors this iteration (blocks and theta are unchanged).
            Rc = [r for r, zi in zip(rankings, state_z) if zi == c]
            if not Rc:
                continue

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
                # Update theta using per-cluster adaptive step size
                tp, ta = self._update_cluster_theta(
                    c, Rc,
                    a_theta=cfg.a_theta,
                    b_theta=cfg.b_theta,
                    step=self._theta_steps[c],
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
            "aug_proposals": aug_proposals,
            "aug_accepts": aug_accepts,
        }

    def run_mcmc(
        self,
        n_iter: int,
        *,
        burn_in: int = 0,
        thin: int = 1,
        theta_jump: int = 1,
        ranking_jump: int = 1,
        save_samples: bool = True,
        save_tau: bool = False,
        save_theta: bool = False,
        save_logp: bool = True,
        save_log_likelihood: bool = False,
        save_acceptance_details: bool = False,
        n_item_moves_per_cluster: int = 2,
        use_py_prior: bool = True,
        include_order_prior: bool = True,
        use_annealing: bool = False,
        annealing_schedule: Optional[List[float]] = None,
        annealing_schedule_type: str = "linear",
        annealing_plateau_frac: float = 0.5,
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
        ranking_jump : int, default=1
            Update augmented (missing) rankings every ranking_jump iterations.
            Only effective when the data contains partial rankings.  Set to k > 1
            to reduce the cost of the augmentation MH sweep.  E.g., ranking_jump=5
            updates completions every 5th iteration.
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
            Options: "linear" (linear interpolation), "exponential" (exponential growth),
            "plateau" (hold at temp_min for an initial fraction, then linearly ramp up).
            Only used if use_annealing=True and annealing_schedule is None.
        annealing_plateau_frac : float, default=0.5
            For the "plateau" schedule only: fraction of annealing iterations to keep
            fixed at temp_min before the ramp toward temp_max begins.
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
        if not 0.0 <= annealing_plateau_frac < 1.0:
            raise ValueError("annealing_plateau_frac must be in [0, 1)")

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
                elif annealing_schedule_type == "plateau":
                    # Plateau schedule: remain exploratory early, then ramp linearly.
                    plateau_iters = min(
                        int(round(annealing_plateau_frac * n_anneal_iters)),
                        max(0, n_anneal_iters - 1),
                    )
                    ramp_iters = n_anneal_iters - plateau_iters
                    if ramp_iters == 1:
                        ramp = [temp_max]
                    else:
                        ramp = [
                            temp_min + (temp_max - temp_min) * (t / max(1, ramp_iters - 1))
                            for t in range(ramp_iters)
                        ]
                    temperature_schedule = ([temp_min] * plateau_iters) + ramp
                elif annealing_schedule_type == "linear":
                    # Linear schedule: temp(t) = temp_min + (temp_max - temp_min) * (t / n_anneal)
                    temperature_schedule = [
                        temp_min + (temp_max - temp_min) * (t / max(1, n_anneal_iters - 1))
                        for t in range(n_anneal_iters)
                    ]
                else:
                    raise ValueError(
                        "annealing_schedule_type must be one of: linear, exponential, plateau"
                    )

        # Reset profiler at the start of a run if enabled
        profiler = get_profiler()
        if profiler is not None:
            profiler.reset()

        # apply any sampler-specific options upfront
        if sampler_kwargs:
            if "n_item_moves_per_cluster" in sampler_kwargs:
                n_item_moves_per_cluster = sampler_kwargs.pop("n_item_moves_per_cluster")
            self.set_sampler(**sampler_kwargs)

        # Apply Pitman-Yor prior flag
        self.cfg.use_py_prior = use_py_prior

        # Apply block-ordering prior flag
        self.cfg.include_order_prior = include_order_prior

        samples: Optional[MCMCSamples] = None
        if save_samples:
            samples = MCMCSamples(
                z_samples=[],
                blocks_samples=[],
                tau_samples=[] if save_tau else None,
                theta_samples=[] if save_theta else None,
                K = [],
                logp = [],
                saved_iterations=[],
                theta_jump=theta_jump,
                theta_accepts = [],
                block_accepts = [],
                theta_proposals = [] if save_acceptance_details else None,
                theta_accept_counts = [] if save_acceptance_details else None,
                block_proposals = [] if save_acceptance_details else None,
                block_accept_counts = [] if save_acceptance_details else None,
                log_likelihood = [] if save_log_likelihood else None,
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
            if save_log_likelihood and samples.log_likelihood is not None:
                samples.log_likelihood.append(self._compute_per_obs_log_lik())

        if self.verbose:
            print(f"\n[MCMC] Starting run: n_iter={n_iter}, burn_in={burn_in}, thin={thin}, theta_jump={theta_jump}")
            print(f"[MCMC] Item moves per cluster: {n_item_moves_per_cluster}")
            if self._partial_info.has_missing:
                print(f"[MCMC] Ranking augmentation: every {ranking_jump} iterations "
                      f"({self._partial_info.n_partial} partial assessors)")
            saved_iters = (n_iter - burn_in + thin - 1) // thin if save_samples else 0
            print(f"[MCMC] Will save {saved_iters} iterations after burn-in\n")

        t_start = time.time()
        theta_accepts_per_cluster = [0] * self.C
        block_accepts_per_cluster = [0] * self.C
        theta_proposals_per_cluster = [0] * self.C
        block_proposals_per_cluster = [0] * self.C
        aug_proposals_total = 0
        aug_accepts_total = 0

        # Adaptive theta step size (Robbins-Monro) during burn-in
        adapt_theta = self.cfg.adapt_theta_step and burn_in > 0
        target_acc = self.cfg.target_theta_acceptance
        # Per-cluster proposal count for computing decaying learning rate
        _theta_n_proposals = [0] * self.C
        # Reset per-cluster steps to current config value at start of run
        self._theta_steps = [self.cfg.theta_step] * self.C

        for it in range(n_iter):
            # Apply temperature annealing if active during burn-in
            if temperature_schedule is not None and it < len(temperature_schedule):
                temp_multiplier = temperature_schedule[it]
                # Scale thetas for this iteration
                original_thetas = [cl.theta for cl in self.state.clusters]
                for c in range(self.C):
                    self.state.clusters[c].theta = original_thetas[c] * temp_multiplier
            
            info = self.step(iteration=it, theta_jump=theta_jump, ranking_jump=ranking_jump, n_item_moves_per_cluster=n_item_moves_per_cluster)
            
            # Restore original thetas after step if annealing was applied
            if temperature_schedule is not None and it < len(temperature_schedule):
                for c in range(self.C):
                    self.state.clusters[c].theta = original_thetas[c]

            # ── Adapt theta step sizes during burn-in (Robbins-Monro) ──
            if adapt_theta and it < burn_in:
                for c in range(self.C):
                    n_prop = info["theta_proposals"][c]
                    if n_prop > 0:
                        _theta_n_proposals[c] += n_prop
                        accepted = info["theta_accept_counts"][c]
                        # Decaying learning rate: γ_t = 1 / sqrt(t)
                        gamma_t = 1.0 / math.sqrt(_theta_n_proposals[c])
                        # Update on log scale: log(σ) += γ * (α - target)
                        log_step = math.log(self._theta_steps[c])
                        log_step += gamma_t * (accepted - target_acc)
                        # Clamp to reasonable range [1e-4, 10]
                        log_step = max(math.log(1e-4), min(math.log(10.0), log_step))
                        self._theta_steps[c] = math.exp(log_step)
            
            for c in range(self.C):
                theta_accepts_per_cluster[c] += info["theta_accepts"][c]
                block_accepts_per_cluster[c] += info["block_accepts"][c]
                theta_proposals_per_cluster[c] += info.get("theta_proposals", [0]*self.C)[c]
                block_proposals_per_cluster[c] += info.get("block_proposals", [0]*self.C)[c]
            aug_proposals_total += info.get("aug_proposals", 0)
            aug_accepts_total += info.get("aug_accepts", 0)
            
            if save_samples and it >= burn_in and ((it - burn_in) % thin == 0):
                snapshot()
                assert samples is not None
                if samples.saved_iterations is not None:
                    samples.saved_iterations.append(it)
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
            if adapt_theta:
                print(f"[MCMC] Adapted theta step sizes (target acceptance={target_acc:.3f}):")
                for c in range(self.C):
                    print(f"  Cluster {c}: {self._theta_steps[c]:.4f} (was {self.cfg.theta_step:.4f})")
            if self._partial_info.has_missing and aug_proposals_total > 0:
                aug_rate = aug_accepts_total / aug_proposals_total
                print(f"[MCMC] Augmentation acceptance: {aug_rate:.1%} "
                      f"({aug_accepts_total}/{aug_proposals_total})")

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
        """Deprecated. Use find_map(method='frequency') instead."""
        import warnings
        warnings.warn(
            "estimate_map() is deprecated and will be removed in a future version. "
            "Use find_map(method='frequency', ci=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.find_map(samples, method="frequency", ci=ci)

    def find_map(
        self,
        samples: Optional[MCMCSamples] = None,
        *,
        method: str = "logp",
        refine: bool = True,
        max_sweeps: int = 50,
        ci: float = 0.95,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Find the MAP estimate from the MCMC chain.

        Two methods are available via the ``method`` parameter:

        ``method='logp'`` (default)
            Selects the sample with the highest log-joint probability and
            optionally refines it with Iterated Conditional Modes (ICM) — a
            deterministic hill-climbing pass that can further improve the
            log-joint beyond any visited sample.

            Returns a dict with keys:
                ``best_t``       – index of the best sample in the chain
                ``logp_chain``   – log-joint of the best sample (before refinement)
                ``logp_refined`` – log-joint after ICM refinement
                ``z``            – cluster assignments at the MAP
                ``tau``          – mixture weights at the MAP
                ``clusters``     – list of dicts per cluster, each with
                                   ``blocks``, ``theta``, ``icm_moves``, ``icm_sweeps``

        ``method='frequency'``
            Finds the most *frequently visited* block partition across the
            post-burn-in chain (the posterior mode by count) and summarises θ
            as a posterior mean/credible interval. Equivalent to the old
            ``estimate_map()`` call; primarily useful as a diagnostic to compare
            against the ``logp`` result.

            Returns a dict with keys:
                ``N``, ``C``, ``T``     – data/model/chain dimensions
                ``labels``              – frequency-based cluster assignments
                ``consensus_blocks``    – list of dicts per cluster with
                                         ``blocks_hat``, ``posterior_prob``,
                                         ``count``, ``n_unique``
                ``theta_summary``       – list of dicts per cluster with
                                         posterior mean and credible interval

        Parameters
        ----------
        samples : MCMCSamples, optional
            If None, uses ``self.samples``.
        method : str
            ``'logp'`` (default) or ``'frequency'``.
        refine : bool
            ``logp`` only. If True (default), run ICM after recovering the best sample.
        max_sweeps : int
            ``logp`` only. Maximum ICM sweeps per cluster.
        ci : float
            ``frequency`` only. Credible interval width for theta summary.
        verbose : bool
            ``logp`` only. Print progress.
        """
        if method == "frequency":
            return self._find_map_frequency(samples, ci=ci)
        if method != "logp":
            raise ValueError(f"method must be 'logp' or 'frequency', got {method!r}")
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

                H_c = self._U_row_sum(mask)
                cl = self.state.clusters[c]

                new_blocks, total_moves, sweeps = icm_sweep_cluster(
                    blocks=cl.blocks,
                    theta=cl.theta,
                    gamma=cl.gamma,
                    delta=cl.delta,
                    H=H_c,
                    N=N_c,
                    n=self.n,
                    max_sweeps=max_sweeps,
                    use_py_prior=self.cfg.use_py_prior,
                    include_uniform_order_prior=self.cfg.include_order_prior,
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

    def _find_map_frequency(
        self,
        samples: Optional[MCMCSamples],
        *,
        ci: float = 0.95,
    ) -> Dict[str, Any]:
        """Frequency-based MAP (posterior mode by count). Called by find_map(method='frequency')."""
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
        from .acceptance import acceptance_probabilities

        if self.samples is None:
            raise RuntimeError("No samples available. Run run_mcmc() first with save_samples=True.")

        return acceptance_probabilities(self.samples, self.C, parameter=parameter)

    def print_acceptance_summary(self) -> None:
        """Print a human-readable acceptance summary to stdout."""
        from .acceptance import print_acceptance_summary

        if self.samples is None:
            raise RuntimeError("No samples available. Run run_mcmc() first with save_samples=True.")

        print_acceptance_summary(self.samples, self.C)
    
    def _compute_per_obs_log_lik(self) -> List[float]:
        """Per-observation marginal log-likelihoods at the current MCMC state.

        Computes, for each ranker i:

            log p(r_i | tau, blocks, theta)
                = logsumexp_c [ log(tau_c) - theta_c * D[i,c] - log Z_c* ]

        where D[i,c] is the cross-block disagreement distance from ranker i to
        cluster c's consensus blocks.  The sum over clusters marginalises out the
        latent cluster assignment z_i, which is the quantity needed for arviz
        WAIC / LOO-CV.

        The computation reuses the cached D matrix (N × C), so the marginal cost
        on top of a normal MCMC iteration is O(N · C) — effectively free.

        Returns
        -------
        list of float, length N
        """
        state = self.state
        C = self.C

        # ── per-cluster log Z* ────────────────────────────────────────────
        qfact_cache = getattr(self, '_qfact_cache', {})
        logZ = np.empty(C)
        for c, cl in enumerate(state.clusters):
            q_c = math.exp(-cl.theta)
            qkey = round(q_c, 15)
            if qkey not in qfact_cache:
                qfact_cache[qkey] = build_log_qfactorials(self.n, q_c)
            logZ[c] = log_Z_star_from_sizes(
                self._cache[c].sizes, cl.theta, qfact_cache[qkey])

        # ── distance matrix ───────────────────────────────────────────────
        if self._D_cache is not None:
            D = self._D_cache                      # (N, C) — already up-to-date
        else:
            D = self._compute_all_disagreements()  # fallback, rare

        # ── vectorised logsumexp over clusters ───────────────────────────
        log_tau = np.log(np.asarray(state.tau, dtype=np.float64))  # (C,)
        thetas  = np.array([cl.theta for cl in state.clusters],
                           dtype=np.float64)                         # (C,)

        # logweights[i, c] = log(tau_c) - theta_c * D[i,c] - logZ_c
        logw = log_tau[None, :] - thetas[None, :] * D - logZ[None, :]  # (N, C)

        # numerically stable logsumexp along cluster axis
        m = logw.max(axis=1, keepdims=True)
        log_lik = (np.log(np.exp(logw - m).sum(axis=1)) + m[:, 0])   # (N,)

        return log_lik.tolist()

    def log_joint(self) -> float:
        """Unnormalized log posterior (up to constants) of current state.

        Uses the cached D matrix (from ``_compute_all_disagreements``) when
        available, avoiding an expensive O(N n²) recomputation per call.
        """
        state = self.state
        init_mu = self.init_mu
        C = self.C

        lp = 0.0

        # tau prior: Dirichlet(mu)
        state_tau = state.tau
        for c, t in enumerate(state_tau):
            if t <= 0:
                return float("-inf")
            lp += (init_mu[c] - 1.0) * math.log(t)

        # z likelihood
        tau_arr = np.asarray(state_tau)
        lp += float(np.log(tau_arr[state.z]).sum())

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
                # S_c = cross-block disagreements (tie penalty cancels with Z*)
                S_c = float(self._D_cache[mask, c].sum())

                # logZ via cached q-factorials
                q_c = math.exp(-cl.theta)
                qkey = round(q_c, 15)
                if qkey not in qfact_cache:
                    qfact_cache[qkey] = build_log_qfactorials(self.n, q_c)
                logZ_c = log_Z_star_from_sizes(
                    self._cache[c].sizes, cl.theta, qfact_cache[qkey])

                # Pitman-Yor prior (optional)
                if self.cfg.use_py_prior:
                    logpy = log_py_eppf_from_sizes(self._cache[c].sizes, cl.gamma, cl.delta)
                else:
                    logpy = 0.0
                K = len(cl.blocks)

                log_ord = math.lgamma(K + 1) if self.cfg.include_order_prior else 0.0
                lp += (-cl.theta * S_c) - (n_c * logZ_c) + logpy - log_ord
            else:
                Rc = [r for r, zi in zip(self.rankings, state.z) if zi == c]
                if not Rc:
                    continue
                lp += log_blocks_posterior(Rc, cl.blocks, cl.theta, cl.gamma, cl.delta,
                                          use_py_prior=self.cfg.use_py_prior,
                                          include_order_prior=self.cfg.include_order_prior)

            lp += self._log_gamma_pdf(cl.theta, self.cfg.a_theta, self.cfg.b_theta)

        return lp
    
    def _require_samples(self) -> None:
        if self.samples is None:
            raise RuntimeError("Run run_mcmc(..., save_samples=True) first.")

    def to_arviz(self, samples: Optional[MCMCSamples] = None) -> Any:
        """Convert MCMC samples to an :class:`arviz.InferenceData` object.

        The returned object contains a ``log_likelihood`` group (variable name
        ``"y"``, shape ``(chain=1, draw=T, obs=N)``), which enables arviz's WAIC
        and LOO-CV model-comparison utilities.

        Parameters
        ----------
        samples : MCMCSamples, optional
            Samples returned by :meth:`run_mcmc`.  Defaults to
            ``self.samples`` (the most recent run).

        Returns
        -------
        arviz.InferenceData

        Examples
        --------
        Run MCMC, compute log-likelihoods, convert, and compare models::

            state, samples = model.run_mcmc(5000, burn_in=1000, thin=5,
                                            save_log_likelihood=True)
            idata = model.to_arviz(samples)

            # LOO-CV model comparison (arviz 1.x)
            import arviz as az
            print(az.loo(idata))

            # Compare two models
            idata2 = model2.to_arviz(samples2)
            print(az.compare({"model1": idata, "model2": idata2}))

        Notes
        -----
        Per-observation log-likelihoods are *marginalised* over the latent
        cluster assignments:

            log p(r_i | tau, blocks, theta)
                = logsumexp_c [ log(tau_c) - theta_c * D[i,c] - log Z_c* ]

        This avoids the label-switching problem that arises when conditioning
        on the discrete z_i samples.
        """
        try:
            import arviz as az
        except ImportError as exc:
            raise ImportError(
                "arviz is required for to_arviz(). "
                "Install it with: pip install arviz"
            ) from exc

        if samples is None:
            samples = self.samples
        if samples is None:
            raise RuntimeError(
                "No samples available. Run run_mcmc(..., save_log_likelihood=True) first."
            )
        if samples.log_likelihood is None:
            raise RuntimeError(
                "log_likelihood was not saved. "
                "Re-run run_mcmc() with save_log_likelihood=True."
            )

        # Shape: (chain, draw, obs) = (1, T, N)
        ll_array = np.array(samples.log_likelihood, dtype=np.float64)[np.newaxis, :, :]

        # arviz 1.x: from_dict takes a single positional dict {group: {var: array}}
        # arviz <0.16: from_dict took keyword arguments per group
        try:
            return az.from_dict({"log_likelihood": {"y": ll_array}})
        except TypeError:
            # Fallback for older arviz where log_likelihood was a keyword arg
            return az.from_dict(log_likelihood={"y": ll_array})  # type: ignore[call-arg]

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

