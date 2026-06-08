import numpy as np
import arviz as az
import json
import os
import time
from datetime import datetime

import pandas as pd

from model import MixtureRankingModel
from model.initialization import init_spectral_with_z

# ── Config ────────────────────────────────────────────────────────────────────
n_iter    = 10000
burn_in   = 3000
thin      = 1
K         = 20
mu        = 0.0001
seed      = 42

output_dir = "pan_cancer/runs"
os.makedirs(output_dir, exist_ok=True)

runtimes = {}

# ── Load rankings ─────────────────────────────────────────────────────────────
print("Loading rankings...")
_t0 = time.perf_counter()
df = pd.read_csv("data/rankings.csv", index_col=0)
# R[i,j] = rank of gene j for assessor i (1-based).
# The model expects orderings: ranking[position] = item.
# argsort converts rank matrix → ordering: ordering[k] = gene at position k.
rankings = np.argsort(df.values.astype(int), axis=1).tolist()  # list of N lists, each length n_items

N       = len(rankings)
n_items = len(rankings[0])
runtimes["load_s"] = time.perf_counter() - _t0
print(f"  N={N} assessors, n_items={n_items} items  [{runtimes['load_s']:.1f}s]")

# ── Spectral init ─────────────────────────────────────────────────────────────
print(f"Running spectral clustering (K={K})...")
_t0 = time.perf_counter()
clusters_init, z_init = init_spectral_with_z(
    rankings    = rankings,
    n_clusters  = K,
    py_sampling = False,
    gamma       = 1.0,
    delta       = 0.5,
    seed        = seed,
)
runtimes["spectral_s"] = time.perf_counter() - _t0
print(f"  Spectral clustering done  [{runtimes['spectral_s']:.1f}s]")

# ── Fit ───────────────────────────────────────────────────────────────────────
print("Fitting model...")
_t0 = time.perf_counter()
model = MixtureRankingModel(
    rankings      = rankings,
    init_clusters = clusters_init,
    init_z        = z_init,
    init_mu       = [mu] * K,
    seed          = seed,
    verbose       = True,
    init_theta    = 1.0,
)

_, samples = model.run_mcmc(
    n_iter                   = n_iter,
    burn_in                  = burn_in,
    thin                     = thin,
    use_py_prior             = False,
    include_order_prior      = False,
    save_samples             = True,
    save_tau                 = True,
    save_theta               = True,
    n_item_moves_per_cluster = 1,
    a_theta                  = 2,
    b_theta                  = 1,
    save_log_likelihood      = True,
    theta_jump               = 5,
    ranking_jump             = 1,
    use_annealing            = True,
    temp_min                 = 0.1,
    temp_max                 = 1.0,
)
runtimes["mcmc_s"] = time.perf_counter() - _t0
print(f"  MCMC done  [{runtimes['mcmc_s']:.1f}s]")

# ── LOO ───────────────────────────────────────────────────────────────────────
print("Computing LOO...")
_t0 = time.perf_counter()
ll_matrix = np.array(samples.log_likelihood)   # (T, N)
T = ll_matrix.shape[0]
idata = az.from_dict({
    "posterior":      {"dummy": np.zeros((1, T))},
    "log_likelihood": {"rankings": ll_matrix[np.newaxis]},
})
loo = az.loo(idata, var_name="rankings", pointwise=True)
pareto_k = loo.pareto_k.values

runtimes["loo_s"] = time.perf_counter() - _t0
print(f"  ELPD={loo.elpd:.1f} (SE={loo.se:.1f}), p_loo={loo.p:.1f}  [{runtimes['loo_s']:.1f}s]")
print(f"  Pareto-k: {(pareto_k < loo.good_k).sum()} good, "
      f"{(pareto_k >= loo.good_k).sum()} bad, "
      f"{(pareto_k >= 1.0).sum()} very bad")

# ── MAP ───────────────────────────────────────────────────────────────────────
print("Extracting MAP...")
_t0 = time.perf_counter()
map_result = model.find_map(samples, refine=True, verbose=False)
z_map      = map_result["z"]
K_actual   = len(set(z_map))

blocks = {}
theta_map = {}
for c, cl in enumerate(map_result["clusters"]):
    if c in z_map:
        blocks[str(c)]    = cl["blocks"]
        theta_map[str(c)] = float(cl["theta"])

z_trace = np.array(samples.z_samples)   # (T, N)
runtimes["map_s"] = time.perf_counter() - _t0
print(f"  K_init={K} → K_actual={K_actual}  [{runtimes['map_s']:.1f}s]")

# ── Save ──────────────────────────────────────────────────────────────────────
tag = f"pan_cancer_K{K}_mu{mu}_seed{seed}"

np.savez_compressed(
    os.path.join(output_dir, f"{tag}.npz"),
    z_map    = np.array(z_map,    dtype=np.int32),
    z_trace  = np.array(z_trace,  dtype=np.int32),
    elpd_i   = np.array(loo.elpd_i.values, dtype=np.float64),
    pareto_k = np.array(pareto_k, dtype=np.float64),
)

meta = {
    "K_init":       K,
    "K_actual":     K_actual,
    "mu":           mu,
    "seed":         seed,
    "N":            N,
    "n_items":      n_items,
    "n_iter":       n_iter,
    "burn_in":      burn_in,
    "thin":         thin,
    "elpd":         float(loo.elpd),
    "se":           float(loo.se),
    "p_loo":        float(loo.p),
    "good_k":       float(loo.good_k),
    "n_bad_k":      int((pareto_k >= loo.good_k).sum()),
    "n_very_bad_k": int((pareto_k >= 1.0).sum()),
    "blocks":       blocks,
    "theta_map":    theta_map,
    "runtimes_s":   runtimes,
    "timestamp":    datetime.now().isoformat(),
}
with open(os.path.join(output_dir, f"{tag}.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSaved → {tag}.npz + {tag}.json")
print("Done.")
