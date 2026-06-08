import numpy as np
import arviz as az
import json
import os
from datetime import datetime

from model import MixtureRankingModel
from model.initialization import init_spectral_with_z

# ── Config ────────────────────────────────────────────────────────────────────
n_iter  = 10000
burn_in = 5000
thin    = 1

K_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
seeds    = [42]

# Parse sushi3a rankings
# Format: each line is "<user_id> <n_items> <rank_1> ... <rank_10>"
# First line is a header
rankings = []
with open("sushi3/sushi3-2016/sushi3a.5000.10.order") as f:
    for i, line in enumerate(f):
        if i == 0:
            continue  # skip header
        parts = line.strip().split()
        ranking = [int(x) for x in parts[2:]]  # skip user_id and n_items
        rankings.append(ranking)

rankings = np.array(rankings)  # shape (5000, 10)

N = rankings.shape[0]
m = rankings.shape[1]

output_dir = "timm_runs_spectral"
os.makedirs(output_dir, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_loo(samples):
    """Compute PSIS-LOO from samples.log_likelihood (T, N)."""
    ll_matrix = np.array(samples.log_likelihood)   # (T, N)
    T = ll_matrix.shape[0]
    idata = az.from_dict({
        "posterior":      {"dummy": np.zeros((1, T))},
        "log_likelihood": {"rankings": ll_matrix[np.newaxis]}  # (1, T, N)
    })
    return az.loo(idata, var_name="rankings", pointwise=True)


def extract_map(model, samples, K_init):
    """
    Extract MAP z, blocks via model.find_map, and z_trace for consensus.
    Returns z_map, blocks, z_trace, K_actual.
    """
    map_result = model.find_map(samples, refine=True, verbose=False)

    z_map  = map_result["z"]   # list (N,)

    # Only active clusters (those present in z_map)
    blocks = {}
    for c, cl in enumerate(map_result["clusters"]):
        if c in z_map:
            blocks[str(c)] = cl["blocks"]  # str key for json compatibility

    z_trace  = np.array(samples.z_samples)   # (T, N)
    K_actual = len(blocks)

    return z_map, blocks, z_trace, K_actual


def save_run(run_data, output_dir):
    """Save arrays to .npz and scalars/metadata to .json."""
    tag = f"K{run_data['K_init']}_seed{run_data['seed']}"

    np.savez_compressed(
        os.path.join(output_dir, f"{tag}.npz"),
        z_map    = np.array(run_data["z_map"],    dtype=np.int32),   # (N,)
        z_trace  = np.array(run_data["z_trace"],  dtype=np.int32),   # (T, N)
        elpd_i   = np.array(run_data["elpd_i"],   dtype=np.float64), # (N,)
        pareto_k = np.array(run_data["pareto_k"], dtype=np.float64), # (N,)
    )

    meta = {
        "K_init":       run_data["K_init"],
        "K_actual":     run_data["K_actual"],
        "seed":         run_data["seed"],
        "N":            N,
        "m":            m,
        "n_iter":       n_iter,
        "burn_in":      burn_in,
        "thin":         thin,
        "elpd":         run_data["elpd"],
        "se":           run_data["se"],
        "p_loo":        run_data["p_loo"],
        "good_k":       run_data["good_k"],
        "n_bad_k":      run_data["n_bad_k"],
        "n_very_bad_k": run_data["n_very_bad_k"],
        "blocks":       run_data["blocks"],        # {cluster: block list}
        "timestamp":    run_data["timestamp"],
    }
    with open(os.path.join(output_dir, f"{tag}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Saved → {tag}.npz + {tag}.json")


# ── Main loop ─────────────────────────────────────────────────────────────────
for K_init in K_values:
    for seed in seeds:
        tag = f"K{K_init}_seed{seed}"
        print(f"\n{'='*55}")
        print(f"  K_init={K_init}  seed={seed}")
        print(f"{'='*55}")

        # Resume: skip if already saved
        if os.path.exists(os.path.join(output_dir, f"{tag}.json")):
            print(f"  Already exists — skipping")
            continue

        # ── Spectral init ─────────────────────────────────────────────────
        clusters, z_init = init_spectral_with_z(
            rankings=rankings,
            n_clusters=K_init,
            py_sampling=False,
            gamma=1,
            delta=0.5,
            seed=seed,
        )

        # ── Fit ───────────────────────────────────────────────────────────
        model = MixtureRankingModel(
            rankings=rankings,
            init_clusters=clusters,
            init_z=z_init,
            init_mu=[1.0] * K_init,
            seed=seed,
            verbose=False,
            init_theta=1.0,
        )

        _, samples = model.run_mcmc(
            n_iter=n_iter,
            burn_in=burn_in,
            thin=thin,
            use_py_prior=False,
            include_order_prior=False,
            save_samples=True,
            save_tau=True,
            save_theta=True,
            n_item_moves_per_cluster=1,
            a_theta=2,
            b_theta=1,
            save_log_likelihood=True,
            theta_jump=10,
            ranking_jump=1,
            use_annealing=True,
            temp_min=0.1,
            temp_max=1.0,
        )

        # ── LOO ───────────────────────────────────────────────────────────
        loo      = compute_loo(samples)
        pareto_k = loo.pareto_k.values

        # ── MAP + blocks + z trace ────────────────────────────────────────
        z_map, blocks, z_trace, K_actual = extract_map(model, samples, K_init)

        print(f"  K_init={K_init} → K_actual={K_actual}")
        print(f"  ELPD={loo.elpd:.1f} (SE={loo.se:.1f}), p_loo={loo.p:.1f}")
        print(f"  Pareto-k: {(pareto_k < loo.good_k).sum()} good, "
              f"{(pareto_k >= loo.good_k).sum()} bad, "
              f"{(pareto_k >= 1.0).sum()} very bad")

        # ── Save ──────────────────────────────────────────────────────────
        save_run({
            "K_init":       K_init,
            "K_actual":     K_actual,
            "seed":         seed,
            "elpd":         float(loo.elpd),
            "se":           float(loo.se),
            "p_loo":        float(loo.p),
            "good_k":       float(loo.good_k),
            "n_bad_k":      int((pareto_k >= loo.good_k).sum()),
            "n_very_bad_k": int((pareto_k >= 1.0).sum()),
            "elpd_i":       loo.elpd_i.values,
            "pareto_k":     pareto_k,
            "z_map":        z_map,
            "z_trace":      z_trace,
            "blocks":       blocks,
            "timestamp":    datetime.now().isoformat(),
        }, output_dir)


print(f"\n{'='*55}")
print("  All runs complete.")
print(f"{'='*55}")


# ── Loading utilities ─────────────────────────────────────────────────────────

def load_run(output_dir, K_init, seed):
    """Load a single run by K_init and seed."""
    tag    = f"K{K_init}_seed{seed}"
    arrays = np.load(os.path.join(output_dir, f"{tag}.npz"), allow_pickle=True)
    with open(os.path.join(output_dir, f"{tag}.json")) as f:
        meta = json.load(f)
    return {
        **meta,
        "z_map":    arrays["z_map"],     # (N,)
        "z_trace":  arrays["z_trace"],   # (T, N)
        "elpd_i":   arrays["elpd_i"],    # (N,)
        "pareto_k": arrays["pareto_k"],  # (N,)
    }


def load_all_runs(output_dir):
    """Load all completed runs, sorted by K_init then seed."""
    runs = []
    for fname in sorted(os.listdir(output_dir)):
        if not fname.endswith(".json"):
            continue
        parts  = fname.replace(".json", "").split("_")
        K_init = int(parts[0][1:])
        seed   = int(parts[1][4:])
        runs.append(load_run(output_dir, K_init, seed))
    return runs


def summarise_runs(output_dir):
    """Print a quick comparison table across all saved runs."""
    runs = load_all_runs(output_dir)
    print(f"\n{'K_init':>8} {'K_actual':>9} {'seed':>7} "
          f"{'ELPD':>10} {'SE':>8} {'p_loo':>7} "
          f"{'bad_k':>7} {'very_bad':>9}")
    print("-" * 75)
    for r in runs:
        print(f"{r['K_init']:>8} {r['K_actual']:>9} {r['seed']:>7} "
              f"{r['elpd']:>10.1f} {r['se']:>8.1f} {r['p_loo']:>7.1f} "
              f"{r['n_bad_k']:>7} {r['n_very_bad_k']:>9}")