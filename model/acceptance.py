"""
Acceptance probability tracking and analysis for TiedMallows MCMC.

This module provides functions to compute and analyze acceptance rates
from MCMC samples produced by MixtureRankingModel.
"""

from typing import Dict, Any, Optional, List
import math


def _saved_theta_update_iterations(samples) -> Optional[int]:
    """Count saved draws where a theta update was actually attempted."""
    saved_iterations = getattr(samples, "saved_iterations", None)
    theta_jump = max(1, getattr(samples, "theta_jump", 1))
    if not saved_iterations:
        return None
    return sum(1 for it in saved_iterations if it % theta_jump == 0)


def acceptance_probabilities(
    samples,
    n_clusters: int,
    parameter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute and return acceptance probabilities/rates for MCMC moves.
    
    Acceptance probabilities are estimated from the log-joint trace during MCMC.
    A move is considered "accepted" if the log-joint increased after the move
    (this is a practical approximation when explicit MH tracking is unavailable).
    
    Parameters
    ----------
    samples : MCMCSamples
        Samples object returned from MixtureRankingModel.run_mcmc()
    n_clusters : int
        Number of clusters C in the model
    parameter : str, optional
        Filter results to a specific parameter type. Possible values:
        - None (default): return acceptance rates for all move types
        - "theta": acceptance rate for theta (MH) updates
        - "blocks": combined acceptance rate for all block moves  
        - "split_merge": placeholder (requires detailed tracking)
        - "item_transfer": placeholder (requires detailed tracking)
        - "ordering": placeholder (requires detailed tracking)
        - "gibbs": acceptance rate for Gibbs moves (always 1.0 by definition)
    
    Returns
    -------
    dict
        Comprehensive dictionary with acceptance rate information.
        Keys include:
        - "overall_acceptance_rate": float in [0,1]
        - "theta": dict with theta-specific statistics (if available)
        - "blocks": dict with block-move statistics (if available)
        - "n_iterations": total number of saved iterations
        
    Notes
    -----
    For more detailed per-move-type acceptance tracking, the MH functions
    would need to be modified to return log_acc values. This function
    provides a practical approximation based on state changes.
    
    Examples
    --------
    >>> model = MixtureRankingModel(rankings, n_clusters=3)
    >>> state, samples = model.run_mcmc(5000, burn_in=1000, thin=5, save_logp=True)
    >>> 
    >>> # Get overall acceptance rates
    >>> acc_stats = acceptance_probabilities(samples, n_clusters=3)
    >>> print(f"Overall acceptance rate: {acc_stats['overall_acceptance_rate']:.2%}")
    >>> 
    >>> # Get theta-specific acceptance
    >>> theta_acc = acceptance_probabilities(samples, n_clusters=3, parameter="theta")
    >>> print(theta_acc)
    >>> 
    >>> # Get block move acceptance
    >>> block_acc = acceptance_probabilities(samples, n_clusters=3, parameter="blocks")
    >>> print(block_acc)
    """
    if samples is None:
        raise RuntimeError("Samples cannot be None. Run run_mcmc(...,save_samples=True) first.")
    
    results = {
        "note": "Acceptance statistics. Prefer exact tracking (theta_accepts/block_accepts) when available.",
    }

    # Prefer detailed per-proposal counts when available
    if getattr(samples, "theta_accept_counts", None) and getattr(samples, "theta_proposals", None):
        Tp = len(samples.theta_proposals)
        if Tp > 0:
            per_cluster_theta = []
            for c in range(n_clusters):
                n_props = sum(samples.theta_proposals[t][c] for t in range(Tp))
                n_accepts = sum(samples.theta_accept_counts[t][c] for t in range(Tp))
                acc_rate = (n_accepts / n_props) if n_props > 0 else None
                per_cluster_theta.append({
                    "cluster": c,
                    "n_proposals": n_props,
                    "n_accepts": n_accepts,
                    "acceptance_rate": acc_rate,
                })
            results["theta"] = {
                "per_cluster": per_cluster_theta,
                "overall_mean_acceptance": sum(x["acceptance_rate"] for x in per_cluster_theta if x["acceptance_rate"] is not None) / len([x for x in per_cluster_theta if x["acceptance_rate"] is not None]) if per_cluster_theta else None,
                "n_iterations": Tp,
            }
    # Fallback to coarse per-snapshot 0/1 indicators
    elif getattr(samples, "theta_accepts", None):
        T_theta = len(samples.theta_accepts)
        if T_theta > 0:
            theta_update_iterations = _saved_theta_update_iterations(samples)
            denom = theta_update_iterations if theta_update_iterations is not None else T_theta
            per_cluster_theta = []
            for c in range(n_clusters):
                n_accepts = sum(samples.theta_accepts[t][c] for t in range(T_theta))
                per_cluster_theta.append({
                    "cluster": c,
                    "n_accepts": n_accepts,
                    "acceptance_rate": (n_accepts / denom) if denom > 0 else None,
                })
            results["theta"] = {
                "per_cluster": per_cluster_theta,
                "overall_mean_acceptance": sum(x["acceptance_rate"] for x in per_cluster_theta if x["acceptance_rate"] is not None) / len([x for x in per_cluster_theta if x["acceptance_rate"] is not None]) if per_cluster_theta else None,
                "n_iterations": denom,
            }

    if getattr(samples, "block_accept_counts", None) and getattr(samples, "block_proposals", None):
        Tb = len(samples.block_proposals)
        if Tb > 0:
            per_cluster_blk = []
            for c in range(n_clusters):
                n_props = sum(samples.block_proposals[t][c] for t in range(Tb))
                n_accepts = sum(samples.block_accept_counts[t][c] for t in range(Tb))
                acc_rate = (n_accepts / n_props) if n_props > 0 else None
                per_cluster_blk.append({
                    "cluster": c,
                    "n_proposals": n_props,
                    "n_accepts": n_accepts,
                    "acceptance_rate": acc_rate,
                })
            results["blocks"] = {
                "per_cluster": per_cluster_blk,
                "overall_mean_acceptance": sum(x["acceptance_rate"] for x in per_cluster_blk if x["acceptance_rate"] is not None) / len([x for x in per_cluster_blk if x["acceptance_rate"] is not None]) if per_cluster_blk else None,
                "n_iterations": Tb,
            }
    # Fallback to coarse per-snapshot 0/1 indicators
    elif getattr(samples, "block_accepts", None):
        T_blk = len(samples.block_accepts)
        if T_blk > 0:
            per_cluster_blk = []
            for c in range(n_clusters):
                n_accepts = sum(samples.block_accepts[t][c] for t in range(T_blk))
                per_cluster_blk.append({
                    "cluster": c,
                    "n_accepts": n_accepts,
                    "acceptance_rate": n_accepts / T_blk,
                })
            results["blocks"] = {
                "per_cluster": per_cluster_blk,
                "overall_mean_acceptance": sum(x["acceptance_rate"] for x in per_cluster_blk) / len(per_cluster_blk),
                "n_iterations": T_blk,
            }

    # Fallback: logp-based improvement heuristic (less specific)
    if samples.logp is not None and len(samples.logp) >= 2:
        logp_trace = samples.logp
        T = len(logp_trace)
        increases = sum(1 for i in range(1, T) if logp_trace[i] > logp_trace[i - 1])
        results["logp_improvement"] = {
            "n_iterations": T - 1,
            "n_improvements": increases,
            "improvement_rate": increases / (T - 1),
        }
    
    # (Finished building results. Preference: use explicit accept indicators
    # `theta_accepts` and `block_accepts` saved in samples.  We keep the
    # logp_improvement entry as a lightweight diagnostic.)
    
    # Handle parameter-specific requests
    if parameter is not None:
        parameter_lower = parameter.lower().strip()
        
        if parameter_lower == "theta":
            if "theta" in results:
                th = results["theta"]
                return {
                    "parameter": "theta",
                    "per_cluster": th.get("per_cluster", []),
                    "overall_mean_acceptance": th.get("overall_mean_acceptance"),
                    "n_iterations": th.get("n_iterations"),
                }
            else:
                return {"error": "No theta acceptance data found (run with save_samples=True to collect)."}

        elif parameter_lower == "blocks":
            if "blocks" in results:
                bl = results["blocks"]
                return {
                    "parameter": "blocks",
                    "per_cluster": bl.get("per_cluster", []),
                    "overall_mean_acceptance": bl.get("overall_mean_acceptance"),
                    "n_iterations": bl.get("n_iterations"),
                    "interpretation": "Combined: split/merge, item transfer, ordering, Gibbs"
                }
            else:
                return {"error": "No block acceptance data found (run with save_samples=True)."}
        
        elif parameter_lower == "gibbs":
            return {
                "parameter": "gibbs",
                "acceptance_probability": 1.0,
                "note": "Gibbs moves are always accepted by definition (collapsed Gibbs sampling)"
            }
        
        elif parameter_lower in ["split_merge", "item_transfer", "ordering"]:
            return {
                "parameter": parameter_lower,
                "status": "detailed tracking not available",
                "note": f"To track {parameter_lower} separately, modify MH functions to return log_acc",
                "workaround": "Use overall 'blocks' acceptance rate or 'theta' rate"
            }
        
        else:
            valid_params = ["theta", "blocks", "gibbs", "split_merge", "item_transfer", "ordering", None]
            raise ValueError(
                f"Unknown parameter: {parameter}. "
                f"Valid options: {valid_params}"
            )
    
    return results


def print_acceptance_summary(
    samples,
    n_clusters: int,
) -> None:
    """
    Print a nicely formatted summary of acceptance probabilities.
    
    Parameters
    ----------
    samples : MCMCSamples
        Samples from MixtureRankingModel.run_mcmc()
    n_clusters : int
        Number of clusters
    
    Examples
    --------
    >>> model = MixtureRankingModel(rankings, n_clusters=3)
    >>> state, samples = model.run_mcmc(5000, burn_in=1000)
    >>> print_acceptance_summary(samples, n_clusters=3)
    """
    stats = acceptance_probabilities(samples, n_clusters)
    
    print("\n" + "=" * 60)
    print("MCMC ACCEPTANCE SUMMARY")
    print("=" * 60)
    
    # logp improvement (fallback)
    if "logp_improvement" in stats:
        lip = stats["logp_improvement"]
        print(f"\nLog-posterior improvements: {lip.get('n_improvements', 0)}/{lip.get('n_iterations', 'N/A')}")
        print(f"Log-posterior improvement rate: {lip.get('improvement_rate', 0):.2%}")

    # Blocks acceptance
    if "blocks" in stats:
        blk = stats["blocks"]
        print(f"\nBlock moves acceptance (saved iterations = {blk.get('n_iterations', 'N/A')}):")
        for cinfo in blk.get("per_cluster", []):
            if "n_proposals" in cinfo:
                np = cinfo.get("n_proposals", 0)
                na = cinfo.get("n_accepts", 0)
                ar = cinfo.get("acceptance_rate")
                ar_str = f"{ar:.2%}" if ar is not None else "N/A"
                print(f"  Cluster {cinfo['cluster']}: {na}/{np} accepts, acceptance_rate={ar_str}")
            else:
                print(f"  Cluster {cinfo['cluster']}: {cinfo['n_accepts']} accepts, acceptance_rate={cinfo['acceptance_rate']:.2%}")
        mean_blk = blk.get('overall_mean_acceptance')
        if mean_blk is not None:
            print(f"  Mean across clusters: {mean_blk:.2%}")
        else:
            print("  Mean across clusters: N/A")

    # Theta acceptance
    if "theta" in stats:
        th = stats["theta"]
        print(f"\nTheta updates acceptance (theta-update iterations = {th.get('n_iterations', 'N/A')}):")
        for cinfo in th.get("per_cluster", []):
            if "n_proposals" in cinfo:
                np = cinfo.get("n_proposals", 0)
                na = cinfo.get("n_accepts", 0)
                ar = cinfo.get("acceptance_rate")
                ar_str = f"{ar:.2%}" if ar is not None else "N/A"
                print(f"  Cluster {cinfo['cluster']}: {na}/{np} accepts, acceptance_rate={ar_str}")
            else:
                ar = cinfo.get("acceptance_rate")
                ar_str = f"{ar:.2%}" if ar is not None else "N/A"
                print(f"  Cluster {cinfo['cluster']}: {cinfo['n_accepts']} accepts, acceptance_rate={ar_str}")
        mean_th = th.get('overall_mean_acceptance')
        if mean_th is not None:
            print(f"  Mean across clusters: {mean_th:.2%}")
        else:
            print("  Mean across clusters: N/A")
    
    print("\n" + "=" * 60 + "\n")
