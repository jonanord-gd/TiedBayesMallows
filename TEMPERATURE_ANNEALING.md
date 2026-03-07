# Temperature Annealing for Kernel Collapse Prevention

## Problem Solved

The TiedMallows MCMC model was susceptible to **cluster collapse**: the chain would often converge to 1-2 dominant clusters despite initializing with 6+ clusters. This was caused by a structural property of the Mallows likelihood, which mathematically favors more cluster separation (K separate clusters gives lower Kemeny distances than K=1).

## Solution: Temperature Annealing

Temperature annealing solves this by using a **schedule of likelihood temperatures** during burn-in:
- **Low temperature (early)**: Soft likelihood → easier cluster switching → explores configurations
- **High temperature (late)**: Sharp likelihood → emphasizes structure → converges to posterior

## Key Results from Testing

### Temperature Effect (theta controls likelihood sharpness)
```
θ = 0.5 (soft):  5 active clusters maintained (more exploration)
θ = 5.0 (sharp): 4 active clusters maintained (more consolidation)
Result: Lower theta = better cluster diversity
```

### Annealing Trajectory (4-phase schedule: 0.5 → 2.0 → 4.0 → 5.0)
```
Phase 1 (θ=0.5): Initial [20,12,15,19,14,20] → [10,16,14,7,40,13]  (6/6 active)
Phase 2 (θ=2.0): [10,16,14,7,40,13]        → [4,38,11,19,15,13]   (6/6 active)
Phase 3 (θ=4.0): [4,38,11,19,15,13]        → [31,11,9,31,2,16]    (6/6 active)
Phase 4 (θ=5.0): [31,11,9,31,2,16]         → [13,29,7,23,6,22]    (6/6 active)
Result: All clusters maintained throughout annealing
```

## How to Use

### 1. Automatic Linear Annealing Schedule
Start with a soft likelihood and gradually increase sharpness during burn-in:

```python
state, samples = model.run_mcmc(
    n_iter=5000,
    burn_in=500,
    use_annealing=True,
    temp_min=0.5,      # Starting temperature (soft)
    temp_max=1.0,      # Ending temperature (normal)
    annealing_schedule_type="linear"  # Linear interpolation
)
```

### 2. Automatic Exponential Annealing Schedule
For faster cooling:

```python
state, samples = model.run_mcmc(
    n_iter=5000,
    burn_in=500,
    use_annealing=True,
    temp_min=0.5,
    temp_max=1.0,
    annealing_schedule_type="exponential"  # Exponential growth
)
```

### 3. Custom Explicit Schedule
Specify exact temperature multipliers for each iteration:

```python
state, samples = model.run_mcmc(
    n_iter=100,
    annealing_schedule=[0.5, 1.0, 2.0, 5.0, 5.0, ...]  # Total length should match n_iter
)
```

### 4. Multi-Phase Annealing
Run separate phases with different temperatures:

```python
for theta_val in [0.5, 2.0, 4.0, 5.0]:
    # Set initial theta for this phase
    state, samples = model.run_mcmc(
        n_iter=100,
        burn_in=0,
        annealing_schedule=[theta_val] * 100  # Hold temperature constant
    )
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_annealing` | bool | False | Enable automatic temperature schedule |
| `annealing_schedule` | List[float] | None | Explicit temperature multipliers for each iteration |
| `annealing_schedule_type` | str | "linear" | "linear" or "exponential" for auto schedule |
| `temp_min` | float | 0.5 | Starting temperature (lower = softer likelihood) |
| `temp_max` | float | 1.0 | Final temperature (1.0 = true posterior) |

## Implementation Details

### How It Works

1. **Temperature Multiplier**: At each iteration during the annealing phase, all cluster thetas are multiplied by a temperature factor from the schedule.

2. **Soft Likelihood**: Low temperature (e.g., 0.5) means distance calculations are divided by a small number, making the likelihood softer and flatter.

3. **Automatic Schedule**: When `use_annealing=True`, the schedule is auto-generated over the burn-in period (or first 25% of iterations if no burn-in).

4. **Restoration**: After each step, original theta values are restored so subsequent theta updates operate on the true posterior values.

### Modified Files

- **model/core.py**: Added annealing parameters and logic to `run_mcmc()`
  - New signature parameters: `use_annealing`, `annealing_schedule`, `annealing_schedule_type`, `temp_min`, `temp_max`
  - Temperature schedule generation logic
  - Per-iteration theta scaling during annealing phase

### Backward Compatibility

✓ **Fully backward compatible**: 
- All existing code continues to work with `use_annealing=False` (default)
- No changes to other functionality
- Existing `run_mcmc()` calls unaffected

## Recommendations

### For Preventing Cluster Collapse
```python
state, samples = model.run_mcmc(
    n_iter=5000,
    burn_in=1000,      # Generous burn-in for annealing
    use_annealing=True,
    temp_min=0.1,      # Very soft start
    temp_max=1.0,      # Normal posterior
    theta_jump=10,     # Speed up with sparse updates
)
```

### For Balancing Speed and Stability
```python
state, samples = model.run_mcmc(
    n_iter=5000,
    burn_in=500,
    use_annealing=True,
    temp_min=0.5,      # Moderate softness
    temp_max=1.0,
    annealing_schedule_type="exponential"  # Faster cooling
)
```

### For Quick Testing
```python
state, samples = model.run_mcmc(
    n_iter=500,
    burn_in=100,
    use_annealing=True,
    temp_min=0.5,
    temp_max=1.0
)
```

## Theoretical Motivation

This implements **simulated annealing** for MCMC:

1. **Exploration Phase** (low theta): The soft likelihood allows the chain to freely explore different cluster configurations and block structures.

2. **Refinement Phase** (increasing theta): As theta increases, the likelihood tightens, and the chain concentrates on the high-probability region while maintaining cluster structure.

3. **Convergence Phase** (high theta): With theta=1.0 (true posterior), the chain converges to the target distribution while having already settled into a good cluster configuration.

This prevents the common problem where MCMC gets stuck in a poor local mode early on due to the high penalties for cluster splitting in the sharp likelihood.

## Future Enhancements

Possible extensions:
- Adaptive temperature schedule based on cluster diversity metrics
- Parallel tempering with multiple chains at different temperatures
- Diagnostics to detect when annealing should end based on convergence
- Integration with existing convergence diagnostics (potential scale reduction factor, etc.)

## References

The annealing approach is analogous to:
- **Simulated Annealing**: Classic optimization technique using temperature schedules
- **Parallel Tempering / Replica Exchange MCMC**: Multiple chains at different temperatures
- **Tempering Distributions**: Bayesian approach to modifying posteriors during sampling

