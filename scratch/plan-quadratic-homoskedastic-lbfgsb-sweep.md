# Quadratic Homoskedastic L-BFGS-B Sweep Plan

## Goal

Study how deterministic homoskedastic value noise affects SciPy L-BFGS-B on
the policy-free quadratic benchmark

$$
J(\theta) = \frac{1}{2}\|\theta\|_2^2,
$$

using the noisy value oracle

$$
\widehat J_s(\theta) = J(\theta) + \tau\,\varepsilon_s(\theta).
$$

Here, $\varepsilon_s(\theta) \sim N(0,1)$ is deterministically keyed by the
exact float64 representation of $\theta$ and the experiment noise seed $s$.
Repeated evaluations at the same $\theta$ therefore return the same value,
matching the reproducibility semantics of the existing action-level noise
field.

The first experiment should isolate noise magnitude and finite-difference
radius. Dimension and L-BFGS-B settings remain fixed.

## Gradient Supplied To L-BFGS-B

L-BFGS-B requires an objective value and a gradient. Use the repository's
`finite_difference` estimator to construct an explicit central-difference
gradient and pass it as `jac=` to
`scipy.optimize.minimize(method="L-BFGS-B")`:

$$
\widehat g_i(\theta)
=
\frac{
  \widehat J(\theta + h e_i)
  - \widehat J(\theta - h e_i)
}{2h}.
$$

The finite-difference radius $h$ is not an L-BFGS-B parameter. It controls the
gradient estimate supplied to L-BFGS-B. With independently hash-keyed noise at

$$
\operatorname{sd}(\widehat g_i - g_i)
= \frac{\tau}{\sqrt{2}\,h}.
$$

The ratio $\tau/h$ is therefore the expected signal-to-noise control variable.
For this exact quadratic, central differences have no truncation bias in exact
arithmetic, so increasing $h$ isolates noise suppression rather than the usual
bias-variance tradeoff.

Do not use SciPy's internal `jac=None` finite-difference path in this first
sweep. That would introduce a second Jacobian implementation and require
exposing SciPy-specific finite-difference step options.

## Fixed Configuration

- Base preset: `quadratic_base`
- Dimension: $d=10$
- Initial point: $\theta_0 = \mathbf{1}/\sqrt{d}$
- Initial norm: $\|\theta_0\|_2=1$
- Initial clean objective: $J(\theta_0)=0.5$
- Estimator: `finite_difference` only
- Perturbation space: `theta`
- Step rule: `l-bfgs-b`
- Maximum iterations: `t_steps=200`
- Gradient tolerance: `grad_norm_tol=1e-8`
- Function tolerance: `ftol=1e-12`
- Dummy sample count: `n_samples=1`; the quadratic ignores `x_batch`
- Per-run plots: disabled
- W&B: disabled
- Correctness source: `denoised_exact`
- Bounds: none, matching the current optimizer integration

Although the SciPy method is named L-BFGS-B, the repository currently passes
no `bounds=` argument. This experiment is therefore unconstrained.

## Full Grid

Finite-difference radii:

```text
h in {1e-4, 1e-3, 1e-2, 1e-1}
```

Homoskedastic noise standard deviations:

```text
tau in {0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2}
```

Run seeds:

```text
7, 8, ..., 26
```

Use 20 replicates per cell. Keep theta, data, split, and optimizer streams
fixed and vary only the noise stream. The coordinate finite-difference
estimator has no optimizer RNG, so varying `optimizer_seed` would not add
variation.

The full grid contains $4 \times 8 \times 20 = 640$ inexpensive runs.

Use variant names of the form

```text
noise-std-<tau>__fd-radius-<h>
```

and write results under

```text
results/quadratic-homoskedastic-lbfgsb-sweep/
```

## Pilot

Before the full run, optionally execute a 60-run calibration pilot:

```text
h in {1e-3, 1e-2, 1e-1}
tau in {0, 1e-6, 1e-4, 1e-2}
run seeds in {7, 8, 9, 10, 11}
```

The pilot should confirm that the grid spans clean convergence, intermediate
degradation, and line-search failure or noise-dominated behavior. Adjust only

## Metrics

Primary metrics:

- Final clean objective $J(\widehat\theta)$
- Final parameter error $\|\widehat\theta\|_2$

Secondary metrics:

- Final noisy objective seen by the optimizer
- Final exploitation gap $\widehat J(\widehat\theta)-J(\widehat\theta)$
- Clean improvement from the fixed initial value $0.5$
- Runtime and trace length
- `optimizer_success`
- Optimizer status and termination message
- Failure rate over seeds

For this objective,

$$
J(\widehat\theta) = \frac{1}{2}\|\widehat\theta\|_2^2,
$$

so the two primary metrics also provide an internal consistency check.
Do not filter runs based on `optimizer_success`; line-search failure under noise
is itself an outcome of interest.

## Aggregate Outputs

Write one detailed CSV with a row per `(noise_std, fd_radius, run_seed)` and one
summary CSV with mean, standard deviation, minimum, maximum, median, and success
rate per grid cell.

Recommended plots:

1. Heatmap of median final $\|\widehat\theta\|_2$ over noise standard deviation
   and finite-difference radius.
2. Heatmap of L-BFGS-B success rate over the same grid.
3. Log-log collapse plot of final parameter error against $\tau/h$, with one
   curve or marker style per $h$.
4. Clean final objective versus noisy final objective to expose optimistic
   noise exploitation.

Use log-scaled axes where applicable. Handle the clean $\tau=0$ baseline
separately on plots whose x-axis is $\tau/h$.

## Lowest-Code Implementation

### Theta-Keyed Noise

Extend `src/objective/noise.py` rather than introducing a quadratic-specific
noise objective.

1. Add a theta-value hook to the existing noise seam.
2. Implement it for `NoNoise` and `HomoskedasticGaussianNoise` using a stable
   hash of `(noise_seed, exact theta bytes)`.
3. In `NoisyObjective.value()`, preserve the existing action-level path when a
   policy exists. For a policy-free objective, add one theta-keyed noise value
   to the clean objective.
4. Keep `base_value()` unchanged so final summaries report the clean objective.
5. Keep `grad()` unavailable on the noisy wrapper and use the wrapped
   quadratic gradient for `denoised_exact` diagnostics.
6. Reject heteroskedastic action noise for policy-free objectives with a clear
   error because an action center has no meaning there.

This reuses `NoisyObjective`, its serialization, `with_noise_seed()`, and the
existing experiment noise seed stream. No new seed stream is needed.

### Sweep Driver

Add `scripts/run_quadratic_homoskedastic_sweep.py` only when promoting this plan
to implementation.

The driver should:

1. Build explicit `NoisyObjective(QuadraticObjective(...), noise)` overrides.
2. Call canonical `experiments.sweep_utils.run_sweep(...)`.
3. Set `vary=("noise",)` with an anchored fixed initialization.
4. Disable heavy per-seed plots.
5. Collect the richer quadratic metrics from saved summaries.
6. Support `--plots-only`, custom seeds, a pilot mode, and dimension override.

Do not add reusable sweep behavior to `src/`; the grid and its plots are a
one-off scientific experiment.

## Tests

Proposed focused tests:

- `tests/objective/test_noise.py`
  - Same theta and seed return the same noise.
  - Changing theta or seed changes the noise.
  - Zero noise works without a seed.
  - `NoisyObjective(QuadraticObjective(...))` reports noisy optimization values
    and clean `base_value()` values.
  - The noisy wrapper still has no analytical gradient.

- `tests/experiments/test_policy_free_objective.py`
  - A noisy quadratic runs with theta-space finite differences.
  - `denoised_exact` records the clean quadratic gradient norm.
  - Repeating an identical seeded run gives identical output.

- `tests/scripts/test_quadratic_homoskedastic_sweep.py`
  - Grid construction covers the noise-by-radius product.
  - Variant naming round-trips.
  - Only the noise seed varies across replicates.
  - Final clean objective, theta norm, and exploitation gap are reconstructed
    correctly.
  - Aggregation includes failed optimizer runs.

Run a reduced end-to-end smoke sweep and verify that all clean-noise cells
converge near zero for every finite-difference radius.

## Documentation And Delivery

When implemented:

- Update the noisy-objective formula and theta-key semantics in `MATH.md`.
- Document the new script in `README.md` and `AGENTS.md`.
- Regenerate pdoc output because public noise docstrings will change.
- Use a stacked feature branch based on `feature/quadratic-objective`; this work
  touches reusable objective/noise behavior and is worktree-required.
- Prefer separate commits for theta-noise support, sweep/reporting, and
  tests/documentation.
