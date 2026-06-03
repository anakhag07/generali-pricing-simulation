# Generali Pricing Simulation

Pricing simulation and black-box optimization demo with pluggable objectives,
policies, and gradient estimators. More detailed docs can be viewed [here](https://anakhag07.github.io/generali-pricing-simulation/index.html). 

## Quickstart

```bash
# Conda
conda create -n simulation_env python=3.11
conda activate simulation_env
pip install -e .
python main.py
```

Runtime dependencies live in `requirements.txt` and mirror `pyproject.toml`
(numpy >= 1.24, matplotlib >= 3.7, scipy >= 1.10, wandb >= 0.19).

To run tests:

```bash
pip install -e ".[dev]"
pytest -q
```

## What This Does

This project optimizes a parameterized policy over random state vectors:

$$
x \sim \mathcal{N}(0, I),\quad \theta \in \mathbb{R}^p,\quad u = \pi_\theta(x)
$$

The optimizer solves the theta-space objective:

$$
\min_{\theta} J(\theta),\qquad
J(\theta) = \mathbb{E}_x\big[f(\pi_\theta(x); x)\big]
$$

Pluggable components:
- **Objectives**: `FixedRegressionObjective`, `PlantedLogisticObjective`, `ModelBasedObjective`
- **Policies**: `ConstantPolicy`, `LinearPolicy`, `SoftmaxPolicy`, `MLPPolicy` (2-layer, default hidden=16)
- **Gradient estimators**: `first_order`, `finite_difference`, `gauss_stein`, `stein_difference`, `spsa`

The default bounded policy is `SoftmaxPolicy`, which maps
`u = 0.5 - sigma(theta^T phi(x))`, so its action range is `(-0.5, 0.5)`.
`LinearPolicy` and `SoftmaxPolicy` support configurable state feature maps
`varphi(x)`. The policy prepends the intercept internally, so
`phi(x) = [1, varphi(x)]` and custom feature maps should not include the
leading `1`. The default `IdentityFeatureMap` gives the previous behavior
`phi(x) = [1, x]`; `QuadraticFeatureMap` expands the state with linear,
square, and pairwise interaction terms. `CubicFeatureMap` and
`QuarticFeatureMap` follow the same pattern with linear terms plus exact
degree-3 or degree-4 monomials.
For the real-data model-based objective, this `u` remains centered and the
revenue term uses premium multiplier `u + 1`.

`finite_difference` is a deterministic coordinate-wise central-difference baseline
that uses `2 * dim(theta)` objective evaluations per gradient call.

Core API convention:
- `sample_states(rng, n, dim)` produces state batches with shape `(n, dim)`.
- `Policy.value/grad` and `Objective.value/grad` operate on 2D `x_batch` arrays.

Optimization step rules:
- `l-bfgs-b` uses `scipy.minimize(method="L-BFGS-B")`.
- `trust-constr` uses `scipy.minimize(method="trust-constr")` and adds the
  acceptance-floor equation directly as a nonlinear constraint. Optional
  `initial_constr_penalty` is passed through to SciPy when configured.
- `constant` uses the repo's manual gradient loop with fixed `step_size`.
- `armijo` uses the same manual loop with Armijo backtracking seeded by `step_size`.

`ExperimentConfig.constant_u_baselines` can be used to evaluate fixed-action
baselines such as `(-0.3, 0.0, 0.2)` on the same batch used for training and
show them in `summary.json`, the console summary, `loss_curves.png`, and
`objective_u_slice.png`.

## Documentation

Full API documentation, objective formulas, and configuration reference are
available in `docs/` (generated via pdoc). `MATH.md` provides a central
reference for all mathematical formulas implemented in `src/`, organized by
module layer.

## Data Sources

Several preset configs are available, selected by config preset:

| Preset | State source | Objective |
|---|---|---|
| `fixed_regression_base` | Synthetic N(0, I) | `FixedRegressionObjective` |
| `real_data_glm_softmax_policy_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, linear-feature softmax policy, analytical grad) |
| `real_data_glm_softmax_policy_lagrangian_small` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, softmax policy, analytical grad, lagrangian floor scalarization) |
| `real_data_glm_softmax_policy_quadratic_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, quadratic softmax policy, analytical grad) |
| `real_data_glm_softmax_policy_cubic_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, cubic softmax policy, analytical grad) |
| `real_data_glm_softmax_policy_quartic_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, quartic softmax policy, analytical grad) |
| `real_data_glm_softmax_policy_quartic_no_pca` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, quartic softmax policy, full sphered policy features, analytical grad) |
| `real_data_glm_mlp_policy_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, 2-layer MLP policy, hidden=16, analytical grad) |
| `real_data_glm_softmax_policy_trust_region_constr` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, softmax policy + trust-constr acceptance floor) |
| `real_data_glm_linear_policy_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, linear-policy diagnostic) |
| `real_data_glm_linear_policy_quadratic_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, quadratic-feature linear policy) |
| `real_data_glm_linear_policy_cubic_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, cubic-feature linear policy) |
| `real_data_glm_linear_policy_quartic_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, quartic-feature linear policy) |
| `real_data_glm_linear_policy_trust_region_constr` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, linear policy + trust-constr acceptance floor) |
| `real_data_glm_constant_policy_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, constant-policy diagnostic) |
| `real_data_glm_constant_policy_trust_region_constr` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (GLM bundle, constant policy + trust-constr acceptance floor) |
| `real_data_xgb_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (XGBoost bundle, FD grad) |
| `real_data_xgb_softmax_policy_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (XGBoost bundle, softmax policy, FD grad) |
| `real_data_xgb_linear_policy_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (XGBoost bundle, linear policy, FD grad) |
| `real_data_xgb_linear_acceptance_floor_base` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (XGBoost bundle, linear policy + penalty acceptance floor) |
| `real_data_xgb_softmax_policy_trust_region_constr` | Seeded `n_samples` draw from raw acceptance CSV | `ModelBasedObjective` (XGBoost bundle, softmax policy + trust-constr acceptance floor) |

The objective for real-data configs is $$f(u; x) = a(x,u)(\hat{Y}(x) - (u + 1) \cdot p(x))$$
where $$a$$ is acceptance probability, $$\hat{Y}$$ is expected financial loss, and $$p$$ is policy premium.

Real-data source rows now live in the canonical `src/data/dataset.csv` file,
with schema/path metadata tracked in `src/data/dataset_metadata.py`. The current
canonical CSV is the GLM acceptance export; both GLM and XGB real-data loaders
sample rows from it.
Model artifacts live under `src/data/models/linear/` and `src/data/models/xgb/`,
and each pickle bundles the fitted estimator with its saved `FeatureProcessor`.
The objective keeps raw CSV rows at the optimization boundary and reuses the
acceptance bundle's saved preprocessing internally for both `u(theta, x)` and
`du/dtheta`.
For policy-feature experiments, `ModelBasedObjective` can instead take a
separate fitted policy-side preprocessor. In that mode the policy sees the
configured policy features, while the sealed acceptance and loss model paths
still receive raw `x` and apply their saved artifact preprocessing internally.
Real-data configs sample `TRAINING["n_samples"]` rows from the acceptance CSV
with the experiment seed and store the sampled row indices so observed-`U`
diagnostics use the same source rows.
When plotting is enabled, real-data runs also write a final customer-level
diagnostic `policy_u_acceptance_histograms.png`: for each estimator it shows the
histogram of final policy actions across sampled customers plus binned mean
acceptance, alongside the per-customer acceptance-vs-`u` scatter.

`ExperimentConfig` supports two acceptance-floor paths for objectives exposing
`mean_acceptance(theta, x_batch)`:

- `step_rule="trust-constr"` enforces `mean_acceptance >= acceptance_floor`
  directly inside SciPy as a nonlinear constraint. This is the constrained GLM
  path used by the softmax and linear trust-region presets.
- penalty-based step rules such as `l-bfgs-b` use
  `acceptance_penalty_weight` and `acceptance_penalty_temperature`, which add a
  differentiable penalty to `ModelBasedObjective`. This remains the XGBoost
  floor path.
- lagrangian scalarization uses `lagrangian_lambda` together with
  `acceptance_floor`, optimizing
  `J(theta) + lagrangian_lambda * (acceptance_floor - mean_acceptance(theta))`.
  This path is available on unconstrained step rules and keeps experiment
  summaries on the raw objective `J(theta)` so lambda sweeps can be compared on
  the same frontier.

`scripts/run_lagrangian_sweep.py` runs a preset sweep over `lagrangian_lambda`
and writes three aggregate plots under `outputs/<project>/lagrangian_frontier_<timestamp>/`:

- `lambda_vs_u_acceptance.png` -- two panels for `lambda -> final u` and
  `lambda -> mean acceptance`
- `pareto_objective_acceptance.png` -- final objective vs acceptance, colored by
  lambda
- `pareto_u_acceptance.png` -- final `u` vs acceptance, colored by lambda

`scripts/run_acceptance_floor_sweep.py` runs the trust-constrained softmax GLM
preset over a dense acceptance-floor grid `c in [0.50, 0.995]` and writes the
same three aggregate plots under
`outputs/<project>/acceptance_floor_frontier_<timestamp>/`:

- `c_vs_u_acceptance.png` -- two panels for `c -> final u` and
  `c -> mean acceptance`
- `pareto_objective_acceptance.png` -- final objective vs acceptance, colored by
  `c`
- `pareto_u_acceptance.png` -- final `u` vs acceptance, colored by `c`

If you already have saved acceptance-floor sweep outputs and only want the
Pareto frontier for one estimator without rerunning optimization, use
`scripts/plot_saved_acceptance_floor_frontier.py`:

```bash
python scripts/plot_saved_acceptance_floor_frontier.py \
  outputs/glm-softmax-acceptance-floor-sweep
```

The script accepts either a direct `acceptance_floor_sweep.csv` path, a single
`acceptance_floor_frontier_<timestamp>/` directory, or the parent project output
directory. It defaults to `--estimator first_order` and writes
`pareto_objective_acceptance_first_order.png` plus
`pareto_u_acceptance_first_order.png` alongside the resolved CSV unless
`--output-dir` is provided.

To run the policy PCA-dimensionality grid over policy classes and policy-side
PCA dimensions, use:

```bash
python scripts/run_policy_pca_grid.py --n-samples 5000
```

This keeps the GLM black-box preprocessing sealed, fits configurable policy-side
preprocessors on the 10 acceptance-state columns, and writes aggregate finals,
traces, summary markdown, headline PCA/richness-gap plots, and final `u` /
acceptance-spread plots under `outputs/policy-pca-grid/`. The grid includes
linear-feature policies, matching softmax-wrapped feature policies, constant,
and MLP policies. The script prints per-condition progress by default; pass
`--quiet` to suppress progress output. Add `--constrained` to use `trust-constr`
with the observed GLM acceptance floor and a default 500-step cap.

To query the existing acceptance model at fixed constant actions without
running optimization, use:

```bash
python scripts/query_acceptance_at_u.py \
  --model-type glm \
  --u-count 101 \
  --n-rows 500
```

The script loads the preset objective and state batch, then reports mean
acceptance for each sampled `u` value. It writes acceptance curves and a
historical-`U` histogram with sampled constant-`u` rug marks under
`outputs/acceptance_queries/<model_type>/` by default. Use `--output-subdir`
to choose a subdirectory under `outputs/acceptance_queries/`, or pass explicit
values with `--u -0.3 0.0 0.2` instead of `--u-count`.

To diagnose how final real-data policies relate processed policy components to
acceptance, loss, and action variation, use a saved run `summary.json`:

```bash
python scripts/plot_pc_outcome_diagnostics.py \
  --preset real_data_glm_mlp_policy_base \
  --summary-json outputs/glm-policy-comparison/mlp/<run_id>/summary.json \
  --estimator first_order
```

This writes scatter grids for processed components vs `f_acc`, loss, and final
`u`, plus `u_vs_acceptance.png` and `pc_diagnostic_correlations.csv` beside the
run summary by default.

## Creating Config Presets

Use `src/experiments/configs/config_template.py` as a scaffold when creating a
new preset. Fill in the `None` placeholders, save it as a new module under
`src/experiments/configs/`, and register that module in
`src/experiments/configs/__init__.py`.

## Outputs

Each run writes artifacts to `outputs/<experiment_name>/<timestamp>/`:

- `summary.json` -- full result payload
  including final trust-constr diagnostics such as `constraint_penalty`
  and any configured constant-`u` baseline evaluations; estimator results
  include both the mean objective `final_value` and summed objective
  `final_objective_sum`
- `steps.csv` -- per-step metrics for every estimator
- `plots/` -- loss curves, gradient norms, objective slices, contour plots, and real-data diagnostics for observed-vs-policy `u` histograms plus `u` vs `M(x, u)` trend plots

Weights & Biases integration is available for experiment tracking. See the
docstrings in `src/experiments/config.py` for W&B configuration fields.

## Adding a New Zeroth-Order Method

To add a new value-query estimator and run it through experiments:

1. Add a `GradientMethod` class in `src/optimization/gradients/methods.py`
   (follow `FiniteDifferenceGradient` / `GaussSteinGradient` /
   `SteinDifferenceGradient` / `SPSAGradient`).
2. Re-export it in `src/optimization/gradients/__init__.py`.
3. Add a solver wrapper in `src/optimization/solvers.py` that instantiates
   `Optimization(..., <YourGradientMethod>(), ...)`.
4. Add a corresponding experiment helper in `src/experiments/helpers.py`, then
   call it from `src/experiments/run.py`.
5. Register the estimator key in `src/experiments/config.py`
   (`allowed_estimators` in `ExperimentConfig.__post_init__`).
6. Add plot metadata in `src/reporting/visualization.py`
   (`ESTIMATOR_STYLES` and `_TRACE_ORDER`) so it renders in plots.

Finally, include your estimator key in `enabled_estimators=(...)` in a preset
under `src/experiments/configs/`.

## Reproducibility

The demo uses a fixed RNG seed (default 7, configurable per
`ExperimentConfig.seed`). The objective is deterministic given a fixed
configuration and state sample batch.

## Contributing

See `AGENTS.md` for development workflow, code organization, and testing
guidelines.
