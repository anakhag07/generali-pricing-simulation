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
- **Policies**: `ConstantPolicy`, `LinearPolicy`, `SoftmaxPolicy`
- **Gradient estimators**: `first_order`, `finite_difference`, `gauss_stein`, `stein_difference`, `spsa`

`finite_difference` is a deterministic coordinate-wise central-difference baseline
that uses `2 * dim(theta)` objective evaluations per gradient call.

Core API convention:
- `sample_states(rng, n, dim)` produces state batches with shape `(n, dim)`.
- `Policy.value/grad` and `Objective.value/grad` operate on 2D `x_batch` arrays.

## Documentation

Full API documentation, objective formulas, and configuration reference are
available in `docs/` (generated via pdoc).

## Data Sources

Four state-distribution modes are available, selected by config preset:

| Preset | State source | Objective |
|---|---|---|
| `fixed_regression_base` | Synthetic N(0, I) | `FixedRegressionObjective` |
| `real_data_glm_base` | First 5K rows of raw acceptance CSV | `ModelBasedObjective` (GLM bundle, analytical grad) |
| `real_data_glm_linear_base` | First 5K rows of raw acceptance CSV | `ModelBasedObjective` (GLM bundle, linear-policy diagnostic) |
| `real_data_glm_linear_acceptance_floor_base` | First 5K rows of raw acceptance CSV | `ModelBasedObjective` (GLM bundle, linear policy + mean-acceptance floor) |
| `real_data_xgb_base` | First 5K rows of raw acceptance CSV | `ModelBasedObjective` (XGBoost bundle, FD grad) |
| `real_data_xgb_linear_acceptance_floor_base` | First 5K rows of raw acceptance CSV | `ModelBasedObjective` (XGBoost bundle, linear policy + mean-acceptance floor) |

The objective for real-data configs is $$f(u; x) = a(x,u)(\hat{Y}(x) - u \cdot p(x))$$
where $$a$$ is acceptance probability, $$\hat{Y}$$ is expected financial loss, and $$p$$ is policy premium.

Real-data artifacts now live under `src/data/artifacts_preproc_pipeline/` and each
pickle bundles the fitted estimator with its saved `FeatureProcessor`. The
objective keeps raw CSV rows at the optimization boundary and reuses the
acceptance bundle's saved preprocessing internally for both `u(theta, x)` and
`du/dtheta`.

`ExperimentConfig` also supports a smooth mean-acceptance floor via
`acceptance_floor`, `acceptance_penalty_weight`, and
`acceptance_penalty_temperature`. This is implemented as a differentiable
penalty on `ModelBasedObjective` while keeping the SciPy `L-BFGS-B` solver.

## Creating Config Presets

Use `src/experiments/configs/config_template.py` as a scaffold when creating a
new preset. Fill in the `None` placeholders, save it as a new module under
`src/experiments/configs/`, and register that module in
`src/experiments/configs/__init__.py`.

## Outputs

Each run writes artifacts to `outputs/<experiment_name>/<timestamp>/`:

- `summary.json` -- full result payload
- `steps.csv` -- per-step metrics for every estimator
- `plots/` -- loss curves, gradient norms, objective slices, contour plots

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
