# Generali Pricing Simulation

Pricing simulation and black-box optimization demo with pluggable objectives,
policies, and gradient estimators.

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
- **Objectives**: `FixedRegressionObjective`, `PlantedLogisticObjective`
- **Policies**: `ConstantPolicy`, `LinearPolicy`, `SoftmaxPolicy`
- **Gradient estimators**: `first_order`, `gauss_stein`, `spsa`

## Documentation

Full API documentation, objective formulas, and configuration reference are
available in `docs/` (generated via pdoc).

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
   (follow `GaussSteinGradient` / `SPSAGradient`).
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
