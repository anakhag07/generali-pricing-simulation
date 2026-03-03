# Generali Pricing Simulation

Pricing simulation and optimization demo using Stein gradient estimators.

## Quickstart

```bash
conda create -n simulation_env python=3.11
conda activate simulation_env
pip install -e .
python main.py
```

Or, using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python main.py
```

Runtime dependencies live in `requirements.txt` and mirror `pyproject.toml`.

To run tests:

```bash
conda activate simulation_env
pytest -q
```

Or, from a virtual environment:

```bash
pip install -e ".[dev]"
pytest
```

If you use a different environment name or tool, update `AGENTS.md` to match your local setup.

## What This Does

- Samples synthetic customer states and contract actions.
- Evaluates a deterministic objective based on acceptance probability and expected loss.
- Runs first-order and zeroth-order Stein gradient estimators to optimize a pricing action.
- Runs an L-BFGS-B baseline over policy theta using SciPy for comparison.
- Runs a fixed objective with explicit acceptance, loss, and revenue.
- Saves run artifacts under `runs/<experiment_name>/<timestamp>/`, including `summary.json` and matplotlib plots in `plots/`.

## Minimization Model

This repo models a pricing action multiplier `u` applied to a baseline price `p` for a customer with features `x`.

```text
Decision variable:  u in [0.5, 1.5]
Customer features:  x
Baseline price:     p

Revenue:            h(p, u) = p * u
Acceptance:         a(x, u) in (0, 1)
Expected loss:      l(x) >= 0

Objective:          f(u; x) = a(x, u) * ( l(x) - h(p, u) )
Goal (demo):        minimize f(u; x) using deterministic queries
```

The demo samples a batch of customer states and optimizes the average objective over that batch.

## Fixed Regression Objective

The objective uses an explicit parametric form.

```text
Acceptance: a(x, u) = sigmoid(beta_1^T x + beta_2 * u)
Loss:       l(x) = beta_3^T x
Revenue:    r(u) = beta_4 * u

Objective:  f(u; x) = a(x, u) * ( l(x) - r(u) )
```

The beta values are configurable via `FixedRegressionObjective` (set in
`ExperimentConfig.objective_model`). `beta_1` and `beta_3` must be
positive, `beta_4` must be positive, and `beta_2` must be negative so acceptance
decreases with higher policy values. The demo plots the objective and gradient over the
action grid when this objective is used.

## Planted Logistic Objective

For estimator comparisons with a known optimum, use the planted logistic objective:

```text
z(u, x) = alpha * u + beta^T x + bias
p*(x) = sigmoid(alpha * u* + beta^T x + bias)
L(u; x) = log(1 + exp(z)) - p*(x) * z
```

This function is convex in `u` and has a known minimum at `u*` for every `x`.
The preset config `src/experiments/configs/planted_logistic.py` wires this in and
exposes `u*` to the logs and plots.

## Experiment Configuration

Configs live in `src/experiments/configs/`. Edit `custom.py` for the most recent run and
set which presets to execute in `main.py` by updating `RUN_CONFIGS`.

Each `ExperimentConfig` includes the required state dimension `state_dim`, a required
`n_samples` batch size for customer states, a policy specification, and an
`objective_model` (for example, `FixedRegressionObjective`). State sampling draws each
feature from a standard normal distribution.

Step-size behavior is controlled by `step_rule`, which must be explicitly set to
`"constant"` or `"armijo"`. The `step_size` field is the constant step size for
`"constant"` and the initial step size for Armijo backtracking. When `step_rule` is not
`"constant"`, the demo also saves a `step_sizes.png` plot of the per-iteration step sizes.

Use `enabled_estimators` in the config to control which optimization methods run (and
which curves/paths appear in plots and logs). For example:

```python
enabled_estimators=("zeroth_order", "first_order", "lbfgs")
```

## Model-to-Code Mapping

```text
x (customer features)            -> StateVector (src/data/models.py)
customer                          -> Customer (src/data/models.py)
u bounds / projection             -> U_BOUNDS, clip_u (src/optimization/common.py)
u as contract action              -> Contract(u=...) (src/data/models.py)
a(x, u) acceptance probability    -> FixedRegressionAcceptance (src/data/fixed_objective.py)
l(x) expected loss                -> FixedRegressionLoss (src/data/fixed_objective.py)
h(p, u) revenue                   -> FixedRegressionRevenue (src/data/fixed_objective.py)
f(u; x) objective                 -> FixedRegressionObjective (src/data/fixed_objective.py)
oracle gradient API               -> FixedRegressionObjective.evaluate (src/data/fixed_objective.py)
policy u = f(theta, x)            -> PolicySpec, apply_policy (src/optimization/policy.py)
experiment runner / config        -> ExperimentConfig, run_experiment (src/experiments/run.py)
```

## Optimization Methods Used

- First-order Stein estimator: uses the explicit gradient to estimate gradients of a smoothed objective.
- Zeroth-order Stein estimator: uses only objective evaluations at perturbed actions.
- L-BFGS-B baseline: uses SciPy's optimizer to minimize the theta-level objective.

Because `u` is clipped to `[0.5, 1.5]`, sufficiently large gradient steps can push iterates to the bounds.

## Reproducibility

The demo uses a fixed RNG seed in `main.py` to make runs repeatable. The objective is deterministic given a fixed configuration and state sample batch.

## Project Structure

- `main.py`: demo entry point.
- `src/data/models.py`: data classes and objective interfaces.
- `src/data/fixed_objective.py`: fixed regression objective implementation.
- `src/experiments/config.py`: experiment configuration interface.
- `src/experiments/defaults.py`: default helpers for experiment presets.
- `src/experiments/helpers.py`: optimization helper routines.
- `src/experiments/run.py`: experiment runner entry (returns results, no I/O).
- `src/experiments/results.py`: experiment result data structures.
- `src/experiments/reporters.py`: console/plot/json reporters and run directories.
- `src/experiments/configs/`: preset configurations (including `custom.py`).
- `src/experiments/logging.py`: console logging helpers.
- `src/experiments/visualization.py`: matplotlib plotting utilities.
- `src/optimization/gradients/`: first-order and zeroth-order Stein estimators.
- `src/optimization/policy.py`: policy specs (softmax policy used by default in the demo).
