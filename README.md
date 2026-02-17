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

- Samples synthetic customer state and contract actions.
- Evaluates a deterministic objective based on acceptance probability and expected loss.
- Runs first-order and zeroth-order Stein gradient estimators to optimize a pricing action.
- Runs an L-BFGS-B baseline using SciPy for comparison.
- Runs a fixed objective with explicit acceptance, loss, and revenue.
- Saves matplotlib plots to `plots/` (loss curves, gradient norms, and fixed-regression truth plots).

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

The demo samples a single customer `x` and then optimizes over `u` using the explicit objective.

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

## Experiment Configuration

Configs live in `src/experiments/configs/`. Edit `custom.py` for the most recent run and
set which presets to execute in `main.py` by updating `RUN_CONFIGS`.

Each `ExperimentConfig` includes the required state dimension `state_dim`, policy
specification, and an `objective_model` (for example, `FixedRegressionObjective`). State
sampling draws each feature from a standard normal distribution.

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
- L-BFGS-B baseline: uses SciPy's bound-constrained optimizer for a deterministic reference.

Because `u` is clipped to `[0.5, 1.5]`, sufficiently large gradient steps can push iterates to the bounds.

## Reproducibility

The demo uses a fixed RNG seed in `main.py` to make runs repeatable. The objective is deterministic given a fixed configuration and state sample.

## Project Structure

- `main.py`: demo entry point.
- `src/data/models.py`: data classes and objective interfaces.
- `src/data/fixed_objective.py`: fixed regression objective implementation.
- `src/experiments/config.py`: experiment configuration interface.
- `src/experiments/defaults.py`: default helpers for experiment presets.
- `src/experiments/helpers.py`: optimization helper routines.
- `src/experiments/run.py`: experiment runner entry.
- `src/experiments/configs/`: preset configurations (including `custom.py`).
- `src/experiments/logging.py`: logging helpers for experiment outputs.
- `src/experiments/visualization.py`: visualization placeholders.
- `src/optimization/gradients/`: first-order and zeroth-order Stein estimators.
- `src/optimization/policy.py`: policy specs (softmax policy used by default in the demo).
