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
- Evaluates a stochastic objective based on acceptance probability and expected loss.
- Runs first-order and zeroth-order Stein gradient estimators to optimize a pricing action.
- Runs an L-BFGS-B baseline using SciPy for comparison.
- Optionally runs a deterministic fixed objective with explicit acceptance, loss, and revenue.
- Saves matplotlib plots to `plots/` (loss curves, gradient norms, and fixed-regression truth plots).

## Minimization Model

This repo models a pricing action multiplier `u` applied to a baseline price `p` for a customer with features `x`.

```text
Decision variable:  u in [0.5, 1.5]
Customer features:  x
Baseline price:     p

Revenue:            h(p, u) = p * u
Acceptance:         a(x, u) in (0, 1)          (black-box)
Expected loss:      l(x) >= 0                  (black-box)

Objective:          f(u; x) = a(x, u) * ( l(x) - h(p, u) )
Goal (demo):        minimize f(u; x) using noisy black-box queries
```

The demo samples a single customer `x` and then optimizes over `u`. Objective evaluations are stochastic because the black-box acceptance probability and expected loss sample randomness on each call.

## Fixed Regression Objective (Deterministic)

You can switch to a deterministic objective with an explicit parametric form.

```text
Acceptance: a(x, u) = sigmoid(beta_1^T x + beta_2 * u)
Loss:       l(x) = beta_3^T x
Revenue:    r(u) = beta_4 * u

Objective:  f(u; x) = a(x, u) * ( l(x) - r(u) )
```

The beta values are configurable in `ExperimentConfig`. `beta_1` and `beta_3` must be
positive, `beta_4` must be positive, and `beta_2` must be negative so acceptance
decreases with higher policy values. The demo plots the objective and gradient over the
action grid when this objective is used.

The fixed objective is the default. To enable the stochastic objective, pass a config
override in `main.py` or from a REPL:

```python
from experiments.config import ExperimentConfig, OBJECTIVE_STOCHASTIC
from experiments.run import run_experiment

run_experiment(ExperimentConfig(objective_kind=OBJECTIVE_STOCHASTIC))
```

## Experiment Configuration

Configs live in `src/experiments/configs/`. Edit `custom.py` for the most recent run and
set which presets to execute in `main.py` by updating `RUN_CONFIGS`.

Each `ExperimentConfig` includes the state dimension `state_dim`, policy specification,
and fixed-regression parameters (beta values). When `state_dim != 3`, the default state
sampler draws features uniformly on `[0, 1]`.

## Model-to-Code Mapping

```text
x (customer features)            -> StateVector (src/data/models.py)
customer                          -> Customer (src/data/models.py)
u bounds / projection             -> U_BOUNDS, clip_u (src/optimization/common.py)
u as contract action              -> Contract(u=...) (src/data/models.py)
a(x, u) acceptance probability    -> AcceptanceProbability (src/data/models.py)
l(x) expected loss                -> ExpectedFinancialLoss (src/data/models.py)
h(p, u) revenue                   -> revenue_h (src/optimization/objective.py)
f(u; x) objective                 -> objective (src/optimization/objective.py)
oracle gradient API (placeholder) -> objective_with_oracle_grad (src/optimization/objective.py)
policy u = f(theta, x)            -> PolicySpec, apply_policy (src/optimization/policy.py)
experiment runner / config        -> ExperimentConfig, run_experiment (src/experiments/run.py)
```

## Optimization Methods Used

- First-order Stein estimator: uses the oracle gradient API (currently a placeholder) to estimate gradients of a smoothed objective.
- Zeroth-order Stein estimator: uses only objective evaluations at perturbed actions.
- L-BFGS-B baseline: uses SciPy's bound-constrained optimizer for a deterministic reference.

Because `u` is clipped to `[0.5, 1.5]`, sufficiently large gradient steps can push iterates to the bounds.

## Stochasticity and Reproducibility

The demo uses a fixed RNG seed in `main.py` to make runs repeatable. Objective evaluations are still noisy (even for fixed inputs) because the black-box generators sample randomness using the shared RNG.

When using the stochastic objective, the L-BFGS-B baseline evaluates a sample-average objective with a dedicated seed to keep the baseline deterministic and independent of the Stein estimators.

## Project Structure

- `main.py`: demo entry point.
- `src/data/models.py`: data classes and blackbox generators.
- `src/experiments/config.py`: experiment configuration interface.
- `src/experiments/run.py`: experiment runner entry.
- `src/experiments/configs/`: preset configurations (including `custom.py`).
- `src/experiments/logging.py`: logging helpers for experiment outputs.
- `src/experiments/visualization.py`: visualization placeholders.
- `src/optimization/gradients/`: first-order and zeroth-order Stein estimators.
- `src/optimization/objective.py`: objective and oracle gradient API.
- `src/optimization/policy.py`: policy specs (softmax policy used by default in the demo).

## Reproducibility

The demo uses a fixed RNG seed in `main.py` to make runs repeatable.
