# Generali Pricing Simulation

Pricing simulation and optimization demo using Stein gradient estimators.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python main.py
```

Runtime dependencies live in `requirements.txt` and mirror `pyproject.toml`.

To run tests:

```bash
pip install -e ".[dev]"
pytest
```

## What This Does

- Samples synthetic customer state and contract actions.
- Evaluates a stochastic objective based on acceptance probability and expected loss.
- Runs first-order and zeroth-order Stein gradient estimators to optimize a pricing action.
- Optionally runs a deterministic fixed regression objective with a known minimizer.
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

You can switch to a deterministic objective with a closed-form minimizer. The objective is

```text
f(u; x) = (w^T phi(x) - c * u)^2
```

with analytic minimizer

```text
u* = clip((w^T phi(x)) / c, 0.5, 1.5)
```

When enabled, the demo prints `u*`, the objective at `u*`, and the final iterates from
the first-order and zeroth-order methods.

To enable the fixed objective, pass a config override in `main.py` or from a REPL:

```python
from experiments.runner import ExperimentConfig, run_demo

run_demo(ExperimentConfig(objective_kind="fixed_regression"))
```

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
experiment runner / config        -> ExperimentConfig, run_demo (src/experiments/runner.py)
```

## Optimization Methods Used

- First-order Stein estimator: uses the oracle gradient API (currently a placeholder) to estimate gradients of a smoothed objective.
- Zeroth-order Stein estimator: uses only objective evaluations at perturbed actions.

Because `u` is clipped to `[0.5, 1.5]`, sufficiently large gradient steps can push iterates to the bounds.

## Stochasticity and Reproducibility

The demo uses a fixed RNG seed in `main.py` to make runs repeatable. Objective evaluations are still noisy (even for fixed inputs) because the black-box generators sample randomness using the shared RNG.

## Project Structure

- `main.py`: demo entry point.
- `src/data/models.py`: data classes and blackbox generators.
- `src/experiments/runner.py`: experiment runner entry.
- `src/experiments/logging.py`: logging helpers for experiment outputs.
- `src/experiments/visualization.py`: visualization placeholders.
- `src/optimization/gradients/`: first-order and zeroth-order Stein estimators.
- `src/optimization/objective.py`: objective and oracle gradient API.
- `src/optimization/policy.py`: policy specs (constant policy by default).

## Reproducibility

The demo uses a fixed RNG seed in `main.py` to make runs repeatable.
