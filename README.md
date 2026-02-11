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
