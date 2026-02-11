# Generali Pricing Simulation

Pricing simulation and optimization demo using Stein gradient estimators.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## What This Does

- Samples synthetic customer state and contract actions.
- Evaluates a stochastic objective based on acceptance probability and expected loss.
- Runs first-order and zeroth-order Stein gradient estimators to optimize a pricing action.

## Project Structure

- `main.py`: demo entry point.
- `data/models.py`: data classes and blackbox generators.
- `experiments/runner.py`: experiment runner entry.
- `experiments/logging.py`: logging helpers for experiment outputs.
- `experiments/visualization.py`: visualization placeholders.
- `optimization/gradients/`: first-order and zeroth-order Stein estimators.
- `optimization/objective.py`: objective and oracle gradient API.

## Reproducibility

The demo uses a fixed RNG seed in `main.py` to make runs repeatable.
