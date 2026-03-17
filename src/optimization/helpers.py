"""Helper functions for optimization batching and objective evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from objective.utils import mean_action
from optimization.steps import STEP_RULE_LBFGSB


def scipy_method(algorithm: str) -> str:
    """Map algorithm string to SciPy method name."""
    if algorithm.lower() == STEP_RULE_LBFGSB:
        return "L-BFGS-B"
    raise ValueError(f"Unsupported algorithm '{algorithm}'.")


def sample_indices(
    rng: np.random.Generator,
    batch_size_eff: int,
    n_total: int,
    full_indices: np.ndarray,
) -> np.ndarray:
    """Sample mini-batch indices."""
    if batch_size_eff >= n_total:
        return full_indices
    return rng.choice(n_total, size=batch_size_eff, replace=False)


def x_batch(x_array: np.ndarray, indices: np.ndarray, n_total: int) -> np.ndarray:
    """Extract mini-batch from x_array."""
    if indices.size == n_total:
        return x_array
    return x_array[indices]


def objective_value_on_indices(
    objective: Any,
    x_array: np.ndarray,
    n_total: int,
    theta: np.ndarray,
    indices: np.ndarray,
) -> float:
    """Compute objective value on mini-batch."""
    return float(objective.value(theta, x_batch(x_array, indices, n_total)))


def objective_grad_on_indices(
    objective: Any,
    x_array: np.ndarray,
    n_total: int,
    theta: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Compute objective gradient on mini-batch."""
    return np.asarray(objective.grad(theta, x_batch(x_array, indices, n_total)), dtype=float)


def mean_action_on_indices(
    objective: Any,
    x_array: np.ndarray,
    n_total: int,
    theta: np.ndarray,
    indices: np.ndarray,
) -> float:
    """Compute mean action on mini-batch."""
    policy = getattr(objective, "policy", None)
    if policy is not None:
        return float(mean_action(policy, theta, x_batch(x_array, indices, n_total)))
    return float("nan")


__all__ = [
    "mean_action_on_indices",
    "objective_grad_on_indices",
    "objective_value_on_indices",
    "sample_indices",
    "scipy_method",
    "x_batch",
]
