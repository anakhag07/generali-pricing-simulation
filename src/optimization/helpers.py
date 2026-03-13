"""Helper functions for optimization batching and objective evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np

from optimization.steps import STEP_RULE_LBFGSB


def scipy_method(algorithm: str) -> str:
    if algorithm.lower() == STEP_RULE_LBFGSB:
        return "L-BFGS-B"
    raise ValueError(f"Unsupported algorithm '{algorithm}'.")


def sample_indices(
    rng: np.random.Generator,
    batch_size_eff: int,
    n_total: int,
    full_indices: np.ndarray,
) -> np.ndarray:
    if batch_size_eff >= n_total:
        return full_indices
    return rng.choice(n_total, size=batch_size_eff, replace=False)


def x_batch(x_array: np.ndarray, indices: np.ndarray, n_total: int) -> np.ndarray:
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
    return float(objective.value(theta, x_batch(x_array, indices, n_total)))


def objective_grad_on_indices(
    objective: Any,
    x_array: np.ndarray,
    n_total: int,
    theta: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    return np.asarray(objective.grad(theta, x_batch(x_array, indices, n_total)), dtype=float)


def mean_action_on_indices(
    objective: Any,
    x_array: np.ndarray,
    n_total: int,
    theta: np.ndarray,
    indices: np.ndarray,
) -> float:
    mean_action = getattr(objective, "mean_action", None)
    if callable(mean_action):
        return float(mean_action(theta, x_batch(x_array, indices, n_total)))
    return float("nan")


__all__ = [
    "mean_action_on_indices",
    "objective_grad_on_indices",
    "objective_value_on_indices",
    "sample_indices",
    "scipy_method",
    "x_batch",
]
