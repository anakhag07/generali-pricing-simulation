"""Helper functions for optimization batching and objective evaluation."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from objective.utils import _mean_action
from optimization.steps import STEP_RULE_LBFGSB, STEP_RULE_TRUST_CONSTR


def _clamp_theta(theta: np.ndarray, bounds: tuple[float, float] | None) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    if bounds is None:
        return theta_arr
    lower, upper = bounds
    return np.clip(theta_arr, lower, upper)


def scipy_method(algorithm: str) -> str:
    """Map algorithm string to SciPy method name."""
    if algorithm.lower() == STEP_RULE_LBFGSB:
        return "L-BFGS-B"
    if algorithm.lower() == STEP_RULE_TRUST_CONSTR:
        return STEP_RULE_TRUST_CONSTR
    raise ValueError(
        f"Unsupported algorithm '{algorithm}'. "
        f"Currently only '{STEP_RULE_LBFGSB}' and '{STEP_RULE_TRUST_CONSTR}' are supported. "
        f"Use one of those step_rule values in your config."
    )


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


def x_batch(x_array: Any, indices: np.ndarray, n_total: int) -> Any:
    """Extract mini-batch from x_array."""
    if indices.size == n_total:
        return x_array
    if hasattr(x_array, "iloc"):
        return x_array.iloc[indices].reset_index(drop=True)
    return x_array[indices]


def objective_value_on_indices(
    objective: Any,
    x_array: Any,
    n_total: int,
    theta: np.ndarray,
    indices: np.ndarray,
) -> float:
    """Compute objective value on mini-batch."""
    return float(objective.value(theta, x_batch(x_array, indices, n_total)))


def objective_grad_on_indices(
    objective: Any,
    x_array: Any,
    n_total: int,
    theta: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Compute objective gradient on mini-batch."""
    return np.asarray(objective.grad(theta, x_batch(x_array, indices, n_total)), dtype=float)


def finite_difference_theta_grad(
    value_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    method: str,
    step: float,
    bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """Compute a theta gradient from scalar objective evaluations.

    Central: $$(f(\\theta+h e_k) - f(\\theta-h e_k))/(2h)$$.
    Forward: $$(f(\\theta+h e_k) - f(\\theta))/h$$.
    Backward: $$(f(\\theta) - f(\\theta-h e_k))/h$$.
    """
    theta_arr = np.asarray(theta, dtype=float)
    if method not in {"central", "forward", "backward"}:
        raise ValueError(f"Unknown numdiff method '{method}'.")
    if step <= 0.0:
        raise ValueError("Finite-difference step must be positive.")

    grad = np.zeros_like(theta_arr)
    center_value = float(value_fn(theta_arr)) if method in {"forward", "backward"} else None

    for idx in range(theta_arr.size):
        basis = np.zeros_like(theta_arr)
        basis[idx] = 1.0
        if method == "central":
            theta_plus = _clamp_theta(theta_arr + step * basis, bounds)
            theta_minus = _clamp_theta(theta_arr - step * basis, bounds)
            denom = theta_plus[idx] - theta_minus[idx]
            if denom == 0.0:
                grad[idx] = 0.0
            else:
                grad[idx] = (float(value_fn(theta_plus)) - float(value_fn(theta_minus))) / denom
            continue
        if method == "forward":
            theta_plus = _clamp_theta(theta_arr + step * basis, bounds)
            denom = theta_plus[idx] - theta_arr[idx]
            if denom == 0.0:
                grad[idx] = 0.0
            else:
                grad[idx] = (float(value_fn(theta_plus)) - float(center_value)) / denom
            continue
        theta_minus = _clamp_theta(theta_arr - step * basis, bounds)
        denom = theta_arr[idx] - theta_minus[idx]
        if denom == 0.0:
            grad[idx] = 0.0
        else:
            grad[idx] = (float(center_value) - float(value_fn(theta_minus))) / denom
    return grad


def stein_difference_theta_grad(
    value_fn: Callable[[np.ndarray], float],
    theta: np.ndarray,
    *,
    step: float,
    epsilon_samples: np.ndarray,
) -> np.ndarray:
    """Estimate a gradient with antithetic Gaussian Stein differences."""
    theta_arr = np.asarray(theta, dtype=float)
    eps_arr = np.asarray(epsilon_samples, dtype=float)
    if step <= 0.0:
        raise ValueError("Stein-difference step must be positive.")
    if eps_arr.ndim != 2 or eps_arr.shape[1] != theta_arr.size or eps_arr.shape[0] == 0:
        raise ValueError("epsilon_samples must have shape (n_samples, theta.size).")
    if not np.all(np.isfinite(eps_arr)):
        raise ValueError("epsilon_samples must be finite.")

    grad = np.zeros_like(theta_arr, dtype=float)
    for epsilon in eps_arr:
        value_plus = float(value_fn(theta_arr + step * epsilon))
        value_minus = float(value_fn(theta_arr - step * epsilon))
        grad += ((value_plus - value_minus) / (2.0 * step)) * epsilon
    return grad / float(eps_arr.shape[0])


def mean_action_on_indices(
    objective: Any,
    x_array: Any,
    n_total: int,
    theta: np.ndarray,
    indices: np.ndarray,
) -> float:
    """Compute mean action on mini-batch."""
    policy = getattr(objective, "policy", None)
    if policy is not None:
        return float(_mean_action(objective, theta, x_batch(x_array, indices, n_total)))
    return float("nan")


__all__ = [
    "finite_difference_theta_grad",
    "mean_action_on_indices",
    "objective_grad_on_indices",
    "objective_value_on_indices",
    "sample_indices",
    "scipy_method",
    "stein_difference_theta_grad",
    "x_batch",
]
