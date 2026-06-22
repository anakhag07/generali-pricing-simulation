"""Utility functions for objective computation and reporting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from objective.base import Objective, Policy


def _policy_value(target: "Objective | Policy", theta: np.ndarray, x_array: np.ndarray) -> np.ndarray:
    """Evaluate policy actions, allowing objectives to own x preprocessing."""
    value_fn = getattr(target, "policy_value", None)
    if callable(value_fn):
        return np.asarray(value_fn(theta, x_array), dtype=float)
    policy = getattr(target, "policy", target)
    return np.asarray(policy.value(theta, x_array), dtype=float)


def _policy_grad(target: "Objective | Policy", theta: np.ndarray, x_array: np.ndarray) -> np.ndarray:
    """Evaluate policy Jacobians, allowing objectives to own x preprocessing."""
    grad_fn = getattr(target, "policy_grad", None)
    if callable(grad_fn):
        return np.asarray(grad_fn(theta, x_array), dtype=float)
    policy = getattr(target, "policy", target)
    return np.asarray(policy.grad(theta, x_array), dtype=float)


def _policy_weighted_grad(
    target: "Objective | Policy",
    theta: np.ndarray,
    x_array: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Evaluate ``sum_i weights_i * d pi_theta(x_i) / d theta``."""
    weights_arr = np.asarray(weights, dtype=float)
    weighted_grad_fn = getattr(target, "policy_weighted_grad", None)
    if callable(weighted_grad_fn):
        return np.asarray(weighted_grad_fn(theta, x_array, weights_arr), dtype=float)
    policy = getattr(target, "policy", target)
    policy_weighted_grad_fn = getattr(policy, "weighted_grad", None)
    if callable(policy_weighted_grad_fn):
        return np.asarray(policy_weighted_grad_fn(theta, x_array, weights_arr), dtype=float)
    policy_grad = _policy_grad(target, theta, x_array)
    return np.einsum("n,nd->d", weights_arr, policy_grad)


def _theta_grad_from_u_grad(
    target: "Objective | Policy",
    theta: np.ndarray,
    x_array: np.ndarray,
    grad_u: np.ndarray,
) -> np.ndarray:
    """Compute df/dtheta = mean(df/du * du/dtheta) via chain rule.

    Args:
        target: Policy or objective exposing policy evaluation hooks.
        theta: Parameter vector.
        x_array: Batch of state vectors, shape (n_samples, state_dim).
        grad_u: Gradient of objective w.r.t. u for each sample, shape (n_samples,).

    Returns:
        Gradient of objective w.r.t. theta, shape (theta_dim,).
    """
    grad_u_arr = np.asarray(grad_u, dtype=float)
    if grad_u_arr.ndim != 1:
        raise ValueError("grad_u must be a 1D array.")
    if grad_u_arr.size == 0:
        raise ValueError("grad_u must contain at least one sample.")
    return _policy_weighted_grad(target, theta, x_array, grad_u_arr) / float(grad_u_arr.size)


def _mean_action(target: "Objective | Policy", theta: np.ndarray, x_array: np.ndarray) -> float:
    """Compute mean policy action across batch.

    Args:
        target: Policy or objective exposing policy evaluation hooks.
        theta: Parameter vector.
        x_array: Batch of state vectors, shape (n_samples, state_dim).

    Returns:
        Mean action value across the batch.
    """
    u_batch = _policy_value(target, theta, x_array)
    clip_fn = getattr(target, "_clip_u", None)
    if callable(clip_fn):
        u_batch = np.asarray(clip_fn(u_batch), dtype=float)
    return float(np.mean(u_batch))


def optimal_u(objective: "Objective") -> float | None:
    """Return optimal action u* if the objective exposes it.

    For objectives with a known optimum (e.g., PlantedLogisticObjective),
    this returns the optimal action value. New objectives can expose this
    by implementing an `optimal_u() -> float` method.

    Args:
        objective: A theta-level objective.

    Returns:
        The optimal action value if available, otherwise None.
    """
    optimal_fn = getattr(objective, "optimal_u", None)
    if callable(optimal_fn):
        result = optimal_fn()
        if result is not None:
            return float(result)
    u_star_attr = getattr(objective, "u_star", None)
    if u_star_attr is not None:
        return float(u_star_attr)
    return None


def value_at_constant_u(
    objective: "Objective",
    x_array: np.ndarray,
    u: float,
) -> float:
    """Compute mean action-level objective value at a fixed action u.

    Args:
        objective: A theta-level objective with internal action objective.
        x_array: Batch of state vectors, shape (n_samples, state_dim).
        u: Fixed action value.

    Returns:
        Mean objective value across the batch at the fixed action.
    """
    base_value_at_u_fn = getattr(objective, "base_value_at_u", None)
    if callable(base_value_at_u_fn):
        return float(base_value_at_u_fn(x_array, u))
    value_at_u_fn = getattr(objective, "value_at_u", None)
    if callable(value_at_u_fn):
        return float(value_at_u_fn(x_array, u))
    raise ValueError("objective does not support value_at_u(x_array, u).")


def value_for_reporting(
    objective: "Objective",
    theta: np.ndarray,
    x_array: np.ndarray,
) -> float:
    """Return the raw objective value used in summaries and frontier plots."""
    base_value_fn = getattr(objective, "base_value", None)
    if callable(base_value_fn):
        return float(base_value_fn(theta, x_array))
    return float(objective.value(theta, x_array))


def mean_acceptance_at_constant_u(
    objective: "Objective",
    x_array: np.ndarray,
    u: float,
) -> float | None:
    """Return mean acceptance at a fixed action when the objective supports it."""
    mean_acceptance_at_u_fn = getattr(objective, "mean_acceptance_at_u", None)
    if callable(mean_acceptance_at_u_fn):
        return float(mean_acceptance_at_u_fn(x_array, u))
    return None


def _action_value_at_u(
    objective: "Objective",
    x_array: np.ndarray,
    u: float,
) -> float:
    """Backward-compatible internal alias for fixed-action objective evaluation."""
    return value_at_constant_u(objective, x_array, u)


__all__ = ["mean_acceptance_at_constant_u", "optimal_u", "value_at_constant_u", "value_for_reporting"]
