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
    policy_grad = _policy_grad(target, theta, x_array)  # (n_samples, theta_dim)
    return np.mean(grad_u[:, None] * policy_grad, axis=0)


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


def _action_value_at_u(
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
    value_at_u_fn = getattr(objective, "value_at_u", None)
    if callable(value_at_u_fn):
        return float(value_at_u_fn(x_array, u))
    raise ValueError("objective does not support value_at_u(x_array, u).")


__all__ = ["optimal_u"]
