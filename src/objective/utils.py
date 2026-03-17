"""Utility functions for objective computation and reporting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from objective.base import Objective, Policy


def theta_grad_from_u_grad(
    policy: "Policy",
    theta: np.ndarray,
    x_array: np.ndarray,
    grad_u: np.ndarray,
) -> np.ndarray:
    """Compute df/dtheta = mean(df/du * du/dtheta) via chain rule.

    Args:
        policy: Policy mapping (theta, x) to action u.
        theta: Parameter vector.
        x_array: Batch of state vectors, shape (n_samples, state_dim).
        grad_u: Gradient of objective w.r.t. u for each sample, shape (n_samples,).

    Returns:
        Gradient of objective w.r.t. theta, shape (theta_dim,).
    """
    policy_grad = policy.grad_batch(theta, x_array)  # (n_samples, theta_dim)
    return np.mean(grad_u[:, None] * policy_grad, axis=0)


def mean_action(policy: "Policy", theta: np.ndarray, x_array: np.ndarray) -> float:
    """Compute mean policy action across batch.

    Args:
        policy: Policy mapping (theta, x) to action u.
        theta: Parameter vector.
        x_array: Batch of state vectors, shape (n_samples, state_dim).

    Returns:
        Mean action value across the batch.
    """
    return float(np.mean(policy.value_batch(theta, x_array)))


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


def action_value_at_u(
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


__all__ = [
    "theta_grad_from_u_grad",
    "mean_action",
    "optimal_u",
    "action_value_at_u",
]
