"""Policy-parameter initialization helpers for experiment runs."""

from __future__ import annotations

import numpy as np

from objective.policy import (
    ConstantPolicy,
    FeatureProcessedPolicy,
    LinearPolicy,
    Policy,
    SoftmaxPolicy,
    policy_theta_dim,
)


def objective_theta_dim(objective: object, state_dim: int) -> int | None:
    """Return the required theta dimension when an objective or its policy exposes one."""
    theta_dim_fn = getattr(objective, "theta_dim", None)
    if callable(theta_dim_fn):
        return int(theta_dim_fn(state_dim))
    policy_theta_dim_fn = getattr(objective, "policy_theta_dim", None)
    if callable(policy_theta_dim_fn):
        return int(policy_theta_dim_fn(state_dim))
    policy = getattr(objective, "policy", None)
    if policy is None:
        return None
    return policy_theta_dim(policy, state_dim)


def random_theta0(
    state_dim: int,
    policy: Policy | None,
    rng: np.random.Generator,
    *,
    parameter_dim: int | None = None,
) -> np.ndarray:
    """Generate random initial theta with the requested objective/policy dimension."""
    inner = policy.policy if isinstance(policy, FeatureProcessedPolicy) else policy
    if parameter_dim is not None:
        dim = int(parameter_dim)
    elif policy is not None:
        dim = policy_theta_dim(policy, state_dim)
    else:
        dim = state_dim + 1

    if isinstance(inner, SoftmaxPolicy):
        return rng.uniform(-1.0, 1.0, size=dim)
    if isinstance(inner, LinearPolicy):
        return rng.uniform(-0.5, 0.5, size=dim)
    if isinstance(inner, ConstantPolicy):
        theta = np.zeros(dim, dtype=float)
        theta[0] = rng.uniform(-0.3, 0.3)
        return theta
    return rng.uniform(-0.5, 0.5, size=dim)


__all__ = ["objective_theta_dim", "random_theta0"]
