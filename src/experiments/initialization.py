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


def random_theta0(
    state_dim: int,
    policy: Policy | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a random initial theta within policy-appropriate ranges."""
    inner = policy.policy if isinstance(policy, FeatureProcessedPolicy) else policy
    dim = policy_theta_dim(policy, state_dim) if policy is not None else state_dim + 1

    if isinstance(inner, SoftmaxPolicy):
        return rng.uniform(-1.0, 1.0, size=dim)
    if isinstance(inner, LinearPolicy):
        return rng.uniform(-0.5, 0.5, size=dim)
    if isinstance(inner, ConstantPolicy):
        theta = np.zeros(dim, dtype=float)
        theta[0] = rng.uniform(-0.3, 0.3)
        return theta
    return rng.uniform(-0.5, 0.5, size=dim)


__all__ = ["random_theta0"]
