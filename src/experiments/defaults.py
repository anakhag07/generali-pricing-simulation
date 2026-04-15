"""Default helpers for experiment configurations."""

from __future__ import annotations

import numpy as np

from objective.policy import (
    ConstantPolicy,
    FeatureProcessedPolicy,
    LinearPolicy,
    Policy,
    SoftmaxPolicy,
)


def random_theta0(
    state_dim: int,
    policy: Policy | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a random initial theta within policy-appropriate ranges.

    Uses uniform sampling with ranges chosen to keep initial actions moderate:
    - SoftmaxPolicy: [-1, 1] keeps logit values reasonable
    - LinearPolicy: [-0.5, 0.5] keeps actions moderate
    - ConstantPolicy: only theta[0] is randomised in [-0.3, 0.3]
    """
    dim = state_dim + 1

    # Unwrap FeatureProcessedPolicy to its inner policy
    inner = policy
    if isinstance(inner, FeatureProcessedPolicy):
        inner = inner.policy

    if isinstance(inner, SoftmaxPolicy):
        return rng.uniform(-1.0, 1.0, size=dim)
    if isinstance(inner, LinearPolicy):
        return rng.uniform(-0.5, 0.5, size=dim)
    if isinstance(inner, ConstantPolicy):
        theta = np.zeros(dim, dtype=float)
        theta[0] = rng.uniform(-0.3, 0.3)
        return theta

    # Fallback for unknown policies
    return rng.uniform(-0.5, 0.5, size=dim)


def default_theta0(state_dim: int) -> np.ndarray:
    """Return default initial theta for a policy with given state dimension."""
    return np.zeros(state_dim + 1, dtype=float)


def default_policy(state_dim: int = 1) -> SoftmaxPolicy:
    """Return default softmax policy."""
    return SoftmaxPolicy()
