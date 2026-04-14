"""Default helpers for experiment configurations."""

from __future__ import annotations

import numpy as np

from objective.policy import SoftmaxPolicy


def default_theta0(state_dim: int) -> np.ndarray:
    """Return default initial theta for a policy with given state dimension."""
    return np.zeros(state_dim + 1, dtype=float)


def default_policy(state_dim: int = 1) -> SoftmaxPolicy:
    """Return default softmax policy."""
    return SoftmaxPolicy()
