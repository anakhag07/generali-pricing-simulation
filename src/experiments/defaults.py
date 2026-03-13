"""Default helpers for experiment configurations."""

from __future__ import annotations

import numpy as np

from objective.policy import POLICY_SOFTMAX, PolicySpec


def default_theta0(state_dim: int) -> np.ndarray:
    return np.asarray([0.1] + [0.01] * state_dim, dtype=float)


def default_policy_spec(state_dim: int) -> PolicySpec:
    return PolicySpec(theta=default_theta0(state_dim), kind=POLICY_SOFTMAX)
