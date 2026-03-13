"""Default helpers for experiment configurations."""

from __future__ import annotations

import numpy as np

from objective.policy import POLICY_SOFTMAX, PolicySpec


def default_policy_spec(state_dim: int) -> PolicySpec:
    theta = np.asarray([0.1] + [0.01] * state_dim, dtype=float)
    return PolicySpec(theta=theta, kind=POLICY_SOFTMAX)
