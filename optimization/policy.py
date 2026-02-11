"""Policy specifications for pricing actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data.models import StateVector
from optimization.common import clip_u


@dataclass(frozen=True)
class PolicySpec:
    theta: np.ndarray = field(default_factory=lambda: np.asarray([1.0], dtype=float))
    name: str = "constant"

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float)
        if theta.size < 1:
            raise ValueError("Policy theta must have at least one element.")
        object.__setattr__(self, "theta", theta)


def policy_u(theta: np.ndarray, x: StateVector) -> float:
    """Return the pricing action for a constant policy."""
    if theta.size < 1:
        raise ValueError("Policy theta must have at least one element.")
    return float(theta[0])


def apply_policy(policy: PolicySpec, x: StateVector) -> float:
    return clip_u(policy_u(policy.theta, x))
