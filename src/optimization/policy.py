"""Policy specifications for pricing actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data.models import StateVector
from optimization.common import clip_u

POLICY_CONSTANT = "constant"
POLICY_LINEAR = "linear"
POLICY_SOFTMAX = "softmax"
POLICY_KINDS = (POLICY_CONSTANT, POLICY_LINEAR, POLICY_SOFTMAX)


@dataclass(frozen=True)
class PolicySpec:
    theta: np.ndarray = field(default_factory=lambda: np.asarray([1.0], dtype=float))
    kind: str = POLICY_CONSTANT

    def __post_init__(self) -> None:
        theta = np.asarray(self.theta, dtype=float)
        if theta.size < 1:
            raise ValueError("Policy theta must have at least one element.")
        if self.kind not in POLICY_KINDS:
            raise ValueError(f"Policy kind must be one of {POLICY_KINDS}.")
        object.__setattr__(self, "theta", theta)


def phi(x: StateVector) -> np.ndarray:
    return np.concatenate(([1.0], x.as_array().astype(float)))


def policy_u_constant(theta: np.ndarray, x: StateVector) -> float:
    """Return the pricing action for a constant policy."""
    if theta.size < 1:
        raise ValueError("Policy theta must have at least one element.")
    return float(theta[0])


def policy_u_linear(theta: np.ndarray, x: StateVector) -> float:
    features = phi(x)
    if theta.size < features.size:
        raise ValueError("Policy theta must match feature size for linear policy.")
    return float(np.dot(theta[: features.size], features))


def policy_u_softmax(theta: np.ndarray, x: StateVector) -> float:
    features = phi(x)
    if theta.size < features.size:
        raise ValueError("Policy theta must match feature size for softmax policy.")
    z = float(np.dot(theta[: features.size], features))
    return float(0.5 + np.exp(z) / (1.0 + np.exp(z)))


def policy_u(theta: np.ndarray, x: StateVector, kind: str = POLICY_CONSTANT) -> float:
    if kind == POLICY_CONSTANT:
        return policy_u_constant(theta, x)
    if kind == POLICY_LINEAR:
        return policy_u_linear(theta, x)
    if kind == POLICY_SOFTMAX:
        return policy_u_softmax(theta, x)
    raise ValueError(f"Policy kind must be one of {POLICY_KINDS}.")


def apply_policy(policy: PolicySpec, x: StateVector) -> float:
    return clip_u(policy_u(policy.theta, x, kind=policy.kind))
