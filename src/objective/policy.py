"""Policy mappings from theta and state to action u."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from objective.base import StateVector

POLICY_CONSTANT = "constant"
POLICY_LINEAR = "linear"
POLICY_SOFTMAX = "softmax"
POLICY_KINDS = (POLICY_CONSTANT, POLICY_LINEAR, POLICY_SOFTMAX)


def phi(x: StateVector) -> np.ndarray:
    return np.concatenate(([1.0], x.as_array().astype(float)))


def phi_batch(x_array: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x_array, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_array must be a 2D array.")
    ones = np.ones((x_arr.shape[0], 1), dtype=float)
    return np.concatenate([ones, x_arr], axis=1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)
    positive = z_arr >= 0.0
    exp_neg = np.exp(-z_arr[positive])
    out[positive] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(z_arr[~positive])
    out[~positive] = exp_pos / (1.0 + exp_pos)
    return out


class Policy(Protocol):
    def action(self, theta: np.ndarray, x: StateVector) -> float:
        ...

    def grad_theta(self, theta: np.ndarray, x: StateVector) -> np.ndarray:
        ...

    def action_batch(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        ...

    def required_theta_dim(self, state_dim: int) -> int:
        ...


@dataclass(frozen=True)
class ConstantPolicy:
    kind: str = POLICY_CONSTANT

    def action(self, theta: np.ndarray, x: StateVector) -> float:
        del x
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < 1:
            raise ValueError("theta must have at least one element for constant policy.")
        return float(theta_arr[0])

    def grad_theta(self, theta: np.ndarray, x: StateVector) -> np.ndarray:
        del x
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < 1:
            raise ValueError("theta must have at least one element for constant policy.")
        grad = np.zeros_like(theta_arr)
        grad[0] = 1.0
        return grad

    def action_batch(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be a 2D array.")
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < 1:
            raise ValueError("theta must have at least one element for constant policy.")
        return np.full(x_arr.shape[0], float(theta_arr[0]), dtype=float)

    def required_theta_dim(self, state_dim: int) -> int:
        del state_dim
        return 1


@dataclass(frozen=True)
class LinearPolicy:
    kind: str = POLICY_LINEAR

    def action(self, theta: np.ndarray, x: StateVector) -> float:
        features = phi(x)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.size:
            raise ValueError("theta must have at least state_dim + 1 elements for linear policy.")
        return float(np.dot(theta_arr[: features.size], features))

    def grad_theta(self, theta: np.ndarray, x: StateVector) -> np.ndarray:
        features = phi(x)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.size:
            raise ValueError("theta must have at least state_dim + 1 elements for linear policy.")
        grad = np.zeros_like(theta_arr)
        grad[: features.size] = features
        return grad

    def action_batch(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_batch, dtype=float)
        features = phi_batch(x_arr)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.shape[1]:
            raise ValueError("theta must have at least state_dim + 1 elements for linear policy.")
        return (features @ theta_arr[: features.shape[1]]).astype(float)

    def required_theta_dim(self, state_dim: int) -> int:
        return int(state_dim) + 1


@dataclass(frozen=True)
class SoftmaxPolicy:
    kind: str = POLICY_SOFTMAX

    def action(self, theta: np.ndarray, x: StateVector) -> float:
        features = phi(x)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.size:
            raise ValueError("theta must have at least state_dim + 1 elements for softmax policy.")
        z = float(np.dot(theta_arr[: features.size], features))
        sigma = float(_sigmoid(np.asarray([z]))[0])
        return float(0.5 + sigma)

    def grad_theta(self, theta: np.ndarray, x: StateVector) -> np.ndarray:
        features = phi(x)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.size:
            raise ValueError("theta must have at least state_dim + 1 elements for softmax policy.")
        z = float(np.dot(theta_arr[: features.size], features))
        sigma = float(_sigmoid(np.asarray([z]))[0])
        du_dz = sigma * (1.0 - sigma)
        grad = np.zeros_like(theta_arr)
        grad[: features.size] = du_dz * features
        return grad

    def action_batch(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_batch, dtype=float)
        features = phi_batch(x_arr)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.shape[1]:
            raise ValueError("theta must have at least state_dim + 1 elements for softmax policy.")
        z = features @ theta_arr[: features.shape[1]]
        sigma = _sigmoid(z)
        return (0.5 + sigma).astype(float)

    def required_theta_dim(self, state_dim: int) -> int:
        return int(state_dim) + 1


@dataclass(frozen=True)
class PolicySpec:
    """Compatibility container for policy kind and initial theta."""

    theta: np.ndarray = field(default_factory=lambda: np.asarray([1.0], dtype=float))
    kind: str = POLICY_CONSTANT

    def __post_init__(self) -> None:
        theta_arr = np.asarray(self.theta, dtype=float)
        if theta_arr.ndim != 1 or theta_arr.size < 1:
            raise ValueError("Policy theta must be a 1D array with at least one element.")
        if self.kind not in POLICY_KINDS:
            raise ValueError(f"Policy kind must be one of {POLICY_KINDS}.")
        object.__setattr__(self, "theta", theta_arr)

    def as_policy(self) -> Policy:
        return policy_from_kind(self.kind)


def policy_from_kind(kind: str) -> ConstantPolicy | LinearPolicy | SoftmaxPolicy:
    if kind == POLICY_CONSTANT:
        return ConstantPolicy()
    if kind == POLICY_LINEAR:
        return LinearPolicy()
    if kind == POLICY_SOFTMAX:
        return SoftmaxPolicy()
    raise ValueError(f"Policy kind must be one of {POLICY_KINDS}.")


def policy_u(theta: np.ndarray, x: StateVector, kind: str = POLICY_CONSTANT) -> float:
    return policy_from_kind(kind).action(theta, x)


def policy_u_batch(
    theta: np.ndarray,
    x_array: np.ndarray,
    kind: str = POLICY_CONSTANT,
    phi_array: np.ndarray | None = None,
) -> np.ndarray:
    del phi_array
    return policy_from_kind(kind).action_batch(theta, x_array)


def policy_grad_theta(theta: np.ndarray, x: StateVector, kind: str = POLICY_CONSTANT) -> np.ndarray:
    return policy_from_kind(kind).grad_theta(theta, x)


def apply_policy(policy: PolicySpec, x: StateVector) -> float:
    return policy.as_policy().action(policy.theta, x)


__all__ = [
    "POLICY_CONSTANT",
    "POLICY_KINDS",
    "POLICY_LINEAR",
    "POLICY_SOFTMAX",
    "Policy",
    "ConstantPolicy",
    "LinearPolicy",
    "PolicySpec",
    "SoftmaxPolicy",
    "apply_policy",
    "phi",
    "phi_batch",
    "policy_from_kind",
    "policy_u",
    "policy_u_batch",
    "policy_grad_theta",
]
