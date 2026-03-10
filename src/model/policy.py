"""Policy specifications for pricing actions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from objective.base import StateVector

# from optimization.common import clip_u

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


def phi_batch(x_array: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x_array, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_array must be a 2D array.")
    ones = np.ones((x_arr.shape[0], 1), dtype=float)
    return np.concatenate([ones, x_arr], axis=1)


def _sigmoid_batch(z: np.ndarray) -> np.ndarray:
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)
    positive = z_arr >= 0.0
    exp_neg = np.exp(-z_arr[positive])
    out[positive] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(z_arr[~positive])
    out[~positive] = exp_pos / (1.0 + exp_pos)
    return out


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


def policy_u_batch(
    theta: np.ndarray,
    x_array: np.ndarray,
    kind: str = POLICY_CONSTANT,
    phi_array: np.ndarray | None = None,
) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    x_arr = np.asarray(x_array, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_array must be a 2D array.")
    if kind == POLICY_CONSTANT:
        if theta_arr.size < 1:
            raise ValueError("Policy theta must have at least one element.")
        return np.full(x_arr.shape[0], float(theta_arr[0]), dtype=float)

    features = phi_array if phi_array is not None else phi_batch(x_arr)
    if theta_arr.size < features.shape[1]:
        raise ValueError("Policy theta must match feature size for linear/softmax policies.")
    z = features @ theta_arr[: features.shape[1]]
    if kind == POLICY_LINEAR:
        return z.astype(float)
    if kind == POLICY_SOFTMAX:
        sigma = _sigmoid_batch(z)
        return (0.5 + sigma).astype(float)
    raise ValueError(f"Policy kind must be one of {POLICY_KINDS}.")


def policy_grad_theta(theta: np.ndarray, x: StateVector, kind: str = POLICY_CONSTANT) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta_arr)
    if kind == POLICY_CONSTANT:
        if theta_arr.size < 1:
            raise ValueError("Policy theta must have at least one element.")
        grad[0] = 1.0
        return grad

    features = phi(x)
    if theta_arr.size < features.size:
        raise ValueError("Policy theta must match feature size for linear/softmax policies.")

    if kind == POLICY_LINEAR:
        grad[: features.size] = features
        return grad

    if kind == POLICY_SOFTMAX:
        z = float(np.dot(theta_arr[: features.size], features))
        sigma = 1.0 / (1.0 + np.exp(-z))
        du_dz = sigma * (1.0 - sigma)
        grad[: features.size] = du_dz * features
        return grad

    raise ValueError(f"Policy kind must be one of {POLICY_KINDS}.")


def apply_policy(policy: PolicySpec, x: StateVector) -> float:
    # return clip_u(policy_u(policy.theta, x, kind=policy.kind))
    return policy_u(policy.theta, x, kind=policy.kind)
