"""Policy mappings from theta and state to action u."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from objective.base import Policy, StateVector

# Canonical policy kind constants
policy_constant = "constant"
policy_linear = "linear"
policy_softmax = "softmax"

_POLICY_KINDS = (policy_constant, policy_linear, policy_softmax)


def _phi(x: StateVector) -> np.ndarray:
    """Prepend bias term to feature vector: [1, x_1, ..., x_d]."""
    return np.concatenate(([1.0], np.asarray(x, dtype=float)))


def _phi_batch(x_array: np.ndarray) -> np.ndarray:
    """Prepend bias column to feature array: [[1, x_1, ..., x_d], ...]."""
    x_arr = np.asarray(x_array, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_array must be a 2D array.")
    ones = np.ones((x_arr.shape[0], 1), dtype=float)
    return np.concatenate([ones, x_arr], axis=1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)
    positive = z_arr >= 0.0
    exp_neg = np.exp(-z_arr[positive])
    out[positive] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(z_arr[~positive])
    out[~positive] = exp_pos / (1.0 + exp_pos)
    return out


@dataclass(frozen=True)
class ConstantPolicy(Policy):
    """Constant policy: $u = \\theta_0$, ignores state $x$."""

    kind: str = policy_constant

    def value(self, theta: np.ndarray, x: StateVector) -> float:
        del x
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < 1:
            raise ValueError("theta must have at least one element for constant policy.")
        return float(theta_arr[0])

    def grad(self, theta: np.ndarray, x: StateVector) -> np.ndarray:
        del x
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < 1:
            raise ValueError("theta must have at least one element for constant policy.")
        grad = np.zeros_like(theta_arr)
        grad[0] = 1.0
        return grad

    def value_batch(self, theta: np.ndarray, x_array: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_array, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_array must be a 2D array.")
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < 1:
            raise ValueError("theta must have at least one element for constant policy.")
        return np.full(x_arr.shape[0], float(theta_arr[0]), dtype=float)

    def grad_batch(self, theta: np.ndarray, x_array: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_array, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_array must be a 2D array.")
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < 1:
            raise ValueError("theta must have at least one element for constant policy.")
        n_samples = x_arr.shape[0]
        grad = np.zeros((n_samples, theta_arr.size), dtype=float)
        grad[:, 0] = 1.0
        return grad


@dataclass(frozen=True)
class LinearPolicy(Policy):
    """Linear policy: $u = \\theta^\\top \\phi(x)$ where $\\phi(x) = [1, x]$."""

    kind: str = policy_linear

    def value(self, theta: np.ndarray, x: StateVector) -> float:
        features = _phi(x)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.size:
            raise ValueError("theta must have at least state_dim + 1 elements for linear policy.")
        return float(np.dot(theta_arr[: features.size], features))

    def grad(self, theta: np.ndarray, x: StateVector) -> np.ndarray:
        features = _phi(x)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.size:
            raise ValueError("theta must have at least state_dim + 1 elements for linear policy.")
        grad = np.zeros_like(theta_arr)
        grad[: features.size] = features
        return grad

    def value_batch(self, theta: np.ndarray, x_array: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_array, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_array must be a 2D array.")
        features = _phi_batch(x_arr)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.shape[1]:
            raise ValueError("theta must have at least state_dim + 1 elements for linear policy.")
        return (features @ theta_arr[: features.shape[1]]).astype(float)

    def grad_batch(self, theta: np.ndarray, x_array: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_array, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_array must be a 2D array.")
        features = _phi_batch(x_arr)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.shape[1]:
            raise ValueError("theta must have at least state_dim + 1 elements for linear policy.")
        n_samples = x_arr.shape[0]
        grad = np.zeros((n_samples, theta_arr.size), dtype=float)
        grad[:, : features.shape[1]] = features
        return grad


@dataclass(frozen=True)
class SoftmaxPolicy(Policy):
    """Softmax policy: $u = 0.5 + \\sigma(\\theta^\\top \\phi(x)) \\in (0.5, 1.5)$."""

    kind: str = policy_softmax

    def value(self, theta: np.ndarray, x: StateVector) -> float:
        features = _phi(x)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.size:
            raise ValueError("theta must have at least state_dim + 1 elements for softmax policy.")
        z = float(np.dot(theta_arr[: features.size], features))
        sigma = float(_sigmoid(np.asarray([z]))[0])
        return float(0.5 + sigma)

    def grad(self, theta: np.ndarray, x: StateVector) -> np.ndarray:
        features = _phi(x)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.size:
            raise ValueError("theta must have at least state_dim + 1 elements for softmax policy.")
        z = float(np.dot(theta_arr[: features.size], features))
        sigma = float(_sigmoid(np.asarray([z]))[0])
        du_dz = sigma * (1.0 - sigma)
        grad = np.zeros_like(theta_arr)
        grad[: features.size] = du_dz * features
        return grad

    def value_batch(self, theta: np.ndarray, x_array: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_array, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_array must be a 2D array.")
        features = _phi_batch(x_arr)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.shape[1]:
            raise ValueError("theta must have at least state_dim + 1 elements for softmax policy.")
        z = features @ theta_arr[: features.shape[1]]
        return (0.5 + _sigmoid(z)).astype(float)

    def grad_batch(self, theta: np.ndarray, x_array: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_array, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_array must be a 2D array.")
        features = _phi_batch(x_arr)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.shape[1]:
            raise ValueError("theta must have at least state_dim + 1 elements for softmax policy.")
        z = features @ theta_arr[: features.shape[1]]
        sigma = _sigmoid(z)
        du_dz = sigma * (1.0 - sigma)
        n_samples = x_arr.shape[0]
        grad = np.zeros((n_samples, theta_arr.size), dtype=float)
        grad[:, : features.shape[1]] = du_dz[:, None] * features
        return grad


def policy_from_kind(kind: str) -> ConstantPolicy | LinearPolicy | SoftmaxPolicy:
    """Create a policy instance from a kind string."""
    if kind == policy_constant:
        return ConstantPolicy()
    if kind == policy_linear:
        return LinearPolicy()
    if kind == policy_softmax:
        return SoftmaxPolicy()
    raise ValueError(f"Policy kind must be one of {_POLICY_KINDS}.")


__all__ = [
    "policy_constant",
    "policy_linear",
    "policy_softmax",
    "ConstantPolicy",
    "LinearPolicy",
    "SoftmaxPolicy",
    "policy_from_kind",
]
