"""Policy mappings from theta and state to action u."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from objective._math import _sigmoid
from objective.base import Policy

# Canonical policy kind constants (internal)
_POLICY_CONSTANT = "constant"
_POLICY_LINEAR = "linear"
_POLICY_SOFTMAX = "softmax"
_POLICY_KINDS = (_POLICY_CONSTANT, _POLICY_LINEAR, _POLICY_SOFTMAX)


def _phi(x_batch: np.ndarray) -> np.ndarray:
    """Prepend bias column to feature array: [[1, x_1, ..., x_d], ...]."""
    x_arr = np.asarray(x_batch, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_batch must be a 2D array.")
    ones = np.ones((x_arr.shape[0], 1), dtype=float)
    return np.concatenate([ones, x_arr], axis=1)


@dataclass(frozen=True)
class ConstantPolicy(Policy):
    """Constant policy: $$u = \\theta_0$$, ignores state $$x$$."""

    kind: str = _POLICY_CONSTANT

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return constant action for all samples, shape (n_samples,)."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be a 2D array.")
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < 1:
            raise ValueError("theta must have at least one element.")
        return np.full(x_arr.shape[0], float(theta_arr[0]), dtype=float)

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return gradient [1, 0, ...] for all samples, shape (n_samples, theta_dim)."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be a 2D array.")
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < 1:
            raise ValueError("theta must have at least one element.")
        n_samples = x_arr.shape[0]
        grad = np.zeros((n_samples, theta_arr.size), dtype=float)
        grad[:, 0] = 1.0
        return grad


@dataclass(frozen=True)
class LinearPolicy(Policy):
    """Linear policy: $$u = \\theta^\\top \\phi(x)$$ where $$\\phi(x) = [1, x]$$."""

    kind: str = _POLICY_LINEAR

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return linear action values, shape (n_samples,)."""
        features = _phi(x_batch)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.shape[1]:
            raise ValueError("theta must have at least state_dim + 1 elements.")
        return (features @ theta_arr[: features.shape[1]]).astype(float)

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return gradient phi(x) for all samples, shape (n_samples, theta_dim)."""
        features = _phi(x_batch)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.shape[1]:
            raise ValueError("theta must have at least state_dim + 1 elements.")
        n_samples = features.shape[0]
        grad = np.zeros((n_samples, theta_arr.size), dtype=float)
        grad[:, : features.shape[1]] = features
        return grad


@dataclass(frozen=True)
class SoftmaxPolicy(Policy):
    """Softmax policy: $$u = 0.5 + \\sigma(\\theta^\\top \\phi(x)) \\in (0.5, 1.5)$$."""

    kind: str = _POLICY_SOFTMAX

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return softmax action values, shape (n_samples,)."""
        features = _phi(x_batch)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.shape[1]:
            raise ValueError("theta must have at least state_dim + 1 elements.")
        z = features @ theta_arr[: features.shape[1]]
        return (0.5 + _sigmoid(z)).astype(float)

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return gradient sigma'(z) * phi(x) for all samples, shape (n_samples, theta_dim)."""
        features = _phi(x_batch)
        theta_arr = np.asarray(theta, dtype=float)
        if theta_arr.size < features.shape[1]:
            raise ValueError("theta must have at least state_dim + 1 elements.")
        z = features @ theta_arr[: features.shape[1]]
        sigma = _sigmoid(z)
        du_dz = sigma * (1.0 - sigma)
        n_samples = features.shape[0]
        grad = np.zeros((n_samples, theta_arr.size), dtype=float)
        grad[:, : features.shape[1]] = du_dz[:, None] * features
        return grad


@dataclass(frozen=True)
class FeatureProcessedPolicy(Policy):
    """Policy wrapper that preprocesses raw state before delegating."""

    policy: Policy
    raw_feature_cols: tuple[str, ...]
    preprocess_feature_cols: tuple[str, ...]
    preprocessor: object
    kind: str = "feature_processed"

    def _transform(self, x_batch: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be a 2D array.")
        if x_arr.shape[1] != len(self.raw_feature_cols):
            raise ValueError(
                f"Expected raw state_dim={len(self.raw_feature_cols)}, got {x_arr.shape[1]}."
            )
        raw_df = pd.DataFrame(x_arr, columns=list(self.raw_feature_cols))
        processed = self.preprocessor.transform(raw_df.loc[:, list(self.preprocess_feature_cols)].copy())
        return np.asarray(processed, dtype=float)

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        return self.policy.value(theta, self._transform(x_batch))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        return self.policy.grad(theta, self._transform(x_batch))


def policy_from_kind(kind: str) -> ConstantPolicy | LinearPolicy | SoftmaxPolicy:
    """Create a policy instance from a kind string."""
    if kind == _POLICY_CONSTANT:
        return ConstantPolicy()
    if kind == _POLICY_LINEAR:
        return LinearPolicy()
    if kind == _POLICY_SOFTMAX:
        return SoftmaxPolicy()
    raise ValueError(f"Policy kind must be one of {_POLICY_KINDS}.")


__all__ = [
    "ConstantPolicy",
    "FeatureProcessedPolicy",
    "LinearPolicy",
    "SoftmaxPolicy",
    "policy_from_kind",
]
