"""Policy mappings from theta and state to action u."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from objective._math import _sigmoid
from objective.base import Policy

# Canonical policy kind constants (internal)
_POLICY_CONSTANT = "constant"
_POLICY_LINEAR = "linear"
_POLICY_SOFTMAX = "softmax"
_POLICY_KINDS = (_POLICY_CONSTANT, _POLICY_LINEAR, _POLICY_SOFTMAX)


def _as_2d_float_array(x_batch: np.ndarray) -> np.ndarray:
    """Validate and coerce a state batch to a 2D float array."""
    x_arr = np.asarray(x_batch, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_batch must be a 2D array.")
    return x_arr


def _with_intercept(features: np.ndarray) -> np.ndarray:
    """Prepend the policy intercept column to state features."""
    feature_arr = np.asarray(features, dtype=float)
    if feature_arr.ndim != 2:
        raise ValueError("feature_map must return a 2D array.")
    if not np.isfinite(feature_arr).all():
        raise ValueError("feature_map must return finite values.")
    ones = np.ones((feature_arr.shape[0], 1), dtype=float)
    return np.concatenate([ones, feature_arr], axis=1)


def _phi(x_batch: np.ndarray, feature_map: "FeatureMap | None" = None) -> np.ndarray:
    """Return policy features with the intercept prepended internally."""
    mapper = feature_map if feature_map is not None else IdentityFeatureMap()
    return _with_intercept(mapper.transform(x_batch))


def _validate_state_dim(state_dim: int) -> int:
    dim = int(state_dim)
    if dim <= 0:
        raise ValueError("state_dim must be positive.")
    return dim


def _validate_feature_output(
    features: np.ndarray,
    x_arr: np.ndarray,
    *,
    expected_dim: int | None = None,
) -> np.ndarray:
    feature_arr = np.asarray(features, dtype=float)
    if feature_arr.ndim != 2:
        raise ValueError("feature_map must return a 2D array.")
    if feature_arr.shape[0] != x_arr.shape[0]:
        raise ValueError("feature_map must preserve the number of samples.")
    if expected_dim is not None and feature_arr.shape[1] != expected_dim:
        raise ValueError(
            f"feature_map returned {feature_arr.shape[1]} features; expected {expected_dim}."
        )
    if not np.isfinite(feature_arr).all():
        raise ValueError("feature_map must return finite values.")
    return feature_arr


class FeatureMap:
    """State feature map $$\varphi(x)$$; policies prepend the intercept internally."""

    kind: str

    def transform(self, x_batch: np.ndarray) -> np.ndarray:
        """Return mapped state features, shape ``(n_samples, feature_dim)``."""
        raise NotImplementedError

    def output_dim(self, state_dim: int) -> int:
        """Return ``feature_dim`` for inputs with ``state_dim`` columns."""
        raise NotImplementedError


@dataclass(frozen=True)
class IdentityFeatureMap(FeatureMap):
    """Default map: $$\varphi(x) = x$$."""

    kind: str = "identity"

    def transform(self, x_batch: np.ndarray) -> np.ndarray:
        """Return ``x_batch`` as finite 2D float features."""
        x_arr = _as_2d_float_array(x_batch)
        if not np.isfinite(x_arr).all():
            raise ValueError("x_batch must contain finite values.")
        return x_arr

    def output_dim(self, state_dim: int) -> int:
        """Return the input state dimension."""
        return _validate_state_dim(state_dim)


@dataclass(frozen=True)
class QuadraticFeatureMap(FeatureMap):
    """Quadratic map with linear terms and upper-triangular pair products."""

    include_interactions: bool = True
    kind: str = "quadratic"

    def transform(self, x_batch: np.ndarray) -> np.ndarray:
        """Return ``[x, x_i*x_j for i <= j]`` features for each sample."""
        x_arr = _as_2d_float_array(x_batch)
        if not np.isfinite(x_arr).all():
            raise ValueError("x_batch must contain finite values.")
        n_samples, state_dim = x_arr.shape
        cols = [x_arr]
        if self.include_interactions:
            row_idx, col_idx = np.triu_indices(state_dim)
            cols.append(x_arr[:, row_idx] * x_arr[:, col_idx])
        else:
            cols.append(x_arr * x_arr)
        if not cols:
            return np.empty((n_samples, 0), dtype=float)
        return np.concatenate(cols, axis=1).astype(float)

    def output_dim(self, state_dim: int) -> int:
        """Return quadratic feature width excluding the policy intercept."""
        dim = _validate_state_dim(state_dim)
        quadratic_dim = dim * (dim + 1) // 2 if self.include_interactions else dim
        return dim + quadratic_dim


@dataclass(frozen=True)
class CallableFeatureMap(FeatureMap):
    """Validated adapter for a user-supplied state feature callable."""

    fn: Callable[[np.ndarray], np.ndarray]
    feature_dim: int
    name: str = "callable"
    kind: str = "callable"

    def __post_init__(self) -> None:
        if not callable(self.fn):
            raise ValueError("fn must be callable.")
        if int(self.feature_dim) <= 0:
            raise ValueError("feature_dim must be positive.")

    def transform(self, x_batch: np.ndarray) -> np.ndarray:
        """Apply and validate the wrapped feature-map callable."""
        x_arr = _as_2d_float_array(x_batch)
        features = self.fn(x_arr)
        return _validate_feature_output(features, x_arr, expected_dim=int(self.feature_dim))

    def output_dim(self, state_dim: int) -> int:
        """Return the declared callable output width."""
        _validate_state_dim(state_dim)
        return int(self.feature_dim)


def _theta_dim(feature_map: FeatureMap, state_dim: int) -> int:
    """Return policy theta dimension: intercept plus mapped features."""
    return 1 + feature_map.output_dim(state_dim)


def _validate_theta(theta: np.ndarray, expected_dim: int) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    if theta_arr.ndim != 1:
        raise ValueError("theta must be a 1D array.")
    if theta_arr.size != expected_dim:
        raise ValueError(f"theta must have exactly {expected_dim} elements.")
    if not np.isfinite(theta_arr).all():
        raise ValueError("theta must contain finite values.")
    return theta_arr


def policy_theta_dim(policy: Policy, state_dim: int) -> int:
    """Return the theta dimension required by ``policy`` for ``state_dim`` inputs."""
    theta_dim_fn = getattr(policy, "theta_dim", None)
    if callable(theta_dim_fn):
        try:
            return int(theta_dim_fn(state_dim))
        except NotImplementedError:
            pass
    return int(state_dim) + 1


@dataclass(frozen=True)
class ConstantPolicy(Policy):
    """Constant policy: $$u = \\theta_0$$, ignores state $$x$$."""

    kind: str = _POLICY_CONSTANT

    def theta_dim(self, state_dim: int) -> int:
        """Return the one scalar parameter used by the constant policy."""
        _validate_state_dim(state_dim)
        return 1

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return constant action for all samples, shape (n_samples,)."""
        x_arr = _as_2d_float_array(x_batch)
        theta_arr = _validate_theta(theta, expected_dim=1)
        return np.full(x_arr.shape[0], float(theta_arr[0]), dtype=float)

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return gradient [1] for all samples, shape (n_samples, 1)."""
        x_arr = _as_2d_float_array(x_batch)
        _validate_theta(theta, expected_dim=1)
        grad = np.zeros((x_arr.shape[0], 1), dtype=float)
        grad[:, 0] = 1.0
        return grad


@dataclass(frozen=True)
class LinearPolicy(Policy):
    """Linear policy: $$u = \\theta^\\top \\phi(x)$$ with intercept prepended internally."""

    feature_map: FeatureMap = field(default_factory=IdentityFeatureMap)
    kind: str = _POLICY_LINEAR

    def theta_dim(self, state_dim: int) -> int:
        """Return ``1 + dim(varphi(x))`` for this policy's feature map."""
        return _theta_dim(self.feature_map, state_dim)

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return linear action values, shape (n_samples,)."""
        features = _phi(x_batch, self.feature_map)
        theta_arr = _validate_theta(theta, expected_dim=features.shape[1])
        return (features @ theta_arr).astype(float)

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return gradient phi(x) for all samples, shape (n_samples, theta_dim)."""
        features = _phi(x_batch, self.feature_map)
        _validate_theta(theta, expected_dim=features.shape[1])
        return features


@dataclass(frozen=True)
class SoftmaxPolicy(Policy):
    """Softmax policy: $$u = 0.5 - \\sigma(\\theta^\\top \\phi(x)) \\in (-0.5, 0.5)$$."""

    feature_map: FeatureMap = field(default_factory=IdentityFeatureMap)
    kind: str = _POLICY_SOFTMAX

    def theta_dim(self, state_dim: int) -> int:
        """Return ``1 + dim(varphi(x))`` for this policy's feature map."""
        return _theta_dim(self.feature_map, state_dim)

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return softmax action values, shape (n_samples,)."""
        features = _phi(x_batch, self.feature_map)
        theta_arr = _validate_theta(theta, expected_dim=features.shape[1])
        z = features @ theta_arr
        return (0.5 - _sigmoid(z)).astype(float)

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return gradient ``-sigma'(z) * phi(x)`` for all samples, shape ``(n_samples, theta_dim)``."""
        features = _phi(x_batch, self.feature_map)
        theta_arr = _validate_theta(theta, expected_dim=features.shape[1])
        z = features @ theta_arr
        sigma = _sigmoid(z)
        du_dz = -sigma * (1.0 - sigma)
        return du_dz[:, None] * features


@dataclass(frozen=True)
class FeatureProcessedPolicy(Policy):
    """Policy wrapper that preprocesses raw state before delegating."""

    policy: Policy
    raw_feature_cols: tuple[str, ...]
    preprocess_feature_cols: tuple[str, ...]
    preprocessor: object
    processed_state_dim: int | None = None
    kind: str = "feature_processed"

    def theta_dim(self, state_dim: int) -> int:
        """Return the inner policy theta dimension for processed state features."""
        _validate_state_dim(state_dim)
        if self.processed_state_dim is None:
            raise ValueError("processed_state_dim is required for FeatureProcessedPolicy theta sizing.")
        return policy_theta_dim(self.policy, self.processed_state_dim)

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
    "CallableFeatureMap",
    "ConstantPolicy",
    "FeatureMap",
    "FeatureProcessedPolicy",
    "IdentityFeatureMap",
    "LinearPolicy",
    "QuadraticFeatureMap",
    "SoftmaxPolicy",
    "policy_from_kind",
    "policy_theta_dim",
]
