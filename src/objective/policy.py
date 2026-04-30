"""Policy mappings from theta and state to action u."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations_with_replacement
from math import comb

import numpy as np
import pandas as pd

from objective._math import _sigmoid
from objective.base import Policy

# Canonical policy kind constants (internal)
_POLICY_CONSTANT = "constant"
_POLICY_LINEAR = "linear"
_POLICY_SOFTMAX = "softmax"
_POLICY_MLP = "mlp"
_POLICY_KINDS = (_POLICY_CONSTANT, _POLICY_LINEAR, _POLICY_SOFTMAX, _POLICY_MLP)


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


@lru_cache(maxsize=None)
def _exact_degree_index_tuples(state_dim: int, degree: int) -> np.ndarray:
    return np.array(list(combinations_with_replacement(range(state_dim), degree)), dtype=int)


def _exact_degree_products(x_arr: np.ndarray, degree: int) -> np.ndarray:
    """Return exact-degree monomial products in lexicographic replacement order."""
    n_samples, state_dim = x_arr.shape
    if state_dim == 0:
        return np.empty((n_samples, 0), dtype=float)
    index_tuples = _exact_degree_index_tuples(state_dim, int(degree))
    products = x_arr[:, index_tuples[:, 0]].copy()
    for degree_idx in range(1, int(degree)):
        products *= x_arr[:, index_tuples[:, degree_idx]]
    return products.astype(float)


def _exact_degree_output_dim(state_dim: int, degree: int, *, include_interactions: bool) -> int:
    dim = _validate_state_dim(state_dim)
    monomial_dim = comb(dim + int(degree) - 1, int(degree)) if include_interactions else dim
    return dim + monomial_dim


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
class CubicFeatureMap(FeatureMap):
    """Cubic map with linear terms and exact degree-3 monomials."""

    include_interactions: bool = True
    kind: str = "cubic"

    def transform(self, x_batch: np.ndarray) -> np.ndarray:
        """Return ``[x, x_i*x_j*x_k for i <= j <= k]`` features for each sample."""
        x_arr = _as_2d_float_array(x_batch)
        if not np.isfinite(x_arr).all():
            raise ValueError("x_batch must contain finite values.")
        degree_terms = _exact_degree_products(x_arr, 3) if self.include_interactions else x_arr**3
        return np.concatenate([x_arr, degree_terms], axis=1).astype(float)

    def output_dim(self, state_dim: int) -> int:
        """Return cubic feature width excluding the policy intercept."""
        return _exact_degree_output_dim(state_dim, 3, include_interactions=self.include_interactions)


@dataclass(frozen=True)
class QuarticFeatureMap(FeatureMap):
    """Quartic map with linear terms and exact degree-4 monomials."""

    include_interactions: bool = True
    kind: str = "quartic"

    def transform(self, x_batch: np.ndarray) -> np.ndarray:
        """Return ``[x, x_i*x_j*x_k*x_l for i <= j <= k <= l]`` features for each sample."""
        x_arr = _as_2d_float_array(x_batch)
        if not np.isfinite(x_arr).all():
            raise ValueError("x_batch must contain finite values.")
        degree_terms = _exact_degree_products(x_arr, 4) if self.include_interactions else x_arr**4
        return np.concatenate([x_arr, degree_terms], axis=1).astype(float)

    def output_dim(self, state_dim: int) -> int:
        """Return quartic feature width excluding the policy intercept."""
        return _exact_degree_output_dim(state_dim, 4, include_interactions=self.include_interactions)


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
class MLPPolicy(Policy):
    """Two-layer MLP policy with bounded action $$u = 0.5 - \\sigma(z) \\in (-0.5, 0.5)$$.

    Architecture: $$\\varphi(x) \\to \\text{Linear}(d_{in} \\to H) \\to \\tanh \\to \\text{Linear}(H \\to H) \\to \\tanh \\to \\text{Linear}(H \\to 1) \\to 0.5 - \\sigma(\\cdot)$$.
    Theta is the flat concatenation of ``[W1, b1, W2, b2, W3, b3]`` row-major.
    The MLP has explicit biases, so the feature map output is consumed directly
    (no intercept column is prepended).
    """

    feature_map: FeatureMap = field(default_factory=IdentityFeatureMap)
    hidden: int = 16
    kind: str = _POLICY_MLP

    def __post_init__(self) -> None:
        if int(self.hidden) <= 0:
            raise ValueError("hidden must be positive.")

    def theta_dim(self, state_dim: int) -> int:
        """Return ``d_in*H + H + H*H + H + H + 1`` where ``d_in = dim(varphi(x))``."""
        d_in = self.feature_map.output_dim(state_dim)
        return self._theta_dim_from_d_in(d_in)

    def _theta_dim_from_d_in(self, d_in: int) -> int:
        H = int(self.hidden)
        return d_in * H + H + H * H + H + H + 1

    def _unpack(
        self, theta: np.ndarray, d_in: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        H = int(self.hidden)
        idx = 0
        W1 = theta[idx : idx + d_in * H].reshape(d_in, H)
        idx += d_in * H
        b1 = theta[idx : idx + H]
        idx += H
        W2 = theta[idx : idx + H * H].reshape(H, H)
        idx += H * H
        b2 = theta[idx : idx + H]
        idx += H
        W3 = theta[idx : idx + H].reshape(H, 1)
        idx += H
        b3 = theta[idx : idx + 1]
        return W1, b1, W2, b2, W3, b3

    def _features(self, x_batch: np.ndarray) -> np.ndarray:
        x_arr = _as_2d_float_array(x_batch)
        if not np.isfinite(x_arr).all():
            raise ValueError("x_batch must contain finite values.")
        features = self.feature_map.transform(x_arr)
        return _validate_feature_output(features, x_arr)

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return MLP action values, shape ``(n_samples,)``."""
        phi = self._features(x_batch)
        d_in = phi.shape[1]
        theta_arr = _validate_theta(theta, expected_dim=self._theta_dim_from_d_in(d_in))
        W1, b1, W2, b2, W3, b3 = self._unpack(theta_arr, d_in)
        h1 = np.tanh(phi @ W1 + b1)
        h2 = np.tanh(h1 @ W2 + b2)
        z3 = (h2 @ W3 + b3).ravel()
        return (0.5 - _sigmoid(z3)).astype(float)

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return Jacobian ``du/dtheta``, shape ``(n_samples, theta_dim)``, via manual backprop."""
        phi = self._features(x_batch)
        n_samples, d_in = phi.shape
        H = int(self.hidden)
        theta_arr = _validate_theta(theta, expected_dim=self._theta_dim_from_d_in(d_in))
        W1, b1, W2, b2, W3, b3 = self._unpack(theta_arr, d_in)

        # Forward (cached)
        z1 = phi @ W1 + b1
        h1 = np.tanh(z1)
        z2 = h1 @ W2 + b2
        h2 = np.tanh(z2)
        z3 = (h2 @ W3 + b3).ravel()
        sigma = _sigmoid(z3)
        du_dz3 = -sigma * (1.0 - sigma)  # (n,)

        # Layer 3: z3 = h2 @ W3 + b3, W3 has shape (H, 1)
        dW3_flat = du_dz3[:, None] * h2  # (n, H) — matches W3.ravel() row-major over (H, 1)
        db3_flat = du_dz3[:, None]  # (n, 1)

        # Layer 2: backprop tanh, z2 = h1 @ W2 + b2
        dh2 = du_dz3[:, None] * W3.ravel()[None, :]  # (n, H)
        dz2 = dh2 * (1.0 - h2 * h2)  # (n, H)
        dW2 = h1[:, :, None] * dz2[:, None, :]  # (n, H, H), W2 row-major: idx = j*H + k
        dW2_flat = dW2.reshape(n_samples, H * H)
        db2_flat = dz2  # (n, H)

        # Layer 1: backprop tanh, z1 = phi @ W1 + b1
        dh1 = dz2 @ W2.T  # (n, H)
        dz1 = dh1 * (1.0 - h1 * h1)  # (n, H)
        dW1 = phi[:, :, None] * dz1[:, None, :]  # (n, d_in, H), W1 row-major: idx = j*H + k
        dW1_flat = dW1.reshape(n_samples, d_in * H)
        db1_flat = dz1  # (n, H)

        return np.concatenate(
            [dW1_flat, db1_flat, dW2_flat, db2_flat, dW3_flat, db3_flat],
            axis=1,
        )


def mlp_init_theta(rng: np.random.Generator, *, d_in: int, hidden: int = 16) -> np.ndarray:
    """Glorot-uniform init for an `MLPPolicy` of the given input/hidden width.

    `d_in` is the post-feature-map input width (e.g., the value returned by
    `feature_map.output_dim(state_dim)` or `acceptance_model.policy_feature_dim()`).
    Biases are initialized to zero.
    """
    d_in_int = int(d_in)
    H = int(hidden)
    if d_in_int <= 0:
        raise ValueError("d_in must be positive.")
    if H <= 0:
        raise ValueError("hidden must be positive.")

    def _glorot(fan_in: int, fan_out: int) -> np.ndarray:
        scale = float(np.sqrt(6.0 / (fan_in + fan_out)))
        return rng.uniform(-scale, scale, size=(fan_in, fan_out))

    W1 = _glorot(d_in_int, H)
    b1 = np.zeros(H, dtype=float)
    W2 = _glorot(H, H)
    b2 = np.zeros(H, dtype=float)
    W3 = _glorot(H, 1)
    b3 = np.zeros(1, dtype=float)
    return np.concatenate(
        [W1.ravel(), b1, W2.ravel(), b2, W3.ravel(), b3]
    ).astype(float)


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


def policy_from_kind(kind: str) -> ConstantPolicy | LinearPolicy | SoftmaxPolicy | MLPPolicy:
    """Create a policy instance from a kind string."""
    if kind == _POLICY_CONSTANT:
        return ConstantPolicy()
    if kind == _POLICY_LINEAR:
        return LinearPolicy()
    if kind == _POLICY_SOFTMAX:
        return SoftmaxPolicy()
    if kind == _POLICY_MLP:
        return MLPPolicy()
    raise ValueError(f"Policy kind must be one of {_POLICY_KINDS}.")


__all__ = [
    "CallableFeatureMap",
    "ConstantPolicy",
    "CubicFeatureMap",
    "FeatureMap",
    "FeatureProcessedPolicy",
    "IdentityFeatureMap",
    "LinearPolicy",
    "MLPPolicy",
    "QuadraticFeatureMap",
    "QuarticFeatureMap",
    "SoftmaxPolicy",
    "mlp_init_theta",
    "policy_from_kind",
    "policy_theta_dim",
]
