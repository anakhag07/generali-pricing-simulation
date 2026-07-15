"""Action-space regularizer wrapper for policy objectives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

import numpy as np

from objective.base import Objective, Policy
from objective.noise import HeteroskedasticGaussianNoise, HomoskedasticGaussianNoise, NoNoise
from objective.utils import _policy_value


class SigmaProvider(Protocol):
    """Provider for uncertainty scale regularization."""

    def values_and_du_grad(self, x_batch: Any, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``sigma(x, u)`` and ``d sigma / d u`` with the same shape as ``u``."""
        ...


@dataclass(frozen=True)
class CallableSigmaProvider:
    """Adapter for an injected ``sigma_fn(x_batch, u)`` callable."""

    sigma_fn: Callable[[Any, np.ndarray], tuple[np.ndarray, np.ndarray]]

    def values_and_du_grad(self, x_batch: Any, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u_arr = _validate_u_field(u)
        values, du_grad = self.sigma_fn(x_batch, u_arr)
        return _validate_sigma_outputs(values, du_grad, u_arr.shape)


@dataclass(frozen=True)
class HomoskedasticNoiseScaleProvider:
    """Constant uncertainty-scale provider with zero action-gradient."""

    std: float = 0.0

    def __post_init__(self) -> None:
        std = float(self.std)
        if not np.isfinite(std) or std < 0.0:
            raise ValueError("std must be finite and nonnegative.")
        object.__setattr__(self, "std", std)

    def values_and_du_grad(self, x_batch: Any, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del x_batch
        u_arr = _validate_u_field(u)
        return (
            np.full_like(u_arr, self.std, dtype=float),
            np.zeros_like(u_arr, dtype=float),
        )


@dataclass(frozen=True)
class HeteroskedasticNoiseScaleProvider:
    """Oracle scale provider for ``HeteroskedasticGaussianNoise``."""

    base_std: float = 0.0
    growth: float = 1.0
    u_center: float = 0.0

    @classmethod
    def from_noise(cls, noise: HeteroskedasticGaussianNoise) -> "HeteroskedasticNoiseScaleProvider":
        return cls(base_std=noise.base_std, growth=noise.growth, u_center=noise.u_center)

    def __post_init__(self) -> None:
        base_std = float(self.base_std)
        growth = float(self.growth)
        u_center = float(self.u_center)
        if not np.isfinite(base_std) or base_std < 0.0:
            raise ValueError("base_std must be finite and nonnegative.")
        if not np.isfinite(growth) or growth < 0.0:
            raise ValueError("growth must be finite and nonnegative.")
        if not np.isfinite(u_center):
            raise ValueError("u_center must be finite.")
        object.__setattr__(self, "base_std", base_std)
        object.__setattr__(self, "growth", growth)
        object.__setattr__(self, "u_center", u_center)

    def values_and_du_grad(self, x_batch: Any, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        del x_batch
        u_arr = _validate_u_field(u)
        delta = u_arr - self.u_center
        values = self.base_std + self.growth * np.abs(delta)
        du_grad = self.growth * np.sign(delta)
        du_grad = np.where(delta == 0.0, 0.0, du_grad)
        return values, du_grad


@dataclass(frozen=True)
class ActionRegularizedObjective(Objective):
    r"""Objective wrapper adding proximal and support-aware action penalties."""

    base_objective: Objective
    proximal_weight: float | None = None
    u_reference: np.ndarray | None = None
    support_weight: float | None = None
    sigma_provider: SigmaProvider | None = None
    policy: Policy | None = None

    def __post_init__(self) -> None:
        proximal_weight = _validate_optional_weight(self.proximal_weight, "proximal_weight")
        support_weight = _validate_optional_weight(self.support_weight, "support_weight")
        object.__setattr__(self, "proximal_weight", proximal_weight)
        object.__setattr__(self, "support_weight", support_weight)

        u_reference = None
        if self.u_reference is not None:
            u_reference = _validate_u_reference(self.u_reference)
        if proximal_weight is not None and u_reference is None:
            raise ValueError("proximal_weight requires row-aligned u_reference.")
        object.__setattr__(self, "u_reference", u_reference)

        sigma_provider = self.sigma_provider
        if support_weight is not None and sigma_provider is None:
            sigma_provider = default_sigma_provider_from_objective(self.base_objective)
            if sigma_provider is None:
                raise ValueError(
                    "support_weight requires sigma_provider or a supported noise scale on the base objective."
                )
        object.__setattr__(self, "sigma_provider", sigma_provider)

        base_policy = getattr(self.base_objective, "policy", None)
        if self.policy is None:
            if base_policy is not None:
                object.__setattr__(self, "policy", base_policy)
            return
        if base_policy is self.policy:
            return
        try:
            updated_base = replace(self.base_objective, policy=self.policy)
        except TypeError as exc:
            raise ValueError("base_objective policy could not be replaced.") from exc
        object.__setattr__(self, "base_objective", updated_base)

    def value(self, theta: np.ndarray, x_batch: Any) -> float:
        """Return optimizer-facing objective value with action regularizers."""
        theta_arr = np.asarray(theta, dtype=float)
        base_value = float(self.base_objective.value(theta_arr, x_batch))
        raw_u = _policy_value(self.base_objective, theta_arr, x_batch).reshape(-1)
        u_arr = self._clip_u(raw_u).reshape(-1)
        proximal, support, _ = self._regularizer_value_and_weights(x_batch, u_arr)
        return base_value + proximal + support

    def value_on_indices(self, theta: np.ndarray, x_array: Any, indices: np.ndarray) -> float:
        """Return value on an optimizer-owned mini-batch with source-row indices."""
        x_batch = _slice_rows(x_array, indices)
        theta_arr = np.asarray(theta, dtype=float)
        value_on_indices_fn = getattr(self.base_objective, "value_on_indices", None)
        if callable(value_on_indices_fn):
            base_value = float(value_on_indices_fn(theta_arr, x_array, indices))
        else:
            base_value = float(self.base_objective.value(theta_arr, x_batch))
        raw_u = _policy_value(self.base_objective, theta_arr, x_batch).reshape(-1)
        u_arr = self._clip_u(raw_u).reshape(-1)
        proximal, support, _ = self._regularizer_value_and_weights(x_batch, u_arr, indices=indices)
        return base_value + proximal + support

    def base_value(self, theta: np.ndarray, x_batch: Any) -> float:
        """Return wrapped raw objective value for reporting."""
        base_value_fn = getattr(self.base_objective, "base_value", None)
        if callable(base_value_fn):
            return float(base_value_fn(theta, x_batch))
        return float(self.base_objective.value(theta, x_batch))

    def grad(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        """Return theta-gradient with action-regularizer gradients added."""
        theta_arr = np.asarray(theta, dtype=float)
        base_grad = np.asarray(self.base_objective.grad(theta_arr, x_batch), dtype=float)
        raw_u = _policy_value(self.base_objective, theta_arr, x_batch).reshape(-1)
        u_arr = self._clip_u(raw_u).reshape(-1)
        _, _, weights = self._regularizer_value_and_weights(x_batch, u_arr)
        weights = self._clip_derivative_weights(weights, raw_u)
        if np.all(weights == 0.0):
            return base_grad
        return base_grad + self.policy_weighted_grad(theta_arr, x_batch, weights)

    def grad_on_indices(self, theta: np.ndarray, x_array: Any, indices: np.ndarray) -> np.ndarray:
        """Return gradient on an optimizer-owned mini-batch with source-row indices."""
        x_batch = _slice_rows(x_array, indices)
        theta_arr = np.asarray(theta, dtype=float)
        grad_on_indices_fn = getattr(self.base_objective, "grad_on_indices", None)
        if callable(grad_on_indices_fn):
            base_grad = np.asarray(grad_on_indices_fn(theta_arr, x_array, indices), dtype=float)
        else:
            base_grad = np.asarray(self.base_objective.grad(theta_arr, x_batch), dtype=float)
        raw_u = _policy_value(self.base_objective, theta_arr, x_batch).reshape(-1)
        u_arr = self._clip_u(raw_u).reshape(-1)
        _, _, weights = self._regularizer_value_and_weights(x_batch, u_arr, indices=indices)
        weights = self._clip_derivative_weights(weights, raw_u)
        if np.all(weights == 0.0):
            return base_grad
        return base_grad + self.policy_weighted_grad(theta_arr, x_batch, weights)

    def value_at_u(self, x_batch: Any, u: float) -> float:
        """Return fixed-action value with action regularizers."""
        value_at_u_fn = getattr(self.base_objective, "value_at_u", None)
        if not callable(value_at_u_fn):
            raise ValueError("base_objective does not support value_at_u(x_batch, u).")
        u_val = float(self._clip_u(np.asarray([float(u)], dtype=float))[0])
        u_arr = np.full(_row_count(x_batch), u_val, dtype=float)
        proximal, support = self._regularizer_values(x_batch, u_arr)
        return float(value_at_u_fn(x_batch, u_val)) + proximal + support

    def base_value_at_u(self, x_batch: Any, u: float) -> float:
        """Return wrapped raw fixed-action value for reporting."""
        base_value_at_u_fn = getattr(self.base_objective, "base_value_at_u", None)
        if callable(base_value_at_u_fn):
            return float(base_value_at_u_fn(x_batch, u))
        value_at_u_fn = getattr(self.base_objective, "value_at_u", None)
        if callable(value_at_u_fn):
            return float(value_at_u_fn(x_batch, u))
        raise ValueError("base_objective does not support value_at_u(x_batch, u).")

    def _value_batch(self, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
        """Return per-row regularized action-level values."""
        u_values = self._clip_u(_validate_u_vector(u_arr, _row_count(x_batch)))
        base_values = self._base_value_batch(x_batch, u_values)
        return base_values + self._regularizer_value_terms(x_batch, u_values)

    def _value_batch_on_indices(self, x_batch: Any, indices: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        """Return per-row regularized action values for an optimizer mini-batch."""
        u_values = self._clip_u(_validate_u_vector(u_arr, _row_count(x_batch)))
        value_batch_on_indices_fn = getattr(self.base_objective, "_value_batch_on_indices", None)
        if callable(value_batch_on_indices_fn):
            base_values = np.asarray(value_batch_on_indices_fn(x_batch, indices, u_values), dtype=float)
        else:
            base_values = self._base_value_batch(x_batch, u_values)
        return base_values + self._regularizer_value_terms(x_batch, u_values, indices=indices)

    def _value_batch_many(self, x_batch: Any, u_matrix: np.ndarray) -> np.ndarray:
        """Return regularized action-level values for many action vectors."""
        u_values = self._clip_u(_validate_u_matrix(u_matrix, _row_count(x_batch)))
        base_values = self._base_value_batch_many(x_batch, u_values)
        return base_values + self._regularizer_value_terms(x_batch, u_values)

    def _value_batch_many_on_indices(
        self,
        x_batch: Any,
        indices: np.ndarray,
        u_matrix: np.ndarray,
    ) -> np.ndarray:
        """Return regularized action values for many mini-batch action vectors."""
        u_values = self._clip_u(_validate_u_matrix(u_matrix, _row_count(x_batch)))
        value_batch_many_on_indices_fn = getattr(self.base_objective, "_value_batch_many_on_indices", None)
        if callable(value_batch_many_on_indices_fn):
            base_values = np.asarray(value_batch_many_on_indices_fn(x_batch, indices, u_values), dtype=float)
        else:
            base_values = self._base_value_batch_many(x_batch, u_values)
        return base_values + self._regularizer_value_terms(x_batch, u_values, indices=indices)

    def policy_value(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        """Delegate policy action evaluation to the wrapped objective."""
        return _policy_value(self.base_objective, theta, x_batch)

    def policy_grad(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        """Delegate policy Jacobian evaluation to the wrapped objective."""
        grad_fn = getattr(self.base_objective, "policy_grad", None)
        if callable(grad_fn):
            return np.asarray(grad_fn(theta, x_batch), dtype=float)
        if self.policy is None:
            raise ValueError("base_objective does not expose a policy gradient.")
        return np.asarray(self.policy.grad(theta, x_batch), dtype=float)

    def policy_weighted_grad(self, theta: np.ndarray, x_batch: Any, weights: np.ndarray) -> np.ndarray:
        """Delegate weighted policy-gradient evaluation to the wrapped objective."""
        weighted_grad_fn = getattr(self.base_objective, "policy_weighted_grad", None)
        if callable(weighted_grad_fn):
            return np.asarray(weighted_grad_fn(theta, x_batch, weights), dtype=float)
        if self.policy is None:
            raise ValueError("base_objective does not expose a weighted policy gradient.")
        return np.asarray(self.policy.weighted_grad(theta, x_batch, weights), dtype=float)

    def _step_metrics(self, theta: np.ndarray, x_batch: Any) -> dict[str, float]:
        """Return per-step metrics including separate action-regularizer penalties."""
        metrics_fn = getattr(self.base_objective, "_step_metrics", None)
        metrics = dict(metrics_fn(theta, x_batch)) if callable(metrics_fn) else {}
        raw_u = _policy_value(self.base_objective, np.asarray(theta, dtype=float), x_batch).reshape(-1)
        u_arr = self._clip_u(raw_u).reshape(-1)
        proximal, support, _ = self._regularizer_value_and_weights(x_batch, u_arr)
        metrics["proximal_penalty"] = proximal
        metrics["support_penalty"] = support
        return metrics

    def step_metrics_on_indices(self, theta: np.ndarray, x_array: Any, indices: np.ndarray) -> dict[str, float]:
        """Return per-step metrics for an optimizer-owned mini-batch."""
        x_batch = _slice_rows(x_array, indices)
        metrics_on_indices_fn = getattr(self.base_objective, "step_metrics_on_indices", None)
        if callable(metrics_on_indices_fn):
            metrics = dict(metrics_on_indices_fn(theta, x_array, indices))
        else:
            metrics_fn = getattr(self.base_objective, "_step_metrics", None)
            metrics = dict(metrics_fn(theta, x_batch)) if callable(metrics_fn) else {}
        raw_u = _policy_value(self.base_objective, np.asarray(theta, dtype=float), x_batch).reshape(-1)
        u_arr = self._clip_u(raw_u).reshape(-1)
        proximal, support, _ = self._regularizer_value_and_weights(x_batch, u_arr, indices=indices)
        metrics["proximal_penalty"] = proximal
        metrics["support_penalty"] = support
        return metrics

    def with_noise_seed(self, seed: int) -> "ActionRegularizedObjective":
        """Return a wrapper copy after forwarding a noise seed to the base objective."""
        with_noise_seed = getattr(self.base_objective, "with_noise_seed", None)
        if not callable(with_noise_seed):
            return self
        return replace(self, base_objective=with_noise_seed(int(seed)))

    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        clip_fn = getattr(self.base_objective, "_clip_u", None)
        if callable(clip_fn):
            return np.asarray(clip_fn(u), dtype=float)
        return np.asarray(u, dtype=float)

    def _clip_derivative_weights(self, weights: np.ndarray, raw_u: np.ndarray) -> np.ndarray:
        weights_arr = np.asarray(weights, dtype=float).reshape(-1)
        raw_arr = np.asarray(raw_u, dtype=float).reshape(-1)
        if weights_arr.shape != raw_arr.shape:
            raise ValueError("regularizer weights must have one value per raw action.")
        u_bounds = getattr(self.base_objective, "u_bounds", None)
        if u_bounds is not None:
            lower, upper = float(u_bounds[0]), float(u_bounds[1])
            return weights_arr * ((raw_arr > lower) & (raw_arr < upper))
        clipped = self._clip_u(raw_arr).reshape(-1)
        return weights_arr * (raw_arr == clipped)

    def _regularizer_value_and_weights(
        self,
        x_batch: Any,
        u_arr: np.ndarray,
        *,
        indices: np.ndarray | None = None,
    ) -> tuple[float, float, np.ndarray]:
        u_values = _validate_u_vector(u_arr, _row_count(x_batch))
        terms = self._regularizer_terms(x_batch, u_values, indices=indices)
        proximal_value = float(np.mean(terms.proximal_values))
        support_value = float(np.mean(terms.support_values))
        n_rows = float(u_values.size)
        weights = (
            terms.proximal_du_grad * (0.0 if self.proximal_weight is None else float(self.proximal_weight))
            + terms.support_du_grad * (0.0 if self.support_weight is None else float(self.support_weight))
        ) / n_rows
        return proximal_value, support_value, weights

    def _regularizer_values(
        self,
        x_batch: Any,
        u_arr: np.ndarray,
        *,
        indices: np.ndarray | None = None,
    ) -> tuple[float, float]:
        terms = self._regularizer_terms(x_batch, _validate_u_vector(u_arr, _row_count(x_batch)), indices=indices)
        return float(np.mean(terms.proximal_values)), float(np.mean(terms.support_values))

    def _regularizer_value_terms(
        self,
        x_batch: Any,
        u_arr: np.ndarray,
        *,
        indices: np.ndarray | None = None,
    ) -> np.ndarray:
        terms = self._regularizer_terms(x_batch, u_arr, indices=indices)
        return terms.proximal_values + terms.support_values

    def _regularizer_terms(
        self,
        x_batch: Any,
        u_arr: np.ndarray,
        *,
        indices: np.ndarray | None = None,
    ) -> "_RegularizerTerms":
        u_values = _validate_u_field(u_arr)
        row_count = _u_row_count(u_values)
        proximal_values = np.zeros_like(u_values, dtype=float)
        proximal_du_grad = np.zeros_like(u_values, dtype=float)
        if self.proximal_weight is not None:
            reference = self._u_reference_for_batch(row_count, indices=indices)
            clipped_reference = self._clip_u(reference)
            if u_values.ndim == 1:
                diff = u_values - clipped_reference
            else:
                diff = u_values - clipped_reference[None, :]
            proximal_values = float(self.proximal_weight) * diff * diff
            proximal_du_grad = 2.0 * diff

        support_values = np.zeros_like(u_values, dtype=float)
        support_du_grad = np.zeros_like(u_values, dtype=float)
        if self.support_weight is not None:
            if self.sigma_provider is None:
                raise ValueError("support_weight requires sigma_provider.")
            sigma_values, sigma_du_grad = self.sigma_provider.values_and_du_grad(x_batch, u_values)
            support_values = float(self.support_weight) * sigma_values
            support_du_grad = sigma_du_grad

        return _RegularizerTerms(
            proximal_values=proximal_values,
            proximal_du_grad=proximal_du_grad,
            support_values=support_values,
            support_du_grad=support_du_grad,
        )

    def _u_reference_for_batch(self, row_count: int, *, indices: np.ndarray | None) -> np.ndarray:
        if self.u_reference is None:
            raise ValueError("proximal regularization requires u_reference.")
        reference = np.asarray(self.u_reference, dtype=float).reshape(-1)
        if indices is not None:
            index_arr = np.asarray(indices, dtype=int).reshape(-1)
            if index_arr.shape != (row_count,):
                raise ValueError("indices must have one entry per x_batch row.")
            if index_arr.size == 0:
                raise ValueError("indices must contain at least one row.")
            if np.any(index_arr < 0):
                raise ValueError("indices must be nonnegative.")
            if reference.size > int(np.max(index_arr)):
                return reference[index_arr].copy()
            if reference.size == row_count:
                return reference.copy()
            raise ValueError("u_reference length does not cover mini-batch indices.")
        if reference.shape != (row_count,):
            raise ValueError("u_reference must be aligned to x_batch rows or called with mini-batch indices.")
        return reference.copy()

    def _base_value_batch(self, x_batch: Any, u_values: np.ndarray) -> np.ndarray:
        value_batch_fn = getattr(self.base_objective, "_value_batch", None)
        if callable(value_batch_fn):
            values = np.asarray(value_batch_fn(x_batch, u_values), dtype=float)
            if values.shape != u_values.shape:
                raise ValueError("base objective _value_batch returned unexpected shape.")
            return values
        value_at_u_fn = getattr(self.base_objective, "value_at_u", None)
        if not callable(value_at_u_fn):
            raise ValueError("base_objective must expose _value_batch or value_at_u.")
        values = np.empty_like(u_values, dtype=float)
        for idx, u_val in enumerate(u_values):
            values[idx] = float(value_at_u_fn(_slice_range(x_batch, idx, idx + 1), float(u_val)))
        return values

    def _base_value_batch_many(self, x_batch: Any, u_values: np.ndarray) -> np.ndarray:
        value_many_fn = getattr(self.base_objective, "_value_batch_many", None)
        if callable(value_many_fn):
            base_values = np.asarray(value_many_fn(x_batch, u_values), dtype=float)
        else:
            base_values = np.vstack([self._base_value_batch(x_batch, row) for row in u_values])
        if base_values.shape != u_values.shape:
            raise ValueError("base objective returned unexpected value matrix shape.")
        return base_values

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_objective, name)


@dataclass(frozen=True)
class _RegularizerTerms:
    proximal_values: np.ndarray
    proximal_du_grad: np.ndarray
    support_values: np.ndarray
    support_du_grad: np.ndarray


def default_sigma_provider_from_objective(objective: object) -> SigmaProvider | None:
    """Return a default support-scale provider from a wrapped noise object."""
    current = objective
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        noise = getattr(current, "noise", None)
        if isinstance(noise, HeteroskedasticGaussianNoise):
            return HeteroskedasticNoiseScaleProvider.from_noise(noise)
        if isinstance(noise, HomoskedasticGaussianNoise):
            return HomoskedasticNoiseScaleProvider(std=noise.std)
        if isinstance(noise, NoNoise):
            return HomoskedasticNoiseScaleProvider(std=0.0)
        base = getattr(current, "base_objective", None)
        if base is None:
            return None
        current = base
    return None


def _validate_optional_weight(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    weight = float(value)
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative when provided.")
    return weight


def _validate_u_reference(u_reference: np.ndarray) -> np.ndarray:
    arr = np.asarray(u_reference, dtype=float)
    if arr.ndim != 1:
        raise ValueError("u_reference must be a 1D array.")
    if arr.size == 0:
        raise ValueError("u_reference must contain at least one value.")
    if not np.isfinite(arr).all():
        raise ValueError("u_reference must contain finite values.")
    return arr.copy()


def _validate_u_field(u: np.ndarray) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    if u_arr.ndim not in {1, 2}:
        raise ValueError("u must be a 1D or 2D array.")
    if u_arr.shape[-1] < 1:
        raise ValueError("u must contain at least one action.")
    if not np.isfinite(u_arr).all():
        raise ValueError("u must contain finite values.")
    return u_arr


def _validate_u_vector(u: np.ndarray, row_count: int) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float).reshape(-1)
    if u_arr.shape != (row_count,):
        raise ValueError("u_arr must have one value per x_batch row.")
    if not np.isfinite(u_arr).all():
        raise ValueError("u_arr must contain finite values.")
    return u_arr


def _validate_u_matrix(u: np.ndarray, row_count: int) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    if u_arr.ndim != 2 or u_arr.shape[1] != row_count:
        raise ValueError("u_matrix must have shape (n_evaluations, n_rows).")
    if not np.isfinite(u_arr).all():
        raise ValueError("u_matrix must contain finite values.")
    return u_arr


def _validate_sigma_outputs(
    values: np.ndarray,
    du_grad: np.ndarray,
    expected_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    values_arr = np.asarray(values, dtype=float)
    grad_arr = np.asarray(du_grad, dtype=float)
    if values_arr.shape != expected_shape:
        raise ValueError("sigma values must have the same shape as u.")
    if grad_arr.shape != expected_shape:
        raise ValueError("sigma du_grad must have the same shape as u.")
    if not np.isfinite(values_arr).all() or not np.isfinite(grad_arr).all():
        raise ValueError("sigma values and du_grad must be finite.")
    return values_arr, grad_arr


def _row_count(x_batch: Any) -> int:
    return int(x_batch.shape[0])


def _u_row_count(u_values: np.ndarray) -> int:
    if u_values.ndim == 1:
        return int(u_values.shape[0])
    return int(u_values.shape[1])


def _slice_rows(x_array: Any, indices: np.ndarray) -> Any:
    index_arr = np.asarray(indices, dtype=int).reshape(-1)
    if index_arr.size == _row_count(x_array) and np.array_equal(index_arr, np.arange(index_arr.size, dtype=int)):
        return x_array
    if hasattr(x_array, "iloc"):
        return x_array.iloc[index_arr].reset_index(drop=True)
    return np.asarray(x_array)[index_arr]


def _slice_range(x_batch: Any, start: int, stop: int) -> Any:
    if hasattr(x_batch, "iloc"):
        return x_batch.iloc[start:stop].reset_index(drop=True)
    return np.asarray(x_batch)[start:stop]


__all__ = [
    "ActionRegularizedObjective",
    "CallableSigmaProvider",
    "HeteroskedasticNoiseScaleProvider",
    "HomoskedasticNoiseScaleProvider",
    "SigmaProvider",
    "default_sigma_provider_from_objective",
]
