"""Theta-space regularization objective modifications."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from objective.base import Objective, Policy


class ThetaRegularizer:
    """Additive regularizer over the theta decision vector."""

    def value(self, theta: np.ndarray) -> float:
        """Return the scalar regularization value at ``theta``."""
        raise NotImplementedError

    def grad(self, theta: np.ndarray) -> np.ndarray:
        """Return the theta-gradient of the regularization value."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialize regularizer parameters for run summaries."""
        raise NotImplementedError


@dataclass(frozen=True)
class ProximalThetaRegularizer(ThetaRegularizer):
    """Mean-squared proximal penalty ``weight * mean((theta - reference)^2)``."""

    weight: float
    reference: np.ndarray | None = None

    def __post_init__(self) -> None:
        weight = _nonnegative(self.weight, "weight")
        reference = None if self.reference is None else np.asarray(self.reference, dtype=float).reshape(-1)
        if reference is not None and (reference.ndim != 1 or not np.all(np.isfinite(reference))):
            raise ValueError("reference must be a finite 1D array.")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "reference", reference)

    def value(self, theta: np.ndarray) -> float:
        theta_arr = _validate_theta(theta)
        ref = self._reference_for(theta_arr)
        return float(self.weight) * float(np.mean((theta_arr - ref) ** 2))

    def grad(self, theta: np.ndarray) -> np.ndarray:
        theta_arr = _validate_theta(theta)
        ref = self._reference_for(theta_arr)
        return (2.0 * float(self.weight) / float(theta_arr.size)) * (theta_arr - ref)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ProximalThetaRegularizer",
            "weight": float(self.weight),
            "reference": None if self.reference is None else self.reference.tolist(),
        }

    def _reference_for(self, theta: np.ndarray) -> np.ndarray:
        if self.reference is None:
            return np.zeros_like(theta, dtype=float)
        if self.reference.shape != theta.shape:
            raise ValueError(
                "reference must have the same length as theta "
                f"({self.reference.size} != {theta.size})."
            )
        return self.reference


@dataclass(frozen=True)
class SupportThetaRegularizer(ThetaRegularizer):
    """Mean absolute support proxy ``weight * mean(growth * abs(theta - center))``."""

    weight: float
    support_center: float = 0.0
    support_growth: float = 1.0

    def __post_init__(self) -> None:
        weight = _nonnegative(self.weight, "weight")
        support_center = float(self.support_center)
        support_growth = _nonnegative(self.support_growth, "support_growth")
        if not np.isfinite(support_center):
            raise ValueError("support_center must be finite.")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "support_center", support_center)
        object.__setattr__(self, "support_growth", support_growth)

    def value(self, theta: np.ndarray) -> float:
        theta_arr = _validate_theta(theta)
        sigma = float(self.support_growth) * np.abs(theta_arr - float(self.support_center))
        return float(self.weight) * float(np.mean(sigma))

    def grad(self, theta: np.ndarray) -> np.ndarray:
        theta_arr = _validate_theta(theta)
        delta = theta_arr - float(self.support_center)
        support_grad = float(self.support_growth) * np.sign(delta)
        support_grad = np.where(delta == 0.0, 0.0, support_grad)
        return (float(self.weight) / float(theta_arr.size)) * support_grad

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "SupportThetaRegularizer",
            "weight": float(self.weight),
            "support_center": float(self.support_center),
            "support_growth": float(self.support_growth),
        }


@dataclass(frozen=True)
class RegularizedObjective(Objective):
    """Objective wrapper adding one or more theta-space regularizers."""

    base_objective: Objective
    regularizers: tuple[ThetaRegularizer, ...]
    policy: Policy | None = None

    def __post_init__(self) -> None:
        regularizers = tuple(_coerce_regularizer(regularizer) for regularizer in self.regularizers)
        if not regularizers:
            raise ValueError("RegularizedObjective requires at least one regularizer.")
        object.__setattr__(self, "regularizers", regularizers)

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

    def with_noise_seed(self, seed: int) -> "RegularizedObjective":
        """Forward noise seeding into the wrapped objective when supported."""
        with_noise_seed = getattr(self.base_objective, "with_noise_seed", None)
        if not callable(with_noise_seed):
            return self
        return replace(self, base_objective=with_noise_seed(int(seed)))

    def theta_dim(self, state_dim: int | None = None) -> int:
        theta_dim_fn = getattr(self.base_objective, "theta_dim", None)
        if callable(theta_dim_fn):
            return int(theta_dim_fn(state_dim))
        raise ValueError("base_objective does not expose theta_dim.")

    def value(self, theta: np.ndarray, x_batch: Any) -> float:
        theta_arr = _validate_theta(theta)
        return float(self.base_objective.value(theta_arr, x_batch)) + self._regularizer_value(theta_arr)

    def base_value(self, theta: np.ndarray, x_batch: Any) -> float:
        base_value_fn = getattr(self.base_objective, "base_value", None)
        if callable(base_value_fn):
            return float(base_value_fn(theta, x_batch))
        return float(self.base_objective.value(theta, x_batch))

    def grad(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        theta_arr = _validate_theta(theta)
        base_grad = np.asarray(self.base_objective.grad(theta_arr, x_batch), dtype=float)
        if base_grad.shape != theta_arr.shape:
            raise ValueError("base objective gradient shape does not match theta.")
        return base_grad + self._regularizer_grad(theta_arr)

    def policy_value(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        value_fn = getattr(self.base_objective, "policy_value", None)
        if callable(value_fn):
            return np.asarray(value_fn(theta, x_batch), dtype=float)
        if self.policy is None:
            raise ValueError("base_objective does not expose a policy value.")
        return np.asarray(self.policy.value(theta, x_batch), dtype=float)

    def policy_grad(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        grad_fn = getattr(self.base_objective, "policy_grad", None)
        if callable(grad_fn):
            return np.asarray(grad_fn(theta, x_batch), dtype=float)
        if self.policy is None:
            raise ValueError("base_objective does not expose a policy gradient.")
        return np.asarray(self.policy.grad(theta, x_batch), dtype=float)

    def policy_weighted_grad(self, theta: np.ndarray, x_batch: Any, weights: np.ndarray) -> np.ndarray:
        weighted_grad_fn = getattr(self.base_objective, "policy_weighted_grad", None)
        if callable(weighted_grad_fn):
            return np.asarray(weighted_grad_fn(theta, x_batch, weights), dtype=float)
        if self.policy is None:
            raise ValueError("base_objective does not expose a weighted policy gradient.")
        return np.asarray(self.policy.weighted_grad(theta, x_batch, weights), dtype=float)

    def _value_batch(self, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
        del x_batch, u_arr
        raise ValueError("theta regularization cannot be evaluated as per-row action values.")

    def _value_batch_many(self, x_batch: Any, u_matrix: np.ndarray) -> np.ndarray:
        del x_batch, u_matrix
        raise ValueError("theta regularization cannot be evaluated as per-row action values.")

    def _regularizer_value(self, theta: np.ndarray) -> float:
        return float(sum(regularizer.value(theta) for regularizer in self.regularizers))

    def _regularizer_grad(self, theta: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(theta, dtype=float)
        for regularizer in self.regularizers:
            regularizer_grad = np.asarray(regularizer.grad(theta), dtype=float)
            if regularizer_grad.shape != theta.shape:
                raise ValueError("regularizer gradient shape does not match theta.")
            grad += regularizer_grad
        return grad

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "RegularizedObjective",
            "regularizers": [regularizer.to_dict() for regularizer in self.regularizers],
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_objective, name)


def regularizer_to_dict(regularizer: ThetaRegularizer) -> dict[str, Any]:
    """Serialize a theta regularizer."""
    return _coerce_regularizer(regularizer).to_dict()


def regularizer_from_dict(payload: dict[str, Any]) -> ThetaRegularizer:
    """Build a regularizer from a summary/config payload."""
    kind = str(payload.get("type") or payload.get("kind") or "")
    if kind in {"ProximalThetaRegularizer", "proximal"}:
        return ProximalThetaRegularizer(
            weight=float(payload["weight"]),
            reference=payload.get("reference"),
        )
    if kind in {"SupportThetaRegularizer", "support"}:
        return SupportThetaRegularizer(
            weight=float(payload["weight"]),
            support_center=float(payload.get("support_center", 0.0)),
            support_growth=float(payload.get("support_growth", 1.0)),
        )
    raise ValueError(f"Unknown theta regularizer type: {kind!r}.")


def _coerce_regularizer(regularizer: ThetaRegularizer | dict[str, Any]) -> ThetaRegularizer:
    if isinstance(regularizer, ThetaRegularizer):
        return regularizer
    if isinstance(regularizer, dict):
        return regularizer_from_dict(regularizer)
    raise TypeError(f"Unsupported theta regularizer: {type(regularizer).__name__}.")


def _validate_theta(theta: np.ndarray) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    if theta_arr.ndim != 1 or theta_arr.size == 0:
        raise ValueError("theta must be a nonempty 1D array.")
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta must contain only finite values.")
    return theta_arr


def _nonnegative(value: float, name: str) -> float:
    value_float = float(value)
    if not np.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value_float


__all__ = [
    "ProximalThetaRegularizer",
    "RegularizedObjective",
    "SupportThetaRegularizer",
    "ThetaRegularizer",
    "regularizer_from_dict",
    "regularizer_to_dict",
]
