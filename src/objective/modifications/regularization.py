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
class ConstantThetaRegularizer(ThetaRegularizer):
    """Constant theta offset $$\phi(\theta)=h$$."""

    height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "height", _nonnegative(self.height, "height"))

    def value(self, theta: np.ndarray) -> float:
        _validate_theta(theta)
        return float(self.height)

    def grad(self, theta: np.ndarray) -> np.ndarray:
        theta_arr = _validate_theta(theta)
        return np.zeros_like(theta_arr, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "ConstantThetaRegularizer",
            "height": float(self.height),
        }


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
class IntervalDistanceThetaRegularizer(ThetaRegularizer):
    r"""Linear distance penalty $$\phi(\theta)=\lambda\,d(\theta,[\ell,h])$$."""

    slope: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower, upper = _interval(self.lower, self.upper)
        object.__setattr__(self, "slope", _nonnegative(self.slope, "slope"))
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def value(self, theta: np.ndarray) -> float:
        theta_arr = _validate_theta(theta)
        distance, _ = _interval_distance_and_direction(theta_arr, self.lower, self.upper)
        return float(self.slope) * float(np.mean(distance))

    def grad(self, theta: np.ndarray) -> np.ndarray:
        theta_arr = _validate_theta(theta)
        _, direction = _interval_distance_and_direction(theta_arr, self.lower, self.upper)
        return (float(self.slope) / float(theta_arr.size)) * direction

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "IntervalDistanceThetaRegularizer",
            "slope": float(self.slope),
            "lower": float(self.lower),
            "upper": float(self.upper),
        }


@dataclass(frozen=True)
class SmoothSaturatingIntervalThetaRegularizer(ThetaRegularizer):
    r"""Bounded $$C^\infty$$ support envelope increasing outside $$[\ell,h]$$.

    For distance $$d=d(\theta,[\ell,h])$$, the per-coordinate term is zero
    inside the interval and
    $$A\exp[-(s/d)^2]$$ outside it.
    """

    amplitude: float
    transition_width: float
    lower: float
    upper: float

    def __post_init__(self) -> None:
        lower, upper = _interval(self.lower, self.upper)
        amplitude = _nonnegative(self.amplitude, "amplitude")
        transition_width = float(self.transition_width)
        if not np.isfinite(transition_width) or transition_width <= 0.0:
            raise ValueError("transition_width must be finite and positive.")
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "transition_width", transition_width)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    def value(self, theta: np.ndarray) -> float:
        theta_arr = _validate_theta(theta)
        distance, _ = _interval_distance_and_direction(theta_arr, self.lower, self.upper)
        return float(self.amplitude) * float(
            np.mean(_smooth_support_escape(distance, self.transition_width))
        )

    def grad(self, theta: np.ndarray) -> np.ndarray:
        theta_arr = _validate_theta(theta)
        distance, direction = _interval_distance_and_direction(
            theta_arr, self.lower, self.upper
        )
        slope = np.zeros_like(distance, dtype=float)
        outside = distance > 0.0
        if np.any(outside) and self.amplitude > 0.0:
            z = float(self.transition_width) / distance[outside]
            # For z >= 40 the exact double-precision slope underflows to zero.
            active = z < 40.0
            active_z = z[active]
            outside_slope = np.zeros_like(z, dtype=float)
            outside_slope[active] = (
                2.0
                * float(self.amplitude)
                / float(self.transition_width)
                * active_z**3
                * np.exp(-(active_z**2))
            )
            slope[outside] = outside_slope
        return direction * slope / float(theta_arr.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "SmoothSaturatingIntervalThetaRegularizer",
            "amplitude": float(self.amplitude),
            "transition_width": float(self.transition_width),
            "lower": float(self.lower),
            "upper": float(self.upper),
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
    if kind in {"ConstantThetaRegularizer", "constant"}:
        return ConstantThetaRegularizer(height=float(payload["height"]))
    if kind in {"IntervalDistanceThetaRegularizer", "interval_distance"}:
        return IntervalDistanceThetaRegularizer(
            slope=float(payload["slope"]),
            lower=float(payload["lower"]),
            upper=float(payload["upper"]),
        )
    if kind in {
        "SmoothSaturatingIntervalThetaRegularizer",
        "smooth_saturating_interval",
    }:
        return SmoothSaturatingIntervalThetaRegularizer(
            amplitude=float(payload["amplitude"]),
            transition_width=float(payload["transition_width"]),
            lower=float(payload["lower"]),
            upper=float(payload["upper"]),
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


def _interval(lower: float, upper: float) -> tuple[float, float]:
    lower_float = float(lower)
    upper_float = float(upper)
    if not np.isfinite(lower_float) or not np.isfinite(upper_float):
        raise ValueError("lower and upper must be finite.")
    if lower_float >= upper_float:
        raise ValueError("lower must be strictly less than upper.")
    return lower_float, upper_float


def _interval_distance_and_direction(
    theta: np.ndarray,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    below = theta < float(lower)
    above = theta > float(upper)
    distance = np.where(
        below,
        float(lower) - theta,
        np.where(above, theta - float(upper), 0.0),
    )
    direction = np.where(below, -1.0, np.where(above, 1.0, 0.0))
    return distance, direction


def _smooth_support_escape(distance: np.ndarray, transition_width: float) -> np.ndarray:
    values = np.zeros_like(distance, dtype=float)
    outside = distance > 0.0
    if np.any(outside):
        z = float(transition_width) / distance[outside]
        values[outside] = np.exp(-(np.minimum(z, 40.0) ** 2))
    return values


__all__ = [
    "ConstantThetaRegularizer",
    "IntervalDistanceThetaRegularizer",
    "ProximalThetaRegularizer",
    "RegularizedObjective",
    "SmoothSaturatingIntervalThetaRegularizer",
    "SupportThetaRegularizer",
    "ThetaRegularizer",
    "regularizer_from_dict",
    "regularizer_to_dict",
]
