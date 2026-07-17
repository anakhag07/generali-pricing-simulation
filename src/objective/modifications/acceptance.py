"""Acceptance-floor scalar objective modifications."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from objective.base import Objective, Policy


@dataclass(frozen=True)
class AcceptancePenaltyObjective(Objective):
    """Add a smooth scalar penalty for violating a mean-acceptance floor."""

    base_objective: Objective
    acceptance_floor: float
    acceptance_penalty_weight: float
    acceptance_penalty_temperature: float = 0.01
    policy: Policy | None = None

    def __post_init__(self) -> None:
        floor = _acceptance_floor(self.acceptance_floor)
        weight = _positive(self.acceptance_penalty_weight, "acceptance_penalty_weight")
        temperature = _positive(self.acceptance_penalty_temperature, "acceptance_penalty_temperature")
        object.__setattr__(self, "acceptance_floor", floor)
        object.__setattr__(self, "acceptance_penalty_weight", weight)
        object.__setattr__(self, "acceptance_penalty_temperature", temperature)
        _attach_policy(self, self.base_objective)

    def with_noise_seed(self, seed: int) -> "AcceptancePenaltyObjective":
        with_noise_seed = getattr(self.base_objective, "with_noise_seed", None)
        if not callable(with_noise_seed):
            return self
        return replace(self, base_objective=with_noise_seed(int(seed)))

    def value(self, theta: np.ndarray, x_batch: Any) -> float:
        acceptance = self._mean_acceptance(theta, x_batch)
        penalty_value, _ = self._penalty(acceptance)
        return float(self.base_objective.value(theta, x_batch)) + penalty_value

    def base_value(self, theta: np.ndarray, x_batch: Any) -> float:
        base_value_fn = getattr(self.base_objective, "base_value", None)
        if callable(base_value_fn):
            return float(base_value_fn(theta, x_batch))
        return float(self.base_objective.value(theta, x_batch))

    def grad(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        base_grad = np.asarray(self.base_objective.grad(theta, x_batch), dtype=float)
        acceptance = self._mean_acceptance(theta, x_batch)
        _, scale = self._penalty(acceptance)
        return base_grad + scale * self._mean_acceptance_grad(theta, x_batch)

    def value_at_u(self, x_batch: Any, u: float) -> float:
        value_at_u_fn = getattr(self.base_objective, "value_at_u", None)
        if not callable(value_at_u_fn):
            raise ValueError("base_objective does not support value_at_u(x_batch, u).")
        mean_acceptance_at_u_fn = getattr(self.base_objective, "mean_acceptance_at_u", None)
        if not callable(mean_acceptance_at_u_fn):
            raise ValueError("acceptance penalty requires mean_acceptance_at_u(x_batch, u).")
        penalty_value, _ = self._penalty(float(mean_acceptance_at_u_fn(x_batch, u)))
        return float(value_at_u_fn(x_batch, u)) + penalty_value

    def base_value_at_u(self, x_batch: Any, u: float) -> float:
        base_value_at_u_fn = getattr(self.base_objective, "base_value_at_u", None)
        if callable(base_value_at_u_fn):
            return float(base_value_at_u_fn(x_batch, u))
        value_at_u_fn = getattr(self.base_objective, "value_at_u", None)
        if callable(value_at_u_fn):
            return float(value_at_u_fn(x_batch, u))
        raise ValueError("base_objective does not support value_at_u(x_batch, u).")

    def _value_batch(self, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
        del x_batch, u_arr
        raise ValueError("acceptance-floor scalarization cannot be evaluated as per-row action values.")

    def _value_batch_many(self, x_batch: Any, u_matrix: np.ndarray) -> np.ndarray:
        del x_batch, u_matrix
        raise ValueError("acceptance-floor scalarization cannot be evaluated as per-row action values.")

    def _mean_acceptance(self, theta: np.ndarray, x_batch: Any) -> float:
        mean_acceptance_fn = getattr(self.base_objective, "mean_acceptance", None)
        if not callable(mean_acceptance_fn):
            raise ValueError("acceptance penalty requires mean_acceptance(theta, x_batch).")
        return float(mean_acceptance_fn(theta, x_batch))

    def _mean_acceptance_grad(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        mean_acceptance_grad_fn = getattr(self.base_objective, "mean_acceptance_grad", None)
        if not callable(mean_acceptance_grad_fn):
            raise ValueError("acceptance penalty requires mean_acceptance_grad(theta, x_batch).")
        return np.asarray(mean_acceptance_grad_fn(theta, x_batch), dtype=float)

    def _penalty(self, mean_acceptance: float) -> tuple[float, float]:
        gap = float(self.acceptance_floor) - float(mean_acceptance)
        temp = float(self.acceptance_penalty_temperature)
        scaled_gap = gap / temp
        soft_gap = temp * float(np.logaddexp(0.0, scaled_gap))
        sigmoid_gap = 1.0 / (1.0 + np.exp(-scaled_gap))
        weight = float(self.acceptance_penalty_weight)
        penalty_value = weight * soft_gap * soft_gap
        penalty_grad_mean = -2.0 * weight * soft_gap * sigmoid_gap
        return float(penalty_value), float(penalty_grad_mean)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "AcceptancePenaltyObjective",
            "acceptance_floor": float(self.acceptance_floor),
            "acceptance_penalty_weight": float(self.acceptance_penalty_weight),
            "acceptance_penalty_temperature": float(self.acceptance_penalty_temperature),
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_objective, name)


@dataclass(frozen=True)
class AcceptanceLagrangianObjective(Objective):
    """Add a scalar Lagrangian term for a mean-acceptance floor."""

    base_objective: Objective
    acceptance_floor: float
    lagrangian_lambda: float
    policy: Policy | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "acceptance_floor", _acceptance_floor(self.acceptance_floor))
        object.__setattr__(self, "lagrangian_lambda", _nonnegative(self.lagrangian_lambda, "lagrangian_lambda"))
        _attach_policy(self, self.base_objective)

    def with_noise_seed(self, seed: int) -> "AcceptanceLagrangianObjective":
        with_noise_seed = getattr(self.base_objective, "with_noise_seed", None)
        if not callable(with_noise_seed):
            return self
        return replace(self, base_objective=with_noise_seed(int(seed)))

    def value(self, theta: np.ndarray, x_batch: Any) -> float:
        mean_acceptance = self._mean_acceptance(theta, x_batch)
        return float(self.base_objective.value(theta, x_batch)) + self._lagrangian_value(mean_acceptance)

    def base_value(self, theta: np.ndarray, x_batch: Any) -> float:
        base_value_fn = getattr(self.base_objective, "base_value", None)
        if callable(base_value_fn):
            return float(base_value_fn(theta, x_batch))
        return float(self.base_objective.value(theta, x_batch))

    def grad(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        base_grad = np.asarray(self.base_objective.grad(theta, x_batch), dtype=float)
        return base_grad - float(self.lagrangian_lambda) * self._mean_acceptance_grad(theta, x_batch)

    def value_at_u(self, x_batch: Any, u: float) -> float:
        value_at_u_fn = getattr(self.base_objective, "value_at_u", None)
        if not callable(value_at_u_fn):
            raise ValueError("base_objective does not support value_at_u(x_batch, u).")
        mean_acceptance_at_u_fn = getattr(self.base_objective, "mean_acceptance_at_u", None)
        if not callable(mean_acceptance_at_u_fn):
            raise ValueError("acceptance lagrangian requires mean_acceptance_at_u(x_batch, u).")
        return float(value_at_u_fn(x_batch, u)) + self._lagrangian_value(
            float(mean_acceptance_at_u_fn(x_batch, u))
        )

    def base_value_at_u(self, x_batch: Any, u: float) -> float:
        base_value_at_u_fn = getattr(self.base_objective, "base_value_at_u", None)
        if callable(base_value_at_u_fn):
            return float(base_value_at_u_fn(x_batch, u))
        value_at_u_fn = getattr(self.base_objective, "value_at_u", None)
        if callable(value_at_u_fn):
            return float(value_at_u_fn(x_batch, u))
        raise ValueError("base_objective does not support value_at_u(x_batch, u).")

    def _value_batch(self, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
        del x_batch, u_arr
        raise ValueError("acceptance-floor scalarization cannot be evaluated as per-row action values.")

    def _value_batch_many(self, x_batch: Any, u_matrix: np.ndarray) -> np.ndarray:
        del x_batch, u_matrix
        raise ValueError("acceptance-floor scalarization cannot be evaluated as per-row action values.")

    def _mean_acceptance(self, theta: np.ndarray, x_batch: Any) -> float:
        mean_acceptance_fn = getattr(self.base_objective, "mean_acceptance", None)
        if not callable(mean_acceptance_fn):
            raise ValueError("acceptance lagrangian requires mean_acceptance(theta, x_batch).")
        return float(mean_acceptance_fn(theta, x_batch))

    def _mean_acceptance_grad(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        mean_acceptance_grad_fn = getattr(self.base_objective, "mean_acceptance_grad", None)
        if not callable(mean_acceptance_grad_fn):
            raise ValueError("acceptance lagrangian requires mean_acceptance_grad(theta, x_batch).")
        return np.asarray(mean_acceptance_grad_fn(theta, x_batch), dtype=float)

    def _lagrangian_value(self, mean_acceptance: float) -> float:
        return float(self.lagrangian_lambda) * (float(self.acceptance_floor) - float(mean_acceptance))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "AcceptanceLagrangianObjective",
            "acceptance_floor": float(self.acceptance_floor),
            "lagrangian_lambda": float(self.lagrangian_lambda),
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_objective, name)


def _attach_policy(wrapper: object, base_objective: Objective) -> None:
    base_policy = getattr(base_objective, "policy", None)
    policy = getattr(wrapper, "policy", None)
    if policy is None:
        if base_policy is not None:
            object.__setattr__(wrapper, "policy", base_policy)
        return
    if base_policy is policy:
        return
    try:
        updated_base = replace(base_objective, policy=policy)
    except TypeError as exc:
        raise ValueError("base_objective policy could not be replaced.") from exc
    object.__setattr__(wrapper, "base_objective", updated_base)


def _acceptance_floor(value: float) -> float:
    floor = float(value)
    if not 0.0 < floor < 1.0:
        raise ValueError("acceptance_floor must be in (0, 1).")
    return floor


def _positive(value: float, name: str) -> float:
    value_float = float(value)
    if not np.isfinite(value_float) or value_float <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value_float


def _nonnegative(value: float, name: str) -> float:
    value_float = float(value)
    if not np.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value_float


__all__ = ["AcceptanceLagrangianObjective", "AcceptancePenaltyObjective"]
