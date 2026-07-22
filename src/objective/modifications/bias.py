"""Deterministic action-bias objective wrapper."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from objective._math import _sigmoid
from objective.base import Objective, Policy
from objective.utils import _policy_value, _theta_grad_from_u_grad


class ActionBias:
    """Action-level additive bias term for objective wrappers."""

    lambda_bias: float

    def values(self, x_batch: Any, u: np.ndarray) -> np.ndarray:
        """Return additive bias values with the same shape as ``u``."""
        raise NotImplementedError

    def grad_u(self, x_batch: Any, u: np.ndarray) -> np.ndarray:
        """Return action-gradient of the additive bias term."""
        raise NotImplementedError


@dataclass(frozen=True)
class ThetaBiasBounds:
    """Global derivative bounds used by the strongly-convex bias proofs."""

    beta: float
    kappa_minus: float
    kappa_plus: float
    rho: float


class ThetaBias:
    """Deterministic scalar bias over a one-dimensional theta vector."""

    alpha: float

    def value(self, theta: np.ndarray) -> float:
        raise NotImplementedError

    def grad(self, theta: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative_bounds(self) -> ThetaBiasBounds:
        raise NotImplementedError


@dataclass(frozen=True)
class LinearThetaBias(ThetaBias):
    r"""Linear theta bias $$b(x)=\alpha x$$."""

    alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _finite_alpha(self.alpha))

    def value(self, theta: np.ndarray) -> float:
        return self.alpha * _theta_scalar(theta)

    def grad(self, theta: np.ndarray) -> np.ndarray:
        _theta_scalar(theta)
        return np.asarray([self.alpha], dtype=float)

    def derivative_bounds(self) -> ThetaBiasBounds:
        return ThetaBiasBounds(abs(self.alpha), 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ArctanThetaBias(ThetaBias):
    r"""Saturating theta bias $$b(x)=\alpha\arctan x$$."""

    alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _finite_alpha(self.alpha))

    def value(self, theta: np.ndarray) -> float:
        return float(self.alpha * np.arctan(_theta_scalar(theta)))

    def grad(self, theta: np.ndarray) -> np.ndarray:
        x = _theta_scalar(theta)
        return np.asarray([self.alpha / (1.0 + x * x)], dtype=float)

    def derivative_bounds(self) -> ThetaBiasBounds:
        curvature = 3.0 * np.sqrt(3.0) * abs(self.alpha) / 8.0
        return ThetaBiasBounds(abs(self.alpha), curvature, curvature, 2.0 * abs(self.alpha))


@dataclass(frozen=True)
class ArctanRemainderThetaBias(ThetaBias):
    r"""Cubic-near-zero bias $$b(x)=\alpha(x-\arctan x)$$."""

    alpha: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _finite_alpha(self.alpha))

    def value(self, theta: np.ndarray) -> float:
        x = _theta_scalar(theta)
        return float(self.alpha * (x - np.arctan(x)))

    def grad(self, theta: np.ndarray) -> np.ndarray:
        x = _theta_scalar(theta)
        return np.asarray([self.alpha * x * x / (1.0 + x * x)], dtype=float)

    def derivative_bounds(self) -> ThetaBiasBounds:
        curvature = 3.0 * np.sqrt(3.0) * abs(self.alpha) / 8.0
        return ThetaBiasBounds(abs(self.alpha), curvature, curvature, 2.0 * abs(self.alpha))


@dataclass(frozen=True)
class ThetaBiasedObjective(Objective):
    r"""Policy-free wrapper $$\widetilde J(\theta)=J(\theta)+b(\theta)$$."""

    base_objective: Objective
    bias: ThetaBias

    def value(self, theta: np.ndarray, x_batch: Any) -> float:
        return float(self.base_objective.value(theta, x_batch)) + self.bias.value(theta)

    def base_value(self, theta: np.ndarray, x_batch: Any) -> float:
        base_value_fn = getattr(self.base_objective, "base_value", None)
        if callable(base_value_fn):
            return float(base_value_fn(theta, x_batch))
        return float(self.base_objective.value(theta, x_batch))

    def grad(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        return np.asarray(self.base_objective.grad(theta, x_batch), dtype=float) + self.bias.grad(theta)

    def with_noise_seed(self, seed: int) -> "ThetaBiasedObjective":
        with_noise_seed = getattr(self.base_objective, "with_noise_seed", None)
        if not callable(with_noise_seed):
            return self
        return replace(self, base_objective=with_noise_seed(int(seed)))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_objective, name)


@dataclass(frozen=True)
class LinearActionBias(ActionBias):
    r"""Linear optimism term $$b(u)=-\lambda_{bias}u$$."""

    lambda_bias: float

    def __post_init__(self) -> None:
        lambda_bias = float(self.lambda_bias)
        if not np.isfinite(lambda_bias):
            raise ValueError("lambda_bias must be finite.")
        object.__setattr__(self, "lambda_bias", lambda_bias)

    def values(self, x_batch: Any, u: np.ndarray) -> np.ndarray:
        del x_batch
        u_arr = _validate_u_field(u)
        return -self.lambda_bias * u_arr

    def grad_u(self, x_batch: Any, u: np.ndarray) -> np.ndarray:
        del x_batch
        u_arr = _validate_u_field(u)
        return np.full_like(u_arr, -self.lambda_bias, dtype=float)


@dataclass(frozen=True)
class UpperSupportHingeBias(ActionBias):
    r"""Optimism only above action support: $$b(u)=-\lambda_{bias}(u-h)_+$$."""

    lambda_bias: float
    support_center: float
    support_radius: float
    smooth_tau: float | None = None

    def __post_init__(self) -> None:
        lambda_bias = float(self.lambda_bias)
        support_center = float(self.support_center)
        support_radius = float(self.support_radius)
        if not np.isfinite(lambda_bias):
            raise ValueError("lambda_bias must be finite.")
        if not np.isfinite(support_center):
            raise ValueError("support_center must be finite.")
        if not np.isfinite(support_radius) or support_radius < 0.0:
            raise ValueError("support_radius must be finite and nonnegative.")
        smooth_tau = None if self.smooth_tau is None else float(self.smooth_tau)
        if smooth_tau is not None and (not np.isfinite(smooth_tau) or smooth_tau <= 0.0):
            raise ValueError("smooth_tau must be positive when provided.")
        object.__setattr__(self, "lambda_bias", lambda_bias)
        object.__setattr__(self, "support_center", support_center)
        object.__setattr__(self, "support_radius", support_radius)
        object.__setattr__(self, "smooth_tau", smooth_tau)

    @property
    def support_upper(self) -> float:
        """Return upper support boundary $$h = u_c + r$$."""
        return float(self.support_center + self.support_radius)

    def excess(self, u: np.ndarray) -> np.ndarray:
        """Return hard upper-support excess $$(u-h)_+$$."""
        u_arr = _validate_u_field(u)
        return np.maximum(0.0, u_arr - self.support_upper)

    def values(self, x_batch: Any, u: np.ndarray) -> np.ndarray:
        del x_batch
        u_arr = _validate_u_field(u)
        if self.smooth_tau is None:
            return -self.lambda_bias * self.excess(u_arr)
        z = (u_arr - self.support_upper) / self.smooth_tau
        smooth_excess = self.smooth_tau * np.logaddexp(0.0, z)
        return -self.lambda_bias * smooth_excess

    def grad_u(self, x_batch: Any, u: np.ndarray) -> np.ndarray:
        del x_batch
        u_arr = _validate_u_field(u)
        if self.smooth_tau is None:
            return -self.lambda_bias * (u_arr > self.support_upper).astype(float)
        z = (u_arr - self.support_upper) / self.smooth_tau
        return -self.lambda_bias * _sigmoid(z)


@dataclass(frozen=True)
class BiasedObjective(Objective):
    r"""Objective wrapper exposing $$\hat{M}(x,u)=M(x,u)+b(x,u)$$."""

    base_objective: Objective
    lambda_bias: float | None = None
    bias: ActionBias | None = None
    policy: Policy | None = None

    def __post_init__(self) -> None:
        bias = self.bias
        if bias is None:
            if self.lambda_bias is None:
                raise ValueError("BiasedObjective requires lambda_bias or bias.")
            bias = LinearActionBias(float(self.lambda_bias))
            object.__setattr__(self, "bias", bias)
        lambda_bias = getattr(bias, "lambda_bias", self.lambda_bias)
        if lambda_bias is None:
            raise ValueError("bias must expose lambda_bias.")
        lambda_bias = float(lambda_bias)
        if self.lambda_bias is not None and not np.isclose(float(self.lambda_bias), lambda_bias):
            raise ValueError("lambda_bias must match bias.lambda_bias when both are provided.")
        object.__setattr__(self, "lambda_bias", lambda_bias)

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
        """Return mean biased objective value for ``theta`` on ``x_batch``."""
        theta_arr = np.asarray(theta, dtype=float)
        base_value = float(self.base_objective.value(theta_arr, x_batch))
        u_arr = self._clip_u(_policy_value(self.base_objective, theta_arr, x_batch))
        return base_value + float(np.mean(self.bias.values(x_batch, u_arr)))

    def base_value(self, theta: np.ndarray, x_batch: Any) -> float:
        """Return the wrapped true objective value used for reporting."""
        base_value_fn = getattr(self.base_objective, "base_value", None)
        if callable(base_value_fn):
            return float(base_value_fn(theta, x_batch))
        return float(self.base_objective.value(theta, x_batch))

    def grad(self, theta: np.ndarray, x_batch: Any) -> np.ndarray:
        """Return theta-gradient of the biased objective."""
        theta_arr = np.asarray(theta, dtype=float)
        base_grad = np.asarray(self.base_objective.grad(theta_arr, x_batch), dtype=float)
        u_arr = self._clip_u(_policy_value(self.base_objective, theta_arr, x_batch))
        bias_grad_u = self.bias.grad_u(x_batch, u_arr).reshape(-1)
        return base_grad + _theta_grad_from_u_grad(self, theta_arr, x_batch, bias_grad_u)

    def value_at_u(self, x_batch: Any, u: float) -> float:
        """Return mean biased objective value at a fixed action ``u``."""
        value_at_u_fn = getattr(self.base_objective, "value_at_u", None)
        if not callable(value_at_u_fn):
            raise ValueError("base_objective does not support value_at_u(x_batch, u).")
        u_val = float(self._clip_u(np.asarray([float(u)], dtype=float))[0])
        u_arr = np.full(_row_count(x_batch), u_val, dtype=float)
        return float(value_at_u_fn(x_batch, u_val)) + float(np.mean(self.bias.values(x_batch, u_arr)))

    def base_value_at_u(self, x_batch: Any, u: float) -> float:
        """Return the wrapped true objective value at a fixed action ``u``."""
        base_value_at_u_fn = getattr(self.base_objective, "base_value_at_u", None)
        if callable(base_value_at_u_fn):
            return float(base_value_at_u_fn(x_batch, u))
        value_at_u_fn = getattr(self.base_objective, "value_at_u", None)
        if callable(value_at_u_fn):
            return float(value_at_u_fn(x_batch, u))
        raise ValueError("base_objective does not support value_at_u(x_batch, u).")

    def _value_batch(self, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
        """Return per-row biased action-level objective values."""
        u_values = self._clip_u(_validate_u_vector(u_arr, _row_count(x_batch)))
        base_values = _base_action_values(self.base_objective, x_batch, u_values)
        return base_values + self.bias.values(x_batch, u_values)

    def _value_batch_many(self, x_batch: Any, u_matrix: np.ndarray) -> np.ndarray:
        """Return biased action-level values for many action vectors."""
        u_values = np.asarray(u_matrix, dtype=float)
        if u_values.ndim != 2:
            raise ValueError("u_matrix must be 2D.")
        n_rows = _row_count(x_batch)
        if u_values.shape[1] != n_rows:
            raise ValueError("u_matrix must have shape (n_evaluations, n_rows).")
        u_values = self._clip_u(u_values)
        value_many_fn = getattr(self.base_objective, "_value_batch_many", None)
        if callable(value_many_fn):
            base_values = np.asarray(value_many_fn(x_batch, u_values), dtype=float)
        else:
            base_values = np.vstack(
                [_base_action_values(self.base_objective, x_batch, u_row) for u_row in u_values]
            )
        if base_values.shape != u_values.shape:
            raise ValueError("base objective returned unexpected value matrix shape.")
        return base_values + self.bias.values(x_batch, u_values)

    def _grad_u_batch(self, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
        """Return per-row action-gradient of the biased objective."""
        u_values = self._clip_u(_validate_u_vector(u_arr, _row_count(x_batch)))
        grad_u_fn = getattr(self.base_objective, "_grad_u_batch", None)
        if not callable(grad_u_fn):
            raise ValueError("base_objective does not support _grad_u_batch(x_batch, u_arr).")
        grad_u = np.asarray(grad_u_fn(x_batch, u_values), dtype=float)
        if grad_u.shape != u_values.shape:
            raise ValueError("base objective _grad_u_batch returned unexpected shape.")
        return grad_u + self.bias.grad_u(x_batch, u_values)

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

    def with_noise_seed(self, seed: int) -> "BiasedObjective":
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

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_objective, name)


def _base_action_values(objective: Objective, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
    value_batch_fn = getattr(objective, "_value_batch", None)
    if callable(value_batch_fn):
        values = np.asarray(value_batch_fn(x_batch, u_arr), dtype=float)
        if values.shape != u_arr.shape:
            raise ValueError("base objective _value_batch returned unexpected shape.")
        return values
    value_at_u_fn = getattr(objective, "value_at_u", None)
    if not callable(value_at_u_fn):
        raise ValueError("base_objective must expose _value_batch or value_at_u.")
    values = np.empty_like(u_arr, dtype=float)
    for idx, u_val in enumerate(u_arr):
        values[idx] = float(value_at_u_fn(_slice_rows(x_batch, idx, idx + 1), float(u_val)))
    return values


def _row_count(x_batch: Any) -> int:
    return int(x_batch.shape[0])


def _slice_rows(x_batch: Any, start: int, stop: int) -> Any:
    if hasattr(x_batch, "iloc"):
        return x_batch.iloc[start:stop].reset_index(drop=True)
    return np.asarray(x_batch)[start:stop]


def _validate_u_vector(u: np.ndarray, n_rows: int) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float).reshape(-1)
    if u_arr.shape != (n_rows,):
        raise ValueError("u_arr must have one value per x_batch row.")
    if not np.isfinite(u_arr).all():
        raise ValueError("u_arr must contain only finite values.")
    return u_arr


def _validate_u_field(u: np.ndarray) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    if u_arr.ndim not in {1, 2}:
        raise ValueError("u must be a 1D action vector or 2D action matrix.")
    if not np.isfinite(u_arr).all():
        raise ValueError("u must contain only finite values.")
    return u_arr


def _finite_alpha(alpha: float) -> float:
    value = float(alpha)
    if not np.isfinite(value):
        raise ValueError("alpha must be finite.")
    return value


def _theta_scalar(theta: np.ndarray) -> float:
    theta_arr = np.asarray(theta, dtype=float)
    if theta_arr.shape != (1,) or not np.isfinite(theta_arr).all():
        raise ValueError("theta bias requires a finite theta vector with shape (1,).")
    return float(theta_arr[0])


__all__ = [
    "ActionBias",
    "ArctanRemainderThetaBias",
    "ArctanThetaBias",
    "BiasedObjective",
    "LinearActionBias",
    "LinearThetaBias",
    "ThetaBias",
    "ThetaBiasBounds",
    "ThetaBiasedObjective",
    "UpperSupportHingeBias",
]
