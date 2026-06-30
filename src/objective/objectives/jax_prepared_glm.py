"""JAX-backed prepared GLM objective for SciPy optimizer callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from objective._math import _sigmoid
from objective.base import Objective, Policy
from objective.objectives.prepared_glm import (
    PreparedGLMBatch,
    ProbabilityTarget,
    _N_METADATA_COLS,
    _prepared_x_array,
    prepare_glm_batch,
)
from objective.policy import ConstantPolicy, LinearPolicy, SoftmaxPolicy

jax.config.update("jax_enable_x64", True)

_LARGE_DESIGN_WARN_BYTES = 1_000_000_000


@dataclass(frozen=True)
class JaxPreparedGLMScipyAdapter:
    """CPU-facing SciPy callback adapter backed by fixed JAX device arrays."""

    objective: "JaxPreparedGLMObjective"

    def objective_value(self, theta: np.ndarray) -> float:
        """Return scalar objective value for SciPy ``minimize``."""
        value, _ = self.objective.objective_value_and_grad(theta)
        return value

    def objective_grad(self, theta: np.ndarray) -> np.ndarray:
        """Return objective gradient for SciPy ``minimize``."""
        _, grad = self.objective.objective_value_and_grad(theta)
        return grad

    def constraint(self, theta: np.ndarray) -> np.ndarray:
        """Return ``mean_acceptance(theta) - floor`` as shape ``(1,)``."""
        return np.asarray([self.objective.constraint_margin(theta)], dtype=float)

    def constraint_jac(self, theta: np.ndarray) -> np.ndarray:
        """Return constraint Jacobian as shape ``(1, theta_dim)``."""
        return np.atleast_2d(self.objective.constraint_margin_grad(theta))


@dataclass(frozen=True)
class JaxPreparedGLMObjective(Objective):
    """Fixed-batch JAX GLM objective with a NumPy/SciPy-compatible API."""

    policy: Policy
    x_array: np.ndarray | PreparedGLMBatch
    u_coef: float
    probability_target: ProbabilityTarget = "acceptance"
    u_bounds: tuple[float, float] | None = None
    acceptance_floor: float | None = None
    acceptance_penalty_weight: float | None = None
    acceptance_penalty_temperature: float = 0.01
    lagrangian_lambda: float | None = None
    _objective_cache_key: tuple[tuple[int, ...], bytes] | None = field(init=False, default=None, repr=False)
    _objective_cache: tuple[float, np.ndarray] | None = field(init=False, default=None, repr=False)
    _value_cache_key: tuple[tuple[int, ...], bytes] | None = field(init=False, default=None, repr=False)
    _value_cache: float | None = field(init=False, default=None, repr=False)
    _acceptance_cache_key: tuple[tuple[int, ...], bytes] | None = field(init=False, default=None, repr=False)
    _acceptance_cache: tuple[float, np.ndarray] | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        x_arr = _prepared_x_array(self.x_array)
        target = str(self.probability_target)
        if target not in {"acceptance", "churn"}:
            raise ValueError("probability_target must be 'acceptance' or 'churn'.")
        if self.u_bounds is not None:
            lower, upper = (float(self.u_bounds[0]), float(self.u_bounds[1]))
            if lower >= upper:
                raise ValueError("u_bounds must be an increasing (lower, upper) tuple.")
            object.__setattr__(self, "u_bounds", (lower, upper))
        if self.acceptance_penalty_temperature <= 0.0:
            raise ValueError("acceptance_penalty_temperature must be positive.")

        design = _policy_design_matrix(self.policy, x_arr[:, _N_METADATA_COLS:])
        policy_kind, action_low, action_span = _policy_backend_spec(self.policy)
        if policy_kind != "constant" and design is None:
            raise ValueError("JAX GLM backend requires a policy design matrix.")

        _warn_if_large_prepared_design(design)
        base_logit_np = np.asarray(x_arr[:, 0], dtype=float)
        loss_np = np.asarray(x_arr[:, 1], dtype=float)
        premium_np = np.asarray(x_arr[:, 2], dtype=float)
        design_np = None if design is None else np.asarray(design, dtype=float)
        design_for_jax = (
            np.empty((x_arr.shape[0], 0), dtype=float)
            if design_np is None
            else design_np
        )
        base_logit_jax = jnp.asarray(base_logit_np, dtype=jnp.float64)
        loss_jax = jnp.asarray(loss_np, dtype=jnp.float64)
        premium_jax = jnp.asarray(premium_np, dtype=jnp.float64)
        design_jax = jnp.asarray(design_for_jax, dtype=jnp.float64)
        u_coef = float(self.u_coef)
        sign = 1.0 if target == "acceptance" else -1.0
        floor = None if self.acceptance_floor is None else float(self.acceptance_floor)
        penalty_weight = (
            None if self.acceptance_penalty_weight is None else float(self.acceptance_penalty_weight)
        )
        penalty_temperature = float(self.acceptance_penalty_temperature)
        lagrangian_lambda = None if self.lagrangian_lambda is None else float(self.lagrangian_lambda)
        u_bounds = self.u_bounds

        def clip_u(u: Any) -> Any:
            if u_bounds is None:
                return u
            return jnp.clip(u, float(u_bounds[0]), float(u_bounds[1]))

        def policy_raw_u(theta: Any, base_logit_arr: Any, design_arr: Any) -> Any:
            if policy_kind == "constant":
                return jnp.full(base_logit_arr.shape, theta[0], dtype=jnp.float64)
            score = design_arr @ theta
            if policy_kind == "linear":
                return score
            return action_low + action_span * jax.nn.sigmoid(score)

        def policy_u(theta: Any, base_logit_arr: Any, design_arr: Any) -> Any:
            return clip_u(policy_raw_u(theta, base_logit_arr, design_arr))

        def acceptance_from_u(base_logit_arr: Any, u: Any) -> Any:
            class1 = jax.nn.sigmoid(base_logit_arr + u_coef * u)
            return class1 if sign > 0.0 else 1.0 - class1

        def action_values_from_u(
            base_logit_arr: Any,
            loss_arr: Any,
            premium_arr: Any,
            u: Any,
        ) -> Any:
            u_clipped = clip_u(u)
            acceptance = acceptance_from_u(base_logit_arr, u_clipped)
            revenue = (u_clipped + 1.0) * premium_arr
            return acceptance * (loss_arr - revenue)

        def action_values_from_u_many(
            base_logit_arr: Any,
            loss_arr: Any,
            premium_arr: Any,
            u_matrix: Any,
        ) -> Any:
            u_clipped = clip_u(u_matrix)
            class1 = jax.nn.sigmoid(base_logit_arr[None, :] + u_coef * u_clipped)
            acceptance = class1 if sign > 0.0 else 1.0 - class1
            revenue = (u_clipped + 1.0) * premium_arr[None, :]
            return acceptance * (loss_arr[None, :] - revenue)

        def raw_objective(
            theta: Any,
            base_logit_arr: Any,
            loss_arr: Any,
            premium_arr: Any,
            design_arr: Any,
        ) -> Any:
            return jnp.mean(
                action_values_from_u(
                    base_logit_arr,
                    loss_arr,
                    premium_arr,
                    policy_u(theta, base_logit_arr, design_arr),
                )
            )

        def mean_acceptance(theta: Any, base_logit_arr: Any, design_arr: Any) -> Any:
            return jnp.mean(
                acceptance_from_u(
                    base_logit_arr,
                    policy_u(theta, base_logit_arr, design_arr),
                )
            )

        def objective(
            theta: Any,
            base_logit_arr: Any,
            loss_arr: Any,
            premium_arr: Any,
            design_arr: Any,
        ) -> Any:
            value = raw_objective(theta, base_logit_arr, loss_arr, premium_arr, design_arr)
            acceptance_mean = mean_acceptance(theta, base_logit_arr, design_arr)
            if floor is not None and penalty_weight is not None:
                gap = floor - acceptance_mean
                soft_gap = penalty_temperature * jax.nn.softplus(gap / penalty_temperature)
                value = value + penalty_weight * soft_gap * soft_gap
            if floor is not None and lagrangian_lambda is not None:
                value = value + lagrangian_lambda * (floor - acceptance_mean)
            return value

        object.__setattr__(self, "x_array", x_arr)
        object.__setattr__(self, "u_coef", u_coef)
        object.__setattr__(self, "probability_target", target)
        object.__setattr__(self, "_base_logit_np", base_logit_np)
        object.__setattr__(self, "_loss_np", loss_np)
        object.__setattr__(self, "_premium_np", premium_np)
        object.__setattr__(
            self,
            "_policy_features_np",
            np.asarray(x_arr[:, _N_METADATA_COLS:], dtype=float),
        )
        object.__setattr__(self, "_design_matrix_np", design_np)
        object.__setattr__(self, "_base_logit_jax", base_logit_jax)
        object.__setattr__(self, "_loss_jax", loss_jax)
        object.__setattr__(self, "_premium_jax", premium_jax)
        object.__setattr__(self, "_design_jax", design_jax)
        object.__setattr__(self, "_policy_kind", policy_kind)
        object.__setattr__(self, "_action_low", action_low)
        object.__setattr__(self, "_action_span", action_span)
        object.__setattr__(self, "_policy_u_jit", jax.jit(policy_u))
        object.__setattr__(self, "_raw_objective_jit", jax.jit(raw_objective))
        object.__setattr__(self, "_objective_value_jit", jax.jit(objective))
        object.__setattr__(
            self,
            "_objective_value_and_grad_jit",
            jax.jit(jax.value_and_grad(objective)),
        )
        object.__setattr__(
            self,
            "_mean_acceptance_value_and_grad_jit",
            jax.jit(jax.value_and_grad(mean_acceptance)),
        )
        object.__setattr__(self, "_action_values_from_u_jit", jax.jit(action_values_from_u))
        object.__setattr__(self, "_action_values_from_u_many_jit", jax.jit(action_values_from_u_many))

    def warmup(self, theta: np.ndarray) -> None:
        """Compile and run all JAX callbacks once before timing benchmarks."""
        self.objective_value_and_grad(theta)
        self.mean_acceptance_value_and_grad(theta)
        u_batch = self.policy_value(theta, self.x_array)
        self._value_batch(self.x_array, u_batch)
        self._value_batch_many(self.x_array, u_batch[None, :])

    def scipy_adapter(self) -> JaxPreparedGLMScipyAdapter:
        """Return explicit SciPy callback wrappers for this fixed batch."""
        return JaxPreparedGLMScipyAdapter(self)

    def _objective_value(self, theta: np.ndarray) -> float:
        """Return cached objective value for value-only estimators."""
        key, theta_jax = self._theta_key_and_jax(theta)
        if key == self._objective_cache_key and self._objective_cache is not None:
            return float(self._objective_cache[0])
        if key == self._value_cache_key and self._value_cache is not None:
            return float(self._value_cache)
        result = float(
            self._objective_value_jit(
                theta_jax,
                self._base_logit_jax,
                self._loss_jax,
                self._premium_jax,
                self._design_jax,
            )
        )
        object.__setattr__(self, "_value_cache_key", key)
        object.__setattr__(self, "_value_cache", result)
        return result

    def objective_value_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Return cached objective value and gradient for ``theta``."""
        key, theta_jax = self._theta_key_and_jax(theta)
        if key == self._objective_cache_key and self._objective_cache is not None:
            return self._objective_cache
        value_jax, grad_jax = self._objective_value_and_grad_jit(
            theta_jax,
            self._base_logit_jax,
            self._loss_jax,
            self._premium_jax,
            self._design_jax,
        )
        result = (float(value_jax), np.asarray(grad_jax, dtype=float))
        object.__setattr__(self, "_objective_cache_key", key)
        object.__setattr__(self, "_objective_cache", result)
        return result

    def mean_acceptance_value_and_grad(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Return cached mean acceptance and gradient for ``theta``."""
        key, theta_jax = self._theta_key_and_jax(theta)
        if key == self._acceptance_cache_key and self._acceptance_cache is not None:
            return self._acceptance_cache
        value_jax, grad_jax = self._mean_acceptance_value_and_grad_jit(
            theta_jax,
            self._base_logit_jax,
            self._design_jax,
        )
        result = (float(value_jax), np.asarray(grad_jax, dtype=float))
        object.__setattr__(self, "_acceptance_cache_key", key)
        object.__setattr__(self, "_acceptance_cache", result)
        return result

    def value(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> float:
        """Return JAX objective value on this fixed prepared batch."""
        self._validate_fixed_batch(x_batch)
        return self._objective_value(theta)

    def base_value(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> float:
        """Return raw pricing objective without penalty or Lagrangian terms."""
        self._validate_fixed_batch(x_batch)
        theta_jax = self._theta_key_and_jax(theta)[1]
        return float(
            self._raw_objective_jit(
                theta_jax,
                self._base_logit_jax,
                self._loss_jax,
                self._premium_jax,
                self._design_jax,
            )
        )

    def grad(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> np.ndarray:
        """Return JAX objective gradient on this fixed prepared batch."""
        self._validate_fixed_batch(x_batch)
        _, grad = self.objective_value_and_grad(theta)
        return grad

    def policy_value(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> np.ndarray:
        """Return policy actions on this fixed prepared batch."""
        self._validate_fixed_batch(x_batch)
        theta_jax = self._theta_key_and_jax(theta)[1]
        return np.asarray(
            self._policy_u_jit(theta_jax, self._base_logit_jax, self._design_jax),
            dtype=float,
        )

    def policy_grad(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> np.ndarray:
        """Return policy Jacobian, primarily for API parity diagnostics."""
        self._validate_fixed_batch(x_batch)
        theta_arr = np.asarray(theta, dtype=float)
        if self._policy_kind == "constant":
            grad = np.ones((self.x_array.shape[0], 1), dtype=float)
            raw_u = np.full(self.x_array.shape[0], theta_arr[0], dtype=float)
            return self._apply_clip_derivative(grad, raw_u)
        design = np.asarray(self._design_matrix_np, dtype=float)
        if self._policy_kind == "linear":
            return self._apply_clip_derivative(design, design @ theta_arr)
        score = design @ theta_arr
        sigma = _sigmoid(score)
        grad = self._action_span * sigma[:, None] * (1.0 - sigma[:, None]) * design
        raw_u = self._action_low + self._action_span * sigma
        return self._apply_clip_derivative(grad, raw_u)

    def policy_weighted_grad(
        self,
        theta: np.ndarray,
        x_batch: np.ndarray | PreparedGLMBatch,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Return ``sum_i weights_i * d pi_theta(x_i) / dtheta``."""
        self._validate_fixed_batch(x_batch)
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape != (self.x_array.shape[0],):
            raise ValueError("weights must have one value per x_batch row.")
        theta_arr = np.asarray(theta, dtype=float)
        if self._policy_kind == "constant":
            raw_u = np.full(self.x_array.shape[0], theta_arr[0], dtype=float)
            return np.asarray([np.sum(self._clip_derivative_weights(weights_arr, raw_u))], dtype=float)
        design = np.asarray(self._design_matrix_np, dtype=float)
        if self._policy_kind == "linear":
            return self._clip_derivative_weights(weights_arr, design @ theta_arr) @ design
        score = design @ theta_arr
        sigma = _sigmoid(score)
        raw_u = self._action_low + self._action_span * sigma
        scaled_weights = weights_arr * self._action_span * sigma * (1.0 - sigma)
        return self._clip_derivative_weights(scaled_weights, raw_u) @ design

    def policy_input_dim(self) -> int:
        """Return the prepared policy input dimension before feature mapping."""
        return int(self.x_array.shape[1] - _N_METADATA_COLS)

    def policy_theta_dim(self, state_dim: int | None = None) -> int:
        """Return theta dimension for the fixed JAX policy representation."""
        del state_dim
        if self._policy_kind == "constant":
            return 1
        return int(self._design_matrix_np.shape[1])

    def value_at_u(self, x_batch: np.ndarray | PreparedGLMBatch, u: float) -> float:
        """Return objective value at a fixed action on this fixed prepared batch."""
        self._validate_fixed_batch(x_batch)
        base_value = self.base_value_at_u(x_batch, u)
        acceptance = self._acceptance_at_u(float(u))
        penalty_value, _ = self._acceptance_penalty(acceptance)
        lagrangian_value, _ = self._lagrangian_adjustment(acceptance)
        return base_value + penalty_value + lagrangian_value

    def base_value_at_u(self, x_batch: np.ndarray | PreparedGLMBatch, u: float) -> float:
        """Return raw objective value at a fixed action."""
        self._validate_fixed_batch(x_batch)
        u_arr = np.full(self.x_array.shape[0], self._clip_scalar_u(float(u)), dtype=float)
        return float(np.mean(self._value_batch(self.x_array, u_arr)))

    def mean_acceptance(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> float:
        """Return mean acceptance on this fixed prepared batch."""
        self._validate_fixed_batch(x_batch)
        value, _ = self.mean_acceptance_value_and_grad(theta)
        return value

    def mean_acceptance_at_u(self, x_batch: np.ndarray | PreparedGLMBatch, u: float) -> float:
        """Return mean acceptance at a fixed action."""
        self._validate_fixed_batch(x_batch)
        return float(np.mean(self._acceptance_at_u(float(u))))

    def mean_acceptance_grad(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> np.ndarray:
        """Return JAX gradient of mean acceptance."""
        self._validate_fixed_batch(x_batch)
        _, grad = self.mean_acceptance_value_and_grad(theta)
        return grad

    def constraint_margin(self, theta: np.ndarray) -> float:
        """Return ``mean_acceptance(theta) - acceptance_floor``."""
        if self.acceptance_floor is None:
            raise ValueError("constraint_margin requires acceptance_floor.")
        mean_acceptance, _ = self.mean_acceptance_value_and_grad(theta)
        return mean_acceptance - float(self.acceptance_floor)

    def constraint_margin_grad(self, theta: np.ndarray) -> np.ndarray:
        """Return gradient of ``mean_acceptance(theta) - acceptance_floor``."""
        if self.acceptance_floor is None:
            raise ValueError("constraint_margin_grad requires acceptance_floor.")
        _, grad = self.mean_acceptance_value_and_grad(theta)
        return grad

    def _step_metrics(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> dict[str, float]:
        """Return per-step metrics for reporter logging."""
        u_batch = self.policy_value(theta, x_batch)
        acceptance = self._acceptance_at_u_array(u_batch)
        revenue = (u_batch + 1.0) * self._premium_np
        return {
            "mean_acceptance": float(np.mean(acceptance)),
            "projected_loss": float(np.mean(self._loss_np)),
            "projected_revenue": float(np.mean(revenue)),
        }

    def _theta_key_and_jax(self, theta: np.ndarray) -> tuple[tuple[tuple[int, ...], bytes], Any]:
        theta_arr = np.ascontiguousarray(np.asarray(theta, dtype=float))
        if theta_arr.ndim != 1:
            raise ValueError("theta must be a 1D array.")
        if theta_arr.size != self.policy_theta_dim():
            raise ValueError(f"theta must have exactly {self.policy_theta_dim()} elements.")
        return (tuple(theta_arr.shape), theta_arr.tobytes()), jnp.asarray(theta_arr, dtype=jnp.float64)

    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        if self.u_bounds is None:
            return np.asarray(u, dtype=float)
        return np.clip(np.asarray(u, dtype=float), *self.u_bounds)

    def _clip_derivative_weights(self, weights: np.ndarray, raw_u: np.ndarray) -> np.ndarray:
        if self.u_bounds is None:
            return weights
        interior = (raw_u > self.u_bounds[0]) & (raw_u < self.u_bounds[1])
        return weights * interior

    def _apply_clip_derivative(self, grad: np.ndarray, raw_u: np.ndarray) -> np.ndarray:
        if self.u_bounds is None:
            return grad
        keep = self._clip_derivative_weights(np.ones_like(raw_u, dtype=float), raw_u)
        return grad * keep[:, None]

    def _value_batch(self, x_batch: np.ndarray | PreparedGLMBatch, u_arr: np.ndarray) -> np.ndarray:
        """Return raw per-row objective values for supplied actions."""
        self._validate_fixed_batch(x_batch)
        u_array = np.asarray(u_arr, dtype=float).reshape(-1)
        if u_array.shape != (self.x_array.shape[0],):
            raise ValueError("u_arr must have one value per x_batch row.")
        values = self._action_values_from_u_jit(
            self._base_logit_jax,
            self._loss_jax,
            self._premium_jax,
            jnp.asarray(u_array, dtype=jnp.float64),
        )
        return np.asarray(values, dtype=float)

    def _value_batch_many(self, x_batch: np.ndarray | PreparedGLMBatch, u_matrix: np.ndarray) -> np.ndarray:
        """Return raw per-row objective values for multiple supplied action vectors."""
        self._validate_fixed_batch(x_batch)
        u_arr = np.asarray(u_matrix, dtype=float)
        if u_arr.ndim != 2 or u_arr.shape[1] != self.x_array.shape[0]:
            raise ValueError("u_matrix must have shape (n_evaluations, n_rows).")
        values = self._action_values_from_u_many_jit(
            self._base_logit_jax,
            self._loss_jax,
            self._premium_jax,
            jnp.asarray(u_arr, dtype=jnp.float64),
        )
        return np.asarray(values, dtype=float)

    def _validate_fixed_batch(self, x_batch: np.ndarray | PreparedGLMBatch) -> None:
        x_arr = _prepared_x_array(x_batch)
        if x_arr.shape != self.x_array.shape:
            raise ValueError("JAX prepared GLM objective is fixed to its prepared batch shape.")

    def _clip_scalar_u(self, u: float) -> float:
        if self.u_bounds is None:
            return float(u)
        return float(np.clip(float(u), *self.u_bounds))

    def _acceptance_at_u(self, u: float) -> np.ndarray:
        u_arr = np.full(self.x_array.shape[0], self._clip_scalar_u(u), dtype=float)
        return self._acceptance_at_u_array(u_arr)

    def _acceptance_at_u_array(self, u_arr: np.ndarray) -> np.ndarray:
        u_clipped = self._clip_u(u_arr)
        class1 = _sigmoid(self._base_logit_np + self.u_coef * u_clipped)
        if self.probability_target == "acceptance":
            return class1
        return 1.0 - class1

    def _acceptance_penalty(self, acceptance: np.ndarray) -> tuple[float, float]:
        if self.acceptance_floor is None or self.acceptance_penalty_weight is None:
            return 0.0, 0.0
        mean_acceptance = float(np.mean(acceptance))
        gap = float(self.acceptance_floor) - mean_acceptance
        temp = float(self.acceptance_penalty_temperature)
        soft_gap = temp * float(np.logaddexp(0.0, gap / temp))
        sigmoid_gap = 1.0 / (1.0 + np.exp(-(gap / temp)))
        weight = float(self.acceptance_penalty_weight)
        return weight * soft_gap * soft_gap, -2.0 * weight * soft_gap * sigmoid_gap

    def _lagrangian_adjustment(self, acceptance: np.ndarray) -> tuple[float, float]:
        if self.acceptance_floor is None or self.lagrangian_lambda is None:
            return 0.0, 0.0
        mean_acceptance = float(np.mean(acceptance))
        lambda_value = float(self.lagrangian_lambda)
        return lambda_value * (float(self.acceptance_floor) - mean_acceptance), -lambda_value


def prepare_jax_glm_objective(
    objective: Any,
    x_batch: Any,
    *,
    row_indices: np.ndarray | None = None,
) -> tuple[JaxPreparedGLMObjective, PreparedGLMBatch]:
    """Materialize a GLM-backed objective into a fixed-batch JAX objective."""
    batch = prepare_glm_batch(objective, x_batch, row_indices=row_indices)
    prepared = JaxPreparedGLMObjective(
        policy=objective.policy,
        x_array=batch.x_array,
        u_coef=batch.u_coef,
        probability_target=batch.probability_target,
        u_bounds=getattr(objective, "u_bounds", None),
        acceptance_floor=getattr(objective, "acceptance_floor", None),
        acceptance_penalty_weight=getattr(objective, "acceptance_penalty_weight", None),
        acceptance_penalty_temperature=getattr(objective, "acceptance_penalty_temperature", 0.01),
        lagrangian_lambda=getattr(objective, "lagrangian_lambda", None),
    )
    return prepared, batch


def _policy_backend_spec(policy: Policy) -> tuple[str, float, float]:
    if isinstance(policy, ConstantPolicy):
        return "constant", 0.0, 0.0
    if isinstance(policy, LinearPolicy):
        return "linear", 0.0, 0.0
    if isinstance(policy, SoftmaxPolicy):
        return "softmax", float(policy.action_low), float(policy.action_span)
    raise ValueError("JAX prepared GLM backend currently supports constant, linear, and softmax policies.")


def _policy_design_matrix(policy: Policy, features: np.ndarray) -> np.ndarray | None:
    features_arr = np.asarray(features, dtype=float)
    if features_arr.ndim != 2:
        raise ValueError("prepared policy features must be a 2D array.")
    if not np.isfinite(features_arr).all():
        raise ValueError("prepared policy features must be finite.")
    if isinstance(policy, ConstantPolicy):
        return None
    if isinstance(policy, (LinearPolicy, SoftmaxPolicy)):
        feature_map = getattr(policy, "feature_map", None)
        transform = getattr(feature_map, "transform", None)
        if not callable(transform):
            raise ValueError(
                "JAX prepared GLM linear/softmax policies require a materializable feature_map."
            )
        mapped = np.asarray(transform(features_arr), dtype=float)
        if mapped.ndim != 2:
            raise ValueError("policy feature_map must return a 2D array.")
        if mapped.shape[0] != features_arr.shape[0]:
            raise ValueError("policy feature_map must preserve the prepared batch row count.")
        if not np.isfinite(mapped).all():
            raise ValueError("policy feature_map must return finite values.")
        ones = np.ones((features_arr.shape[0], 1), dtype=float)
        design = np.concatenate([ones, mapped], axis=1)
        theta_dim_fn = getattr(policy, "theta_dim", None)
        if callable(theta_dim_fn):
            expected_dim = int(theta_dim_fn(features_arr.shape[1]))
            if design.shape[1] != expected_dim:
                raise ValueError(
                    f"policy feature_map produced design width {design.shape[1]}; "
                    f"policy theta_dim expects {expected_dim}."
                )
        return design
    raise ValueError("JAX prepared GLM backend currently supports constant, linear, and softmax policies.")


def _prepared_design_memory_summary(design: np.ndarray | None) -> dict[str, object]:
    """Return shape and byte-count diagnostics for a materialized design matrix."""
    if design is None:
        return {
            "design_shape": None,
            "design_dtype": None,
            "design_nbytes": 0,
            "design_gb": 0.0,
            "design_gib": 0.0,
        }
    design_arr = np.asarray(design)
    nbytes = int(design_arr.nbytes)
    return {
        "design_shape": tuple(int(dim) for dim in design_arr.shape),
        "design_dtype": str(design_arr.dtype),
        "design_nbytes": nbytes,
        "design_gb": nbytes / 1e9,
        "design_gib": nbytes / float(1024**3),
    }


def _warn_if_large_prepared_design(design: np.ndarray | None) -> None:
    summary = _prepared_design_memory_summary(design)
    nbytes = int(summary["design_nbytes"])
    if nbytes < _LARGE_DESIGN_WARN_BYTES:
        return
    warnings.warn(
        "JAX prepared GLM materialized a large policy design matrix: "
        f"shape={summary['design_shape']}, "
        f"dtype={summary['design_dtype']}, "
        f"estimated_memory={summary['design_gb']:.2f} GB "
        f"({summary['design_gib']:.2f} GiB). "
        "The matrix is passed to JAX callbacks as runtime device data rather "
        "than a captured compile-time constant, but large full-batch higher-order "
        "feature maps can still exceed GPU memory.",
        RuntimeWarning,
        stacklevel=3,
    )


__all__ = [
    "JaxPreparedGLMObjective",
    "JaxPreparedGLMScipyAdapter",
    "prepare_jax_glm_objective",
]
