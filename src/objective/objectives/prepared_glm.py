"""Array-backed GLM pricing objective prepared from real-data artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from objective._math import _sigmoid
from objective.base import Objective, Policy
from objective.policy import policy_theta_dim
from objective.utils import _theta_grad_from_u_grad

ProbabilityTarget = Literal["acceptance", "churn"]
_N_METADATA_COLS = 3


@dataclass(frozen=True)
class PreparedGLMBatch:
    """Numeric GLM batch: ``[base_logit, loss, premium, policy_features...]``."""

    x_array: np.ndarray
    u_coef: float
    probability_target: ProbabilityTarget = "acceptance"
    row_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        x_arr = np.asarray(self.x_array, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_array must be a 2D array.")
        if x_arr.shape[1] <= _N_METADATA_COLS:
            raise ValueError("x_array must include at least one policy feature column.")
        if not np.isfinite(x_arr).all():
            raise ValueError("x_array must contain finite values.")
        target = str(self.probability_target)
        if target not in {"acceptance", "churn"}:
            raise ValueError("probability_target must be 'acceptance' or 'churn'.")
        row_indices = None
        if self.row_indices is not None:
            row_indices = np.asarray(self.row_indices, dtype=int)
            if row_indices.shape != (x_arr.shape[0],):
                raise ValueError("row_indices must have one entry per x_array row.")
        object.__setattr__(self, "x_array", x_arr)
        object.__setattr__(self, "u_coef", float(self.u_coef))
        object.__setattr__(self, "probability_target", target)
        object.__setattr__(self, "row_indices", row_indices)

    @classmethod
    def from_arrays(
        cls,
        *,
        base_logit: np.ndarray,
        loss: np.ndarray,
        premium: np.ndarray,
        policy_features: np.ndarray,
        u_coef: float,
        probability_target: ProbabilityTarget = "acceptance",
        row_indices: np.ndarray | None = None,
    ) -> "PreparedGLMBatch":
        """Build a prepared batch from separately materialized numeric arrays."""
        base_logit_arr = _as_vector(base_logit, "base_logit")
        loss_arr = _as_vector(loss, "loss")
        premium_arr = _as_vector(premium, "premium")
        features_arr = np.asarray(policy_features, dtype=float)
        if features_arr.ndim != 2:
            raise ValueError("policy_features must be a 2D array.")
        n_rows = base_logit_arr.shape[0]
        if loss_arr.shape != (n_rows,) or premium_arr.shape != (n_rows,) or features_arr.shape[0] != n_rows:
            raise ValueError("prepared arrays must share the same row count.")
        x_array = np.column_stack([base_logit_arr, loss_arr, premium_arr, features_arr])
        return cls(
            x_array=x_array,
            u_coef=u_coef,
            probability_target=probability_target,
            row_indices=row_indices,
        )

    @property
    def n_rows(self) -> int:
        """Return the number of prepared rows."""
        return int(self.x_array.shape[0])

    @property
    def policy_feature_dim(self) -> int:
        """Return the number of numeric columns seen by the policy."""
        return int(self.x_array.shape[1] - _N_METADATA_COLS)


@dataclass(frozen=True)
class PreparedGLMObjective(Objective):
    """Pure NumPy GLM objective over a ``PreparedGLMBatch.x_array`` matrix."""

    policy: Policy
    policy_feature_dim: int
    u_coef: float
    probability_target: ProbabilityTarget = "acceptance"
    u_bounds: tuple[float, float] | None = None
    acceptance_floor: float | None = None
    acceptance_penalty_weight: float | None = None
    acceptance_penalty_temperature: float = 0.01
    lagrangian_lambda: float | None = None

    def __post_init__(self) -> None:
        feature_dim = int(self.policy_feature_dim)
        if feature_dim <= 0:
            raise ValueError("policy_feature_dim must be positive.")
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
        object.__setattr__(self, "policy_feature_dim", feature_dim)
        object.__setattr__(self, "u_coef", float(self.u_coef))
        object.__setattr__(self, "probability_target", target)

    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        if self.u_bounds is None:
            return u
        return np.clip(u, *self.u_bounds)

    def _arrays(self, x_batch: np.ndarray | PreparedGLMBatch) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_arr = _prepared_x_array(x_batch)
        expected_cols = _N_METADATA_COLS + self.policy_feature_dim
        if x_arr.shape[1] != expected_cols:
            raise ValueError(f"x_batch must have {expected_cols} columns.")
        return x_arr[:, 0], x_arr[:, 1], x_arr[:, 2], x_arr[:, _N_METADATA_COLS:]

    def _acceptance(self, base_logit: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        class1 = _sigmoid(base_logit + self.u_coef * u_arr)
        if self.probability_target == "acceptance":
            return class1
        return 1.0 - class1

    def _d_acceptance_du(self, base_logit: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        class1 = _sigmoid(base_logit + self.u_coef * u_arr)
        sign = 1.0 if self.probability_target == "acceptance" else -1.0
        return sign * class1 * (1.0 - class1) * self.u_coef

    def _value_batch_from_arrays(
        self,
        acceptance: np.ndarray,
        loss: np.ndarray,
        premium: np.ndarray,
        u_arr: np.ndarray,
    ) -> np.ndarray:
        revenue = (u_arr + 1.0) * premium
        return acceptance * (loss - revenue)

    def _grad_u_batch_from_arrays(
        self,
        base_logit: np.ndarray,
        loss: np.ndarray,
        premium: np.ndarray,
        u_arr: np.ndarray,
    ) -> np.ndarray:
        acceptance = self._acceptance(base_logit, u_arr)
        revenue = (u_arr + 1.0) * premium
        d_acceptance_du = self._d_acceptance_du(base_logit, u_arr)
        return d_acceptance_du * (loss - revenue) - acceptance * premium

    def _acceptance_penalty(self, acceptance: np.ndarray) -> tuple[float, float]:
        if self.acceptance_floor is None or self.acceptance_penalty_weight is None:
            return 0.0, 0.0
        mean_acceptance = float(np.mean(acceptance))
        gap = float(self.acceptance_floor) - mean_acceptance
        temp = float(self.acceptance_penalty_temperature)
        scaled_gap = gap / temp
        soft_gap = temp * float(np.logaddexp(0.0, scaled_gap))
        sigmoid_gap = 1.0 / (1.0 + np.exp(-scaled_gap))
        weight = float(self.acceptance_penalty_weight)
        return weight * soft_gap * soft_gap, -2.0 * weight * soft_gap * sigmoid_gap

    def _lagrangian_adjustment(self, acceptance: np.ndarray) -> tuple[float, float]:
        if self.acceptance_floor is None or self.lagrangian_lambda is None:
            return 0.0, 0.0
        mean_acceptance = float(np.mean(acceptance))
        lambda_value = float(self.lagrangian_lambda)
        return lambda_value * (float(self.acceptance_floor) - mean_acceptance), -lambda_value

    def value(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> float:
        """Compute mean objective value on a prepared numeric batch."""
        base_logit, loss, premium, features = self._arrays(x_batch)
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self._clip_u(self.policy.value(theta_arr, features))
        acceptance = self._acceptance(base_logit, u_batch)
        base_value = float(np.mean(self._value_batch_from_arrays(acceptance, loss, premium, u_batch)))
        penalty_value, _ = self._acceptance_penalty(acceptance)
        lagrangian_value, _ = self._lagrangian_adjustment(acceptance)
        return base_value + penalty_value + lagrangian_value

    def base_value(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> float:
        """Return the raw pricing objective without penalties or Lagrangian terms."""
        base_logit, loss, premium, features = self._arrays(x_batch)
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self._clip_u(self.policy.value(theta_arr, features))
        acceptance = self._acceptance(base_logit, u_batch)
        return float(np.mean(self._value_batch_from_arrays(acceptance, loss, premium, u_batch)))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> np.ndarray:
        """Compute theta-gradient via a policy VJP over prepared features."""
        base_logit, loss, premium, features = self._arrays(x_batch)
        theta_arr = np.asarray(theta, dtype=float)
        u_raw = self.policy.value(theta_arr, features)
        u_clipped = self._clip_u(u_raw)
        grad_u = self._grad_u_batch_from_arrays(base_logit, loss, premium, u_clipped)
        if self.u_bounds is not None:
            interior = (u_raw > self.u_bounds[0]) & (u_raw < self.u_bounds[1])
            grad_u = grad_u * interior
        grad_theta = _theta_grad_from_u_grad(self.policy, theta_arr, features, grad_u)
        acceptance = self._acceptance(base_logit, u_clipped)
        _, penalty_scale = self._acceptance_penalty(acceptance)
        _, lagrangian_scale = self._lagrangian_adjustment(acceptance)
        if penalty_scale == 0.0 and lagrangian_scale == 0.0:
            return grad_theta
        return grad_theta + (penalty_scale + lagrangian_scale) * self.mean_acceptance_grad(theta_arr, x_batch)

    def policy_value(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> np.ndarray:
        """Evaluate policy actions on prepared numeric policy features."""
        _, _, _, features = self._arrays(x_batch)
        return np.asarray(self.policy.value(np.asarray(theta, dtype=float), features), dtype=float)

    def policy_grad(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> np.ndarray:
        """Evaluate policy Jacobians on prepared numeric policy features."""
        _, _, _, features = self._arrays(x_batch)
        return np.asarray(self.policy.grad(np.asarray(theta, dtype=float), features), dtype=float)

    def policy_weighted_grad(
        self,
        theta: np.ndarray,
        x_batch: np.ndarray | PreparedGLMBatch,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Evaluate weighted policy-gradient sum on prepared numeric features."""
        _, _, _, features = self._arrays(x_batch)
        return np.asarray(
            self.policy.weighted_grad(np.asarray(theta, dtype=float), features, np.asarray(weights, dtype=float)),
            dtype=float,
        )

    def policy_input_dim(self) -> int:
        """Return the prepared feature dimension seen by the policy."""
        return self.policy_feature_dim

    def policy_theta_dim(self, state_dim: int | None = None) -> int:
        """Return theta dimension required by the policy over prepared features."""
        del state_dim
        return policy_theta_dim(self.policy, self.policy_feature_dim)

    def value_at_u(self, x_batch: np.ndarray | PreparedGLMBatch, u: float) -> float:
        """Compute mean objective value at a fixed action."""
        base_logit, loss, premium, _ = self._arrays(x_batch)
        u_val = float(u)
        if self.u_bounds is not None:
            u_val = float(np.clip(u_val, *self.u_bounds))
        u_arr = np.full(base_logit.shape[0], u_val, dtype=float)
        acceptance = self._acceptance(base_logit, u_arr)
        base_value = float(np.mean(self._value_batch_from_arrays(acceptance, loss, premium, u_arr)))
        penalty_value, _ = self._acceptance_penalty(acceptance)
        lagrangian_value, _ = self._lagrangian_adjustment(acceptance)
        return base_value + penalty_value + lagrangian_value

    def base_value_at_u(self, x_batch: np.ndarray | PreparedGLMBatch, u: float) -> float:
        """Return the raw pricing objective at a fixed action."""
        base_logit, loss, premium, _ = self._arrays(x_batch)
        u_val = float(u)
        if self.u_bounds is not None:
            u_val = float(np.clip(u_val, *self.u_bounds))
        u_arr = np.full(base_logit.shape[0], u_val, dtype=float)
        acceptance = self._acceptance(base_logit, u_arr)
        return float(np.mean(self._value_batch_from_arrays(acceptance, loss, premium, u_arr)))

    def mean_acceptance(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> float:
        """Return mean acceptance under the current policy."""
        base_logit, _, _, features = self._arrays(x_batch)
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self._clip_u(self.policy.value(theta_arr, features))
        return float(np.mean(self._acceptance(base_logit, u_batch)))

    def mean_acceptance_at_u(self, x_batch: np.ndarray | PreparedGLMBatch, u: float) -> float:
        """Return mean acceptance at a fixed action."""
        base_logit, _, _, _ = self._arrays(x_batch)
        u_val = float(u)
        if self.u_bounds is not None:
            u_val = float(np.clip(u_val, *self.u_bounds))
        u_arr = np.full(base_logit.shape[0], u_val, dtype=float)
        return float(np.mean(self._acceptance(base_logit, u_arr)))

    def mean_acceptance_grad(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> np.ndarray:
        """Return theta-gradient of mean acceptance under the current policy."""
        base_logit, _, _, features = self._arrays(x_batch)
        theta_arr = np.asarray(theta, dtype=float)
        u_raw = self.policy.value(theta_arr, features)
        u_clipped = self._clip_u(u_raw)
        d_acceptance_du = self._d_acceptance_du(base_logit, u_clipped)
        if self.u_bounds is not None:
            interior = (u_raw > self.u_bounds[0]) & (u_raw < self.u_bounds[1])
            d_acceptance_du = d_acceptance_du * interior
        return _theta_grad_from_u_grad(self.policy, theta_arr, features, d_acceptance_du)

    def _step_metrics(self, theta: np.ndarray, x_batch: np.ndarray | PreparedGLMBatch) -> dict[str, float]:
        """Return per-step mean metrics for reporter logging."""
        base_logit, loss, premium, features = self._arrays(x_batch)
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self._clip_u(self.policy.value(theta_arr, features))
        acceptance = self._acceptance(base_logit, u_batch)
        revenue = (u_batch + 1.0) * premium
        return {
            "mean_acceptance": float(np.mean(acceptance)),
            "projected_loss": float(np.mean(loss)),
            "projected_revenue": float(np.mean(revenue)),
        }


def prepare_glm_batch(
    objective: Any,
    x_batch: Any,
    *,
    row_indices: np.ndarray | None = None,
) -> PreparedGLMBatch:
    """Materialize a GLM-backed ``ModelBasedObjective`` into numeric arrays."""
    coeffs_fn = getattr(objective, "_glm_acceptance_coefficients", None)
    base_logit_fn = getattr(objective, "_glm_acceptance_base_logit", None)
    loss_coeffs_fn = getattr(objective, "_linear_loss_coefficients", None)
    loss_prediction_fn = getattr(objective, "_loss_prediction", None)
    premium_values_fn = getattr(objective, "_premium_values", None)
    policy_features_fn = getattr(objective, "_policy_features", None)
    if not all(
        callable(fn)
        for fn in (
            coeffs_fn,
            base_logit_fn,
            loss_prediction_fn,
            premium_values_fn,
            policy_features_fn,
        )
    ):
        raise ValueError("objective must expose ModelBasedObjective GLM preparation hooks.")

    coeffs = coeffs_fn()
    if coeffs is None:
        raise ValueError("objective does not expose GLM acceptance coefficients.")
    if getattr(objective, "loss_source", "predicted") != "observed":
        if not callable(loss_coeffs_fn) or loss_coeffs_fn() is None:
            raise ValueError("prepared GLM objective requires linear loss coefficients or observed loss.")

    base_logit = base_logit_fn(x_batch)
    if base_logit is None:
        raise ValueError("could not materialize GLM base logits for x_batch.")

    u_coef_fn = getattr(objective, "_effective_glm_u_coef", None)
    u_coef = float(u_coef_fn(coeffs)) if callable(u_coef_fn) else float(coeffs["u_coef"])
    return PreparedGLMBatch.from_arrays(
        base_logit=np.asarray(base_logit, dtype=float),
        loss=np.asarray(loss_prediction_fn(x_batch), dtype=float),
        premium=np.asarray(premium_values_fn(x_batch), dtype=float),
        policy_features=np.asarray(policy_features_fn(x_batch), dtype=float),
        u_coef=u_coef,
        probability_target=coeffs.get("probability_target", "acceptance"),
        row_indices=row_indices,
    )


def prepare_glm_objective(
    objective: Any,
    x_batch: Any,
    *,
    row_indices: np.ndarray | None = None,
) -> tuple[PreparedGLMObjective, PreparedGLMBatch]:
    """Return a prepared NumPy objective and its compact numeric batch."""
    batch = prepare_glm_batch(objective, x_batch, row_indices=row_indices)
    prepared = PreparedGLMObjective(
        policy=objective.policy,
        policy_feature_dim=batch.policy_feature_dim,
        u_coef=batch.u_coef,
        probability_target=batch.probability_target,
        u_bounds=getattr(objective, "u_bounds", None),
        acceptance_floor=getattr(objective, "acceptance_floor", None),
        acceptance_penalty_weight=getattr(objective, "acceptance_penalty_weight", None),
        acceptance_penalty_temperature=getattr(objective, "acceptance_penalty_temperature", 0.01),
        lagrangian_lambda=getattr(objective, "lagrangian_lambda", None),
    )
    return prepared, batch


def _as_vector(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite values.")
    return arr


def _prepared_x_array(x_batch: np.ndarray | PreparedGLMBatch) -> np.ndarray:
    if isinstance(x_batch, PreparedGLMBatch):
        return x_batch.x_array
    x_arr = np.asarray(x_batch, dtype=float)
    if x_arr.ndim != 2:
        raise ValueError("x_batch must be a 2D prepared array.")
    if x_arr.shape[1] <= _N_METADATA_COLS:
        raise ValueError("x_batch must include prepared policy feature columns.")
    if not np.isfinite(x_arr).all():
        raise ValueError("x_batch must contain finite values.")
    return x_arr


__all__ = [
    "PreparedGLMBatch",
    "PreparedGLMObjective",
    "prepare_glm_batch",
    "prepare_glm_objective",
]
