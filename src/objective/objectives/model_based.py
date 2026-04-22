"""Model-based objective using trained sklearn/XGBoost artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from objective.base import Objective, Policy
from objective.utils import _theta_grad_from_u_grad


@dataclass(frozen=True)
class ModelBasedObjective(Objective):
    r"""Pricing objective backed by trained ML models.

    $$f(u; x) = a(x,u) \cdot (\hat{Y}(x) - (u + 1) \cdot p(x))$$

    where $$a(x,u)$$ is acceptance derived from the bundled churn classifier,
    $$\hat{Y}(x)$$ is the expected financial loss (LinearRegression or XGBRegressor),
    and $$p(x)$$ is the policy premium extracted from column ``premium_col`` of x.
    The policy output ``u`` stays centered at 0, so ``u = 0`` means the baseline
    premium multiplier and revenue uses ``(u + 1) * p(x)``.

    ``acceptance_model`` expects a DataFrame with ``acceptance_state_cols + ["U"]`` and
    returns churn probability in class 1, which this objective maps to acceptance via
    ``1 - p_churn``. ``loss_model`` expects a DataFrame with ``loss_cols``.

    If ``u_coef`` is provided, it is interpreted as $$d\,\text{logit}(p_{churn}) / dU$$,
    so the analytical acceptance gradient uses the opposite sign.
    Otherwise numerical central finite differences are used (XGBoost path).

    When ``acceptance_floor`` and ``acceptance_penalty_weight`` are both set,
    ``value()`` adds a smooth mean-acceptance penalty. Direct trust-region
    constraints are handled at the optimizer level via ``step_rule="trust-constr"``.
    """

    policy: Policy
    acceptance_model: Any
    loss_model: Any
    acceptance_state_cols: tuple[str, ...]
    loss_cols: tuple[str, ...]
    premium_col: int = 9
    u_coef: float | None = None
    u_bounds: tuple[float, float] | None = None
    acceptance_floor: float | None = None
    acceptance_penalty_weight: float | None = None
    acceptance_penalty_temperature: float = 0.01
    _fd_eps: float = 1e-4

    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        """Clip u to bounds if set."""
        if self.u_bounds is not None:
            return np.clip(u, *self.u_bounds)
        return u

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        """Compute mean objective value across batch."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be 2D.")
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self._clip_u(self.policy_value(theta_arr, x_arr))
        base_value = float(np.mean(self._value_batch(x_arr, u_batch)))
        return base_value + self._acceptance_penalty(self._acceptance_proba(x_arr, u_batch))[0]

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Compute theta-gradient via chain rule: df/dtheta = mean(df/du * du/dtheta)."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be 2D.")
        theta_arr = np.asarray(theta, dtype=float)
        u_raw = self.policy_value(theta_arr, x_arr)
        u_clipped = self._clip_u(u_raw)
        grad_u = self._grad_u_batch(x_arr, u_clipped)
        # Zero gradient for samples where u was clipped (subgradient)
        if self.u_bounds is not None:
            interior = (u_raw > self.u_bounds[0]) & (u_raw < self.u_bounds[1])
            grad_u = grad_u * interior
        grad_theta = _theta_grad_from_u_grad(self, theta_arr, x_arr, grad_u)
        penalty_value, penalty_scale = self._acceptance_penalty(self._acceptance_proba(x_arr, u_clipped))
        del penalty_value
        if penalty_scale == 0.0:
            return grad_theta
        return grad_theta + penalty_scale * self.mean_acceptance_grad(theta_arr, x_arr)

    def policy_value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Evaluate policy actions on acceptance-preprocessed features from raw state."""
        theta_arr = np.asarray(theta, dtype=float)
        x_arr = np.asarray(x_batch, dtype=float)
        return np.asarray(self.policy.value(theta_arr, self._policy_features(x_arr)), dtype=float)

    def policy_grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Evaluate policy Jacobians on acceptance-preprocessed features from raw state."""
        theta_arr = np.asarray(theta, dtype=float)
        x_arr = np.asarray(x_batch, dtype=float)
        return np.asarray(self.policy.grad(theta_arr, self._policy_features(x_arr)), dtype=float)

    def value_at_u(self, x_batch: np.ndarray, u: float) -> float:
        """Compute mean objective value at a fixed action u."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be 2D.")
        u_val = float(u)
        if self.u_bounds is not None:
            u_val = float(np.clip(u_val, *self.u_bounds))
        u_arr = np.full(x_arr.shape[0], u_val, dtype=float)
        base_value = float(np.mean(self._value_batch(x_arr, u_arr)))
        return base_value + self._acceptance_penalty(self._acceptance_proba(x_arr, u_arr))[0]

    def mean_acceptance(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        """Return mean acceptance probability under the current policy."""
        x_arr = np.asarray(x_batch, dtype=float)
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self._clip_u(self.policy_value(theta_arr, x_arr))
        return float(np.mean(self._acceptance_proba(x_arr, u_batch)))

    def mean_acceptance_at_u(self, x_batch: np.ndarray, u: float) -> float:
        """Return mean acceptance probability at a fixed action u."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be 2D.")
        u_val = float(u)
        if self.u_bounds is not None:
            u_val = float(np.clip(u_val, *self.u_bounds))
        u_arr = np.full(x_arr.shape[0], u_val, dtype=float)
        return float(np.mean(self._acceptance_proba(x_arr, u_arr)))

    def _step_metrics(self, theta: np.ndarray, x_batch: np.ndarray) -> dict[str, float]:
        """Return per-step mean metrics for reporter logging."""
        x_arr = np.asarray(x_batch, dtype=float)
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self._clip_u(self.policy_value(theta_arr, x_arr))
        acceptance = self._acceptance_proba(x_arr, u_batch)
        loss = self._loss_prediction(x_arr)
        premium = x_arr[:, self.premium_col]
        revenue = (u_batch + 1.0) * premium
        return {
            "mean_acceptance": float(np.mean(acceptance)),
            "projected_loss": float(np.mean(loss)),
            "projected_revenue": float(np.mean(revenue)),
        }

    def mean_acceptance_grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return theta-gradient of mean acceptance under the current policy."""
        x_arr = np.asarray(x_batch, dtype=float)
        theta_arr = np.asarray(theta, dtype=float)
        u_raw = self.policy_value(theta_arr, x_arr)
        u_clipped = self._clip_u(u_raw)
        d_acceptance_du = self._d_acceptance_du_batch(x_arr, u_clipped)
        if self.u_bounds is not None:
            interior = (u_raw > self.u_bounds[0]) & (u_raw < self.u_bounds[1])
            d_acceptance_du = d_acceptance_du * interior
        return _theta_grad_from_u_grad(self, theta_arr, x_arr, d_acceptance_du)

    @staticmethod
    def _artifact_model(artifact: Any) -> Any:
        return getattr(artifact, "model", artifact)

    @staticmethod
    def _artifact_preprocessor(artifact: Any) -> Any:
        return getattr(artifact, "preprocessor", None)

    @staticmethod
    def _artifact_x_feature_cols(artifact: Any, fallback_cols: tuple[str, ...]) -> tuple[str, ...]:
        cols = getattr(artifact, "x_feature_cols", None)
        return tuple(cols) if cols is not None else fallback_cols

    @staticmethod
    def _artifact_frame(artifact: Any, raw_frame: pd.DataFrame) -> pd.DataFrame:
        model_frame = getattr(artifact, "model_frame", None)
        if callable(model_frame):
            return model_frame(raw_frame)
        return raw_frame

    def _policy_features(self, x_batch: np.ndarray) -> np.ndarray:
        """Return the acceptance-side processed features used by the policy."""
        x_state = x_batch[:, : len(self.acceptance_state_cols)]
        raw_df = pd.DataFrame(x_state, columns=list(self.acceptance_state_cols))
        x_feature_cols = self._artifact_x_feature_cols(self.acceptance_model, self.acceptance_state_cols)
        state_df = raw_df.loc[:, list(x_feature_cols)].copy()
        preprocessor = self._artifact_preprocessor(self.acceptance_model)
        if preprocessor is None:
            return state_df.to_numpy(dtype=float)
        processed = preprocessor.transform(state_df)
        return np.asarray(processed, dtype=float)

    def _churn_proba(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        """Call classifier and return class-1 churn probability, shape (n,)."""
        x_state = x_batch[:, : len(self.acceptance_state_cols)]
        raw_df = pd.DataFrame(
            np.column_stack([x_state, u_arr]),
            columns=list(self.acceptance_state_cols) + ["U"],
        )
        model_df = self._artifact_frame(self.acceptance_model, raw_df)
        model = self._artifact_model(self.acceptance_model)
        return np.asarray(model.predict_proba(model_df)[:, 1], dtype=float)

    def _acceptance_proba(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        """Return acceptance probability by flipping the bundled churn classifier output."""
        return 1.0 - self._churn_proba(x_batch, u_arr)

    def _loss_prediction(self, x_batch: np.ndarray) -> np.ndarray:
        """Call loss model on loss_cols subset of x_batch. Returns shape (n,)."""
        x_loss = x_batch[:, : len(self.loss_cols)]
        raw_df = pd.DataFrame(x_loss, columns=list(self.loss_cols))
        model_df = self._artifact_frame(self.loss_model, raw_df)
        model = self._artifact_model(self.loss_model)
        return np.asarray(model.predict(model_df), dtype=float)

    def _value_batch(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        """Compute per-sample objective values."""
        acceptance = self._acceptance_proba(x_batch, u_arr)
        loss = self._loss_prediction(x_batch)
        premium = x_batch[:, self.premium_col]
        revenue = (u_arr + 1.0) * premium
        return acceptance * (loss - revenue)

    def _d_acceptance_du_batch(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        """Compute d acceptance / du for each sample."""
        acceptance = self._acceptance_proba(x_batch, u_arr)
        if self.u_coef is not None:
            return -acceptance * (1.0 - acceptance) * self.u_coef
        eps = self._fd_eps
        a_plus = self._acceptance_proba(x_batch, u_arr + eps)
        a_minus = self._acceptance_proba(x_batch, u_arr - eps)
        return (a_plus - a_minus) / (2.0 * eps)

    def _acceptance_penalty(self, acceptance: np.ndarray) -> tuple[float, float]:
        """Return ``(penalty_value, d penalty / d mean_acceptance)``."""
        if self.acceptance_floor is None or self.acceptance_penalty_weight is None:
            return 0.0, 0.0
        mean_acceptance = float(np.mean(acceptance))
        gap = float(self.acceptance_floor) - mean_acceptance
        temp = float(self.acceptance_penalty_temperature)
        scaled_gap = gap / temp
        soft_gap = temp * float(np.logaddexp(0.0, scaled_gap))
        sigmoid_gap = 1.0 / (1.0 + np.exp(-scaled_gap))
        weight = float(self.acceptance_penalty_weight)
        penalty_value = weight * soft_gap * soft_gap
        penalty_grad_mean = -2.0 * weight * soft_gap * sigmoid_gap
        return penalty_value, penalty_grad_mean

    def _grad_u_batch(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        """Compute df/du for each sample."""
        acceptance = self._acceptance_proba(x_batch, u_arr)
        loss = self._loss_prediction(x_batch)
        premium = x_batch[:, self.premium_col]
        revenue = (u_arr + 1.0) * premium

        d_acceptance_du = self._d_acceptance_du_batch(x_batch, u_arr)
        return d_acceptance_du * (loss - revenue) - acceptance * premium


__all__ = ["ModelBasedObjective"]
