"""Model-based objective using trained sklearn/XGBoost artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any

import numpy as np
import pandas as pd

from objective._math import _sigmoid
from objective.base import Objective, Policy
from objective.policy import policy_theta_dim
from objective.policy_preprocessing import PolicyFeaturePreprocessor
from objective.utils import _theta_grad_from_u_grad


@dataclass(frozen=True)
class ModelBasedObjective(Objective):
    r"""Pricing objective backed by trained ML models.

    $$f(u; x) = a(x,u) \cdot (\hat{Y}(x) - (u + 1) \cdot p(x))$$

    where $$a(x,u)$$ is acceptance derived from the bundled acceptance classifier,
    $$\hat{Y}(x)$$ is the expected financial loss (LinearRegression or XGBRegressor),
    and $$p(x)$$ is the policy premium extracted from column ``premium_col`` of x.
    The policy output ``u`` stays centered at 0, so ``u = 0`` means the baseline
    premium multiplier and revenue uses ``(u + 1) * p(x)``.

    ``acceptance_model`` expects a raw DataFrame with ``acceptance_state_cols + ["U"]``;
    for the canonical 052726 artifacts class 1 is direct acceptance probability.
    ``loss_model`` expects a raw DataFrame with ``loss_cols``. When GLM coefficients
    can be extracted from the artifacts, value/acceptance calls use the equivalent
    array formula instead of repeated sklearn predictions.

    If ``u_coef`` is provided, it is interpreted as $$d\,\text{logit}(p_{accept}) / dU$$
    for direct-acceptance artifacts. Otherwise numerical central finite differences
    are used (XGBoost path).

    When ``acceptance_floor`` and ``acceptance_penalty_weight`` are both set,
    ``value()`` adds a smooth mean-acceptance penalty. When ``lagrangian_lambda``
    is set, ``value()`` optimizes the scalarized target
    $$J(\theta) + \lambda (\text{floor} - \bar{a}(\theta))$$ while
    ``base_value()`` keeps exposing the raw pricing objective $$J(\theta)$$ for
    reporting. Direct trust-region constraints are handled at the optimizer level
    via ``step_rule="trust-constr"``.
    """

    policy: Policy
    acceptance_model: Any
    loss_model: Any
    acceptance_state_cols: tuple[str, ...]
    loss_cols: tuple[str, ...]
    premium_col: int | str = "X_policy_premium"
    u_coef: float | None = None
    u_bounds: tuple[float, float] | None = None
    acceptance_floor: float | None = None
    acceptance_penalty_weight: float | None = None
    acceptance_penalty_temperature: float = 0.01
    lagrangian_lambda: float | None = None
    policy_preprocessor: PolicyFeaturePreprocessor | None = None
    policy_feature_cols: tuple[str, ...] | None = None
    _fd_eps: float = 1e-4
    _eval_counts: dict[str, float] = field(default_factory=dict, compare=False, repr=False)
    _cache: dict[object, Any] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def _clip_u(self, u: np.ndarray) -> np.ndarray:
        """Clip u to bounds if set."""
        if self.u_bounds is not None:
            return np.clip(u, *self.u_bounds)
        return u

    def reset_eval_counts(self) -> None:
        """Reset mutable diagnostic counters for model/objective evaluations."""
        self._eval_counts.clear()

    def eval_counts(self) -> dict[str, float]:
        """Return diagnostic counts for objective and model prediction calls."""
        return dict(self._eval_counts)

    def _record_eval(self, key: str, rows: int | None = None) -> None:
        self._eval_counts[key] = self._eval_counts.get(key, 0) + 1
        if rows is not None:
            row_key = f"{key}_rows"
            self._eval_counts[row_key] = self._eval_counts.get(row_key, 0) + int(rows)

    def _record_time(self, key: str, elapsed: float) -> None:
        self._eval_counts[key] = self._eval_counts.get(key, 0.0) + float(elapsed)

    def _row_count(self, x_batch: Any) -> int:
        return int(x_batch.shape[0])

    def _x_frame(self, x_batch: Any) -> pd.DataFrame:
        """Return raw source-space X columns as a DataFrame."""
        if isinstance(x_batch, pd.DataFrame):
            if x_batch.ndim != 2:
                raise ValueError("x_batch must be 2D.")
            return x_batch.reset_index(drop=True)
        x_arr = np.asarray(x_batch)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be 2D.")
        if x_arr.shape[1] != len(self.acceptance_state_cols):
            raise ValueError(
                f"x_batch has {x_arr.shape[1]} columns but expected "
                f"{len(self.acceptance_state_cols)} acceptance-state columns."
            )
        return pd.DataFrame(x_arr, columns=list(self.acceptance_state_cols))

    def _cache_key(self, name: str, x_batch: Any) -> tuple[object, ...]:
        if isinstance(x_batch, pd.DataFrame):
            return (name, "df", id(x_batch), tuple(x_batch.shape), tuple(x_batch.columns))
        x_arr = np.asarray(x_batch)
        return (name, "array", id(x_arr), tuple(x_arr.shape), str(x_arr.dtype))

    def _premium_values(self, x_batch: Any) -> np.ndarray:
        frame = self._x_frame(x_batch)
        if isinstance(self.premium_col, str):
            if self.premium_col not in frame.columns:
                raise ValueError(f"Missing premium column '{self.premium_col}' in x_batch.")
            return frame[self.premium_col].to_numpy(dtype=float)
        return frame.iloc[:, int(self.premium_col)].to_numpy(dtype=float)

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        """Compute mean objective value across batch."""
        start = time.perf_counter()
        self._x_frame(x_batch)
        try:
            self._record_eval("objective_value_calls", self._row_count(x_batch))
            theta_arr = np.asarray(theta, dtype=float)
            u_batch = self._clip_u(self.policy_value(theta_arr, x_batch))
            acceptance = self._acceptance_proba(x_batch, u_batch)
            base_value = self._mean_base_value(x_batch, u_batch)
            penalty_value, _ = self._acceptance_penalty(acceptance)
            lagrangian_value, _ = self._lagrangian_adjustment(acceptance)
            return base_value + penalty_value + lagrangian_value
        finally:
            self._record_time("objective_value_seconds", time.perf_counter() - start)

    def base_value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        """Return the raw pricing objective without penalties or Lagrangian terms."""
        self._x_frame(x_batch)
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self._clip_u(self.policy_value(theta_arr, x_batch))
        return self._mean_base_value(x_batch, u_batch)

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Compute theta-gradient via chain rule: df/dtheta = mean(df/du * du/dtheta)."""
        self._x_frame(x_batch)
        theta_arr = np.asarray(theta, dtype=float)
        u_raw = self.policy_value(theta_arr, x_batch)
        u_clipped = self._clip_u(u_raw)
        grad_u = self._grad_u_batch(x_batch, u_clipped)
        # Zero gradient for samples where u was clipped (subgradient)
        if self.u_bounds is not None:
            interior = (u_raw > self.u_bounds[0]) & (u_raw < self.u_bounds[1])
            grad_u = grad_u * interior
        grad_theta = _theta_grad_from_u_grad(self, theta_arr, x_batch, grad_u)
        acceptance = self._acceptance_proba(x_batch, u_clipped)
        penalty_value, penalty_scale = self._acceptance_penalty(acceptance)
        del penalty_value
        lagrangian_value, lagrangian_scale = self._lagrangian_adjustment(acceptance)
        del lagrangian_value
        if penalty_scale == 0.0 and lagrangian_scale == 0.0:
            return grad_theta
        return grad_theta + (penalty_scale + lagrangian_scale) * self.mean_acceptance_grad(theta_arr, x_batch)

    def policy_value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Evaluate policy actions on acceptance-preprocessed features from raw state."""
        theta_arr = np.asarray(theta, dtype=float)
        return np.asarray(self.policy.value(theta_arr, self._policy_features(x_batch)), dtype=float)

    def policy_grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Evaluate policy Jacobians on acceptance-preprocessed features from raw state."""
        theta_arr = np.asarray(theta, dtype=float)
        return np.asarray(self.policy.grad(theta_arr, self._policy_features(x_batch)), dtype=float)

    def policy_input_dim(self) -> int:
        """Return the processed state dimension seen by the policy."""
        if self.policy_preprocessor is not None:
            output_dim = getattr(self.policy_preprocessor, "output_dim_", None)
            if output_dim is None:
                raise ValueError("policy_preprocessor must be fitted before theta sizing.")
            return int(output_dim)
        feature_dim_fn = getattr(self.acceptance_model, "policy_feature_dim", None)
        if callable(feature_dim_fn):
            return int(feature_dim_fn())
        return len(self._artifact_x_feature_cols(self.acceptance_model, self.acceptance_state_cols))

    def policy_theta_dim(self, state_dim: int | None = None) -> int:
        """Return theta dimension required by the policy over processed features."""
        del state_dim
        return policy_theta_dim(self.policy, self.policy_input_dim())

    def value_at_u(self, x_batch: np.ndarray, u: float) -> float:
        """Compute mean objective value at a fixed action u."""
        self._x_frame(x_batch)
        u_val = float(u)
        if self.u_bounds is not None:
            u_val = float(np.clip(u_val, *self.u_bounds))
        u_arr = np.full(self._row_count(x_batch), u_val, dtype=float)
        acceptance = self._acceptance_proba(x_batch, u_arr)
        base_value = self._mean_base_value(x_batch, u_arr)
        penalty_value, _ = self._acceptance_penalty(acceptance)
        lagrangian_value, _ = self._lagrangian_adjustment(acceptance)
        return base_value + penalty_value + lagrangian_value

    def base_value_at_u(self, x_batch: np.ndarray, u: float) -> float:
        """Return the raw pricing objective at a fixed action u."""
        self._x_frame(x_batch)
        u_val = float(u)
        if self.u_bounds is not None:
            u_val = float(np.clip(u_val, *self.u_bounds))
        u_arr = np.full(self._row_count(x_batch), u_val, dtype=float)
        return self._mean_base_value(x_batch, u_arr)

    def mean_acceptance(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        """Return mean acceptance probability under the current policy."""
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self._clip_u(self.policy_value(theta_arr, x_batch))
        return float(np.mean(self._acceptance_proba(x_batch, u_batch)))

    def mean_acceptance_at_u(self, x_batch: np.ndarray, u: float) -> float:
        """Return mean acceptance probability at a fixed action u."""
        self._x_frame(x_batch)
        u_val = float(u)
        if self.u_bounds is not None:
            u_val = float(np.clip(u_val, *self.u_bounds))
        u_arr = np.full(self._row_count(x_batch), u_val, dtype=float)
        return float(np.mean(self._acceptance_proba(x_batch, u_arr)))

    def _step_metrics(self, theta: np.ndarray, x_batch: np.ndarray) -> dict[str, float]:
        """Return per-step mean metrics for reporter logging."""
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self._clip_u(self.policy_value(theta_arr, x_batch))
        acceptance = self._acceptance_proba(x_batch, u_batch)
        loss = self._loss_prediction(x_batch)
        premium = self._premium_values(x_batch)
        revenue = (u_batch + 1.0) * premium
        return {
            "mean_acceptance": float(np.mean(acceptance)),
            "projected_loss": float(np.mean(loss)),
            "projected_revenue": float(np.mean(revenue)),
        }

    def mean_acceptance_grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Return theta-gradient of mean acceptance under the current policy."""
        theta_arr = np.asarray(theta, dtype=float)
        u_raw = self.policy_value(theta_arr, x_batch)
        u_clipped = self._clip_u(u_raw)
        d_acceptance_du = self._d_acceptance_du_batch(x_batch, u_clipped)
        if self.u_bounds is not None:
            interior = (u_raw > self.u_bounds[0]) & (u_raw < self.u_bounds[1])
            d_acceptance_du = d_acceptance_du * interior
        return _theta_grad_from_u_grad(self, theta_arr, x_batch, d_acceptance_du)

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

    def _glm_churn_coefficients(self) -> dict[str, Any] | None:
        key = ("glm_churn_coefficients",)
        if key not in self._cache:
            try:
                from data.loader import extract_glm_churn_coefficients

                coeffs = extract_glm_churn_coefficients(self.acceptance_model)
            except (AttributeError, KeyError, TypeError, ValueError):
                coeffs = None
            self._cache[key] = coeffs
        return self._cache[key]

    def _linear_loss_coefficients(self) -> dict[str, Any] | None:
        key = ("linear_loss_coefficients",)
        if key not in self._cache:
            try:
                from data.loader import extract_linear_loss_coefficients

                coeffs = extract_linear_loss_coefficients(self.loss_model)
            except (AttributeError, KeyError, TypeError, ValueError):
                coeffs = None
            self._cache[key] = coeffs
        return self._cache[key]

    def _glm_churn_base_logit(self, x_batch: Any) -> np.ndarray | None:
        coeffs = self._glm_churn_coefficients()
        if coeffs is None:
            return None
        key = self._cache_key("glm_churn_base_logit", x_batch)
        if key in self._cache:
            self._record_eval("glm_churn_base_logit_cache_hits", self._row_count(x_batch))
            cached = self._cache[key]
            return None if cached is None else np.asarray(cached, dtype=float)

        n_rows = self._row_count(x_batch)
        self._record_eval("glm_churn_base_logit_cache_misses", n_rows)
        raw_df = self._x_frame(x_batch).loc[:, list(self.acceptance_state_cols)].copy()
        raw_df["U"] = np.zeros(n_rows, dtype=float)
        model_df = self._artifact_frame(self.acceptance_model, raw_df)
        feature_names = list(coeffs["x_feature_names"])
        if any(name not in model_df.columns for name in feature_names):
            self._cache[key] = None
            return None
        x_matrix = model_df.loc[:, feature_names].to_numpy(dtype=float)
        coef = np.asarray(coeffs["x_coef"], dtype=float)
        if x_matrix.shape[1] != coef.shape[0]:
            self._cache[key] = None
            return None
        base_logit = float(coeffs["intercept"]) + x_matrix @ coef
        self._cache[key] = base_logit
        return base_logit

    def _glm_acceptance_proba(self, x_batch: Any, u_arr: np.ndarray) -> np.ndarray | None:
        coeffs = self._glm_churn_coefficients()
        if coeffs is None:
            return None
        base_logit = self._glm_churn_base_logit(x_batch)
        if base_logit is None:
            return None
        self._record_eval("acceptance_analytic_calls", self._row_count(x_batch))
        start = time.perf_counter()
        try:
            class1 = _sigmoid(base_logit + float(coeffs["u_coef"]) * u_arr)
            if coeffs.get("probability_target", getattr(self.acceptance_model, "probability_target", "churn")) == "acceptance":
                return class1
            return 1.0 - class1
        finally:
            self._record_time("acceptance_analytic_seconds", time.perf_counter() - start)

    def _linear_loss_prediction_from_coefficients(self, x_batch: Any) -> np.ndarray | None:
        coeffs = self._linear_loss_coefficients()
        if coeffs is None:
            return None
        raw_df = self._x_frame(x_batch).loc[:, list(self.loss_cols)].copy()
        model_df = self._artifact_frame(self.loss_model, raw_df)
        feature_names = list(coeffs["x_feature_names"])
        if any(name not in model_df.columns for name in feature_names):
            return None
        x_matrix = model_df.loc[:, feature_names].to_numpy(dtype=float)
        coef = np.asarray(coeffs["x_coef"], dtype=float)
        if x_matrix.shape[1] != coef.shape[0]:
            return None
        self._record_eval("loss_analytic_calls", self._row_count(x_batch))
        start = time.perf_counter()
        try:
            return float(coeffs["intercept"]) + x_matrix @ coef
        finally:
            self._record_time("loss_analytic_seconds", time.perf_counter() - start)

    def _policy_features(self, x_batch: Any) -> np.ndarray:
        """Return the acceptance-side processed features used by the policy."""
        key = self._cache_key("policy_features", x_batch)
        cached = self._cache.get(key)
        if cached is not None:
            self._record_eval("policy_features_cache_hits", self._row_count(x_batch))
            return np.asarray(cached, dtype=float)
        self._record_eval("policy_features_cache_misses", self._row_count(x_batch))
        raw_df = self._x_frame(x_batch)
        if self.policy_preprocessor is not None:
            if self.policy_feature_cols is None:
                x_feature_cols = self._artifact_x_feature_cols(self.acceptance_model, self.acceptance_state_cols)
                state_df = raw_df.loc[:, list(x_feature_cols)].copy()
                preprocessor = self._artifact_preprocessor(self.acceptance_model)
                base_features = state_df.to_numpy(dtype=float) if preprocessor is None else np.asarray(preprocessor.transform(state_df), dtype=float)
            else:
                policy_cols = self.policy_feature_cols
                missing = [col for col in policy_cols if col not in raw_df.columns]
                if missing:
                    raise ValueError(f"Missing policy feature columns: {missing}")
                base_features = raw_df.loc[:, list(policy_cols)].to_numpy(dtype=float)
            features = self.policy_preprocessor.transform(base_features)
        else:
            x_feature_cols = self._artifact_x_feature_cols(self.acceptance_model, self.acceptance_state_cols)
            state_df = raw_df.loc[:, list(x_feature_cols)].copy()
            preprocessor = self._artifact_preprocessor(self.acceptance_model)
            if preprocessor is None:
                features = state_df.to_numpy(dtype=float)
            else:
                processed = preprocessor.transform(state_df)
                features = np.asarray(processed, dtype=float)
        features_arr = np.asarray(features, dtype=float)
        self._cache[key] = features_arr
        return features_arr

    def _churn_proba(self, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
        """Call classifier and return class-1 probability, shape (n,)."""
        raw_df = self._x_frame(x_batch).loc[:, list(self.acceptance_state_cols)].copy()
        raw_df["U"] = np.asarray(u_arr, dtype=float)
        model_df = self._artifact_frame(self.acceptance_model, raw_df)
        model = self._artifact_model(self.acceptance_model)
        self._record_eval("acceptance_predict_calls", self._row_count(x_batch))
        start = time.perf_counter()
        try:
            return np.asarray(model.predict_proba(model_df)[:, 1], dtype=float)
        finally:
            self._record_time("acceptance_predict_seconds", time.perf_counter() - start)

    def _acceptance_proba(self, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
        """Return acceptance probability for policy-generated actions."""
        analytical = self._glm_acceptance_proba(x_batch, u_arr)
        if analytical is not None:
            return analytical
        class1 = self._churn_proba(x_batch, u_arr)
        if getattr(self.acceptance_model, "probability_target", "churn") == "acceptance":
            return class1
        return 1.0 - class1

    def _loss_prediction(self, x_batch: Any) -> np.ndarray:
        """Call loss model on loss_cols subset of x_batch. Returns shape (n,)."""
        key = self._cache_key("loss_prediction", x_batch)
        cached = self._cache.get(key)
        if cached is not None:
            self._record_eval("loss_prediction_cache_hits", self._row_count(x_batch))
            return np.asarray(cached, dtype=float)
        self._record_eval("loss_prediction_cache_misses", self._row_count(x_batch))
        analytical = self._linear_loss_prediction_from_coefficients(x_batch)
        if analytical is not None:
            self._cache[key] = analytical
            return analytical
        raw_df = self._x_frame(x_batch).loc[:, list(self.loss_cols)].copy()
        model_df = self._artifact_frame(self.loss_model, raw_df)
        model = self._artifact_model(self.loss_model)
        self._record_eval("loss_predict_calls", self._row_count(x_batch))
        start = time.perf_counter()
        try:
            prediction = np.asarray(model.predict(model_df), dtype=float)
        finally:
            self._record_time("loss_predict_seconds", time.perf_counter() - start)
        self._cache[key] = prediction
        return prediction

    def _value_batch(self, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
        """Compute per-sample objective values."""
        acceptance = self._acceptance_proba(x_batch, u_arr)
        loss = self._loss_prediction(x_batch)
        premium = self._premium_values(x_batch)
        revenue = (u_arr + 1.0) * premium
        return acceptance * (loss - revenue)

    def _d_acceptance_du_batch(self, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
        """Compute d acceptance / du for each sample."""
        acceptance = self._acceptance_proba(x_batch, u_arr)
        u_coef = self.u_coef
        probability_target = getattr(self.acceptance_model, "probability_target", "acceptance")
        if u_coef is None:
            coeffs = self._glm_churn_coefficients()
            if coeffs is not None:
                u_coef = float(coeffs["u_coef"])
                probability_target = coeffs.get("probability_target", probability_target)
        if u_coef is not None:
            sign = 1.0 if probability_target == "acceptance" else -1.0
            return sign * acceptance * (1.0 - acceptance) * u_coef
        eps = self._fd_eps
        a_plus = self._acceptance_proba(x_batch, u_arr + eps)
        a_minus = self._acceptance_proba(x_batch, u_arr - eps)
        return (a_plus - a_minus) / (2.0 * eps)

    def _mean_base_value(self, x_batch: Any, u_arr: np.ndarray) -> float:
        """Return the raw mean pricing objective for a fixed batch of actions."""
        return float(np.mean(self._value_batch(x_batch, u_arr)))

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

    def _lagrangian_adjustment(self, acceptance: np.ndarray) -> tuple[float, float]:
        """Return ``(lagrangian_value, d lagrangian / d mean_acceptance)``."""
        if self.acceptance_floor is None or self.lagrangian_lambda is None:
            return 0.0, 0.0
        mean_acceptance = float(np.mean(acceptance))
        lambda_value = float(self.lagrangian_lambda)
        value = lambda_value * (float(self.acceptance_floor) - mean_acceptance)
        return value, -lambda_value

    def _grad_u_batch(self, x_batch: Any, u_arr: np.ndarray) -> np.ndarray:
        """Compute df/du for each sample."""
        acceptance = self._acceptance_proba(x_batch, u_arr)
        loss = self._loss_prediction(x_batch)
        premium = self._premium_values(x_batch)
        revenue = (u_arr + 1.0) * premium

        d_acceptance_du = self._d_acceptance_du_batch(x_batch, u_arr)
        return d_acceptance_du * (loss - revenue) - acceptance * premium


__all__ = ["ModelBasedObjective"]
