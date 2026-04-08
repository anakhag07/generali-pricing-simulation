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

    $$f(u; x) = a(x,u) \cdot (\hat{Y}(x) - u \cdot p(x))$$

    where $$a(x,u)$$ is acceptance derived from the bundled churn classifier,
    $$\hat{Y}(x)$$ is the expected financial loss (LinearRegression or XGBRegressor),
    and $$p(x)$$ is the policy premium extracted from column ``premium_col`` of x.

    ``acceptance_model`` expects a DataFrame with ``acceptance_state_cols + ["U"]`` and
    returns churn probability in class 1, which this objective maps to acceptance via
    ``1 - p_churn``. ``loss_model`` expects a DataFrame with ``loss_cols``.

    If ``u_coef`` is provided, it is interpreted as $$d\,\text{logit}(p_{churn}) / dU$$,
    so the analytical acceptance gradient uses the opposite sign.
    Otherwise numerical central finite differences are used (XGBoost path).
    """

    policy: Policy
    acceptance_model: Any
    loss_model: Any
    acceptance_state_cols: tuple[str, ...]
    loss_cols: tuple[str, ...]
    premium_col: int = 9
    u_coef: float | None = None
    _fd_eps: float = 1e-4

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        """Compute mean objective value across batch."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be 2D.")
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self.policy_value(theta_arr, x_arr)
        return float(np.mean(self._value_batch(x_arr, u_batch)))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Compute theta-gradient via chain rule: df/dtheta = mean(df/du * du/dtheta)."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be 2D.")
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self.policy_value(theta_arr, x_arr)
        grad_u = self._grad_u_batch(x_arr, u_batch)
        return _theta_grad_from_u_grad(self, theta_arr, x_arr, grad_u)

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
        u_arr = np.full(x_arr.shape[0], float(u), dtype=float)
        return float(np.mean(self._value_batch(x_arr, u_arr)))

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
        revenue = u_arr * premium
        return acceptance * (loss - revenue)

    def _grad_u_batch(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        """Compute df/du for each sample."""
        acceptance = self._acceptance_proba(x_batch, u_arr)
        loss = self._loss_prediction(x_batch)
        premium = x_batch[:, self.premium_col]
        revenue = u_arr * premium

        if self.u_coef is not None:
            d_acceptance_du = -acceptance * (1.0 - acceptance) * self.u_coef
        else:
            eps = self._fd_eps
            a_plus = self._acceptance_proba(x_batch, u_arr + eps)
            a_minus = self._acceptance_proba(x_batch, u_arr - eps)
            d_acceptance_du = (a_plus - a_minus) / (2.0 * eps)

        return d_acceptance_du * (loss - revenue) - acceptance * premium


__all__ = ["ModelBasedObjective"]
