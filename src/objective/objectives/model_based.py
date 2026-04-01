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
    """Pricing objective backed by trained ML models.

    $$f(u; x) = a(x,u) \\cdot (\\hat{Y}(x) - u \\cdot p(x))$$

    where $$a(x,u)$$ is the acceptance probability (sklearn Pipeline or XGBClassifier),
    $$\\hat{Y}(x)$$ is the expected financial loss (LinearRegression or XGBRegressor),
    and $$p(x)$$ is the policy premium extracted from column ``premium_col`` of x.

    ``acceptance_model`` expects a DataFrame with ``acceptance_state_cols + ["U"]``.
    ``loss_model`` expects a DataFrame with ``loss_cols``.

    If ``u_coef`` is provided, the analytical gradient
    $$da/dU = a(1-a) \\cdot u_{coef}$$ is used (GLM path).
    Otherwise numerical central finite differences are used (XGBoost path).
    """

    policy: Policy
    acceptance_model: Any
    loss_model: Any
    # Column names for model inference DataFrames
    acceptance_state_cols: tuple[str, ...]  # 10 state cols passed to acceptance model (no U)
    loss_cols: tuple[str, ...]              # 9 cols passed to loss model
    premium_col: int = 9                    # index of X_policy_premium in x_batch
    u_coef: float | None = None             # w_U / std_U for analytical GLM gradient
    _fd_eps: float = 1e-4                   # step size for numerical d_acceptance/dU

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        """Compute mean objective value across batch."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be 2D.")
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self.policy.value(theta_arr, x_arr)
        return float(np.mean(self._value_batch(x_arr, u_batch)))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Compute theta-gradient via chain rule: df/dtheta = mean(df/du * du/dtheta)."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be 2D.")
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self.policy.value(theta_arr, x_arr)
        grad_u = self._grad_u_batch(x_arr, u_batch)
        return _theta_grad_from_u_grad(self.policy, theta_arr, x_arr, grad_u)

    def value_at_u(self, x_batch: np.ndarray, u: float) -> float:
        """Compute mean objective value at a fixed action u."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be 2D.")
        u_arr = np.full(x_arr.shape[0], float(u), dtype=float)
        return float(np.mean(self._value_batch(x_arr, u_arr)))

    # --- Private helpers ---

    def _acceptance_proba(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        """Call acceptance model on (x_batch state cols + u_arr). Returns shape (n,)."""
        x_state = x_batch[:, : len(self.acceptance_state_cols)]
        df = pd.DataFrame(
            np.column_stack([x_state, u_arr]),
            columns=list(self.acceptance_state_cols) + ["U"],
        )
        return np.asarray(self.acceptance_model.predict_proba(df)[:, 1], dtype=float)

    def _loss_prediction(self, x_batch: np.ndarray) -> np.ndarray:
        """Call loss model on loss_cols subset of x_batch. Returns shape (n,)."""
        x_loss = x_batch[:, : len(self.loss_cols)]
        df = pd.DataFrame(x_loss, columns=list(self.loss_cols))
        return np.asarray(self.loss_model.predict(df), dtype=float)

    def _value_batch(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        """Compute per-sample objective values."""
        acceptance = self._acceptance_proba(x_batch, u_arr)
        loss = self._loss_prediction(x_batch)
        premium = x_batch[:, self.premium_col]
        revenue = u_arr * premium
        return acceptance * (loss - revenue)

    def _grad_u_batch(self, x_batch: np.ndarray, u_arr: np.ndarray) -> np.ndarray:
        """Compute df/du for each sample.

        GLM path (u_coef set): analytical d_acceptance/du = a(1-a) * u_coef.
        XGBoost path (u_coef is None): central FD on acceptance model.
        """
        acceptance = self._acceptance_proba(x_batch, u_arr)
        loss = self._loss_prediction(x_batch)
        premium = x_batch[:, self.premium_col]
        revenue = u_arr * premium

        if self.u_coef is not None:
            # Analytical: d_acceptance/dU = a(1-a) * u_coef
            d_acceptance_du = acceptance * (1.0 - acceptance) * self.u_coef
        else:
            # Numerical central FD: d_acceptance/dU ≈ (a(u+ε) - a(u-ε)) / (2ε)
            eps = self._fd_eps
            a_plus = self._acceptance_proba(x_batch, u_arr + eps)
            a_minus = self._acceptance_proba(x_batch, u_arr - eps)
            d_acceptance_du = (a_plus - a_minus) / (2.0 * eps)

        # df/du = d_acceptance/du * (loss - revenue) - acceptance * premium
        return d_acceptance_du * (loss - revenue) - acceptance * premium


__all__ = ["ModelBasedObjective"]
