"""Fixed regression objective for pricing simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from objective.base import Objective, Policy, StateVector
from objective.utils import theta_grad_from_u_grad


def _logistic(z: float) -> float:
    """Numerically stable scalar sigmoid."""
    if z >= 0.0:
        exp_neg = float(np.exp(-z))
        return float(1.0 / (1.0 + exp_neg))
    exp_pos = float(np.exp(z))
    return float(exp_pos / (1.0 + exp_pos))


def _logistic_batch(z: np.ndarray) -> np.ndarray:
    """Numerically stable vectorized sigmoid."""
    z_arr = np.asarray(z, dtype=float)
    out = np.empty_like(z_arr, dtype=float)
    positive = z_arr >= 0.0
    exp_neg = np.exp(-z_arr[positive])
    out[positive] = 1.0 / (1.0 + exp_neg)
    exp_pos = np.exp(z_arr[~positive])
    out[~positive] = exp_pos / (1.0 + exp_pos)
    return out


@dataclass(frozen=True)
class FixedRegressionObjective(Objective):
    """Pricing objective: $f(u; x) = a(x,u)(\\ell(x) - r(u))$.

    Components: $a = \\sigma(\\beta_1^\\top x + \\beta_2 u)$, $\\ell = \\beta_3^\\top x$, $r = \\beta_4 u$.
    Computes theta-gradients via chain rule through the attached policy.
    """

    policy: Policy
    beta_1: np.ndarray
    beta_2: float
    beta_3: np.ndarray
    beta_4: float

    def __post_init__(self) -> None:
        beta_1 = np.asarray(self.beta_1, dtype=float)
        beta_2 = float(self.beta_2)
        beta_3 = np.asarray(self.beta_3, dtype=float)
        beta_4 = float(self.beta_4)
        if np.any(beta_1 <= 0.0):
            raise ValueError("beta_1 entries must be positive.")
        if beta_2 >= 0.0:
            raise ValueError(
                "beta_2 must be negative; acceptance probability should decrease as policy value increases."
            )
        if np.any(beta_3 <= 0.0):
            raise ValueError("beta_3 entries must be positive.")
        if beta_4 <= 0.0:
            raise ValueError("beta_4 must be positive.")
        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "beta_2", beta_2)
        object.__setattr__(self, "beta_3", beta_3)
        object.__setattr__(self, "beta_4", beta_4)

    @classmethod
    def from_parameters(
        cls,
        policy: Policy,
        beta_1: np.ndarray | Sequence[float],
        beta_2: float,
        beta_3: np.ndarray | Sequence[float],
        beta_4: float,
    ) -> "FixedRegressionObjective":
        """Create objective from parameter values."""
        return cls(
            policy=policy,
            beta_1=np.asarray(beta_1, dtype=float),
            beta_2=float(beta_2),
            beta_3=np.asarray(beta_3, dtype=float),
            beta_4=float(beta_4),
        )

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        """Compute mean objective value across batch."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be a 2D array.")
        u_batch = self.policy.value_batch(theta, x_arr)
        values = self._value_batch(x_arr, u_batch)
        return float(np.mean(values))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Compute theta-gradient via chain rule: df/dtheta = df/du * du/dtheta."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be a 2D array.")
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self.policy.value_batch(theta_arr, x_arr)
        grad_u = self._grad_u_batch(x_arr, u_batch)
        return theta_grad_from_u_grad(self.policy, theta_arr, x_arr, grad_u)

    def value_at_u(self, x_batch: np.ndarray, u: float) -> float:
        """Compute mean objective value at a fixed action u."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be a 2D array.")
        u_arr = np.full(x_arr.shape[0], float(u), dtype=float)
        values = self._value_batch(x_arr, u_arr)
        return float(np.mean(values))

    def _value_batch(self, x_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
        """Compute objective values for batch of (x, u) pairs."""
        beta_1_x = x_array @ self.beta_1[: x_array.shape[1]]
        beta_3_x = x_array @ self.beta_3[: x_array.shape[1]]
        logits = beta_1_x + self.beta_2 * u_array
        acceptance = _logistic_batch(logits)
        revenue = self.beta_4 * u_array
        return acceptance * (beta_3_x - revenue)

    def _grad_u_batch(self, x_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
        """Compute gradient w.r.t. u for batch of (x, u) pairs."""
        beta_1_x = x_array @ self.beta_1[: x_array.shape[1]]
        beta_3_x = x_array @ self.beta_3[: x_array.shape[1]]
        logits = beta_1_x + self.beta_2 * u_array
        acceptance = _logistic_batch(logits)
        d_acceptance_du = acceptance * (1.0 - acceptance) * self.beta_2
        revenue = self.beta_4 * u_array
        return d_acceptance_du * (beta_3_x - revenue) - acceptance * self.beta_4

    # Scalar methods for single-sample evaluation (used in some tests/visualizations)
    def value_scalar(self, x: StateVector, u: float) -> float:
        """Compute objective value for a single (x, u) pair."""
        x_arr = np.asarray(x, dtype=float)
        logit = float(np.dot(self.beta_1[: x_arr.size], x_arr)) + self.beta_2 * u
        acceptance = _logistic(logit)
        loss = float(np.dot(self.beta_3[: x_arr.size], x_arr))
        revenue = self.beta_4 * u
        return float(acceptance * (loss - revenue))

    def grad_u_scalar(self, x: StateVector, u: float) -> float:
        """Compute gradient w.r.t. u for a single (x, u) pair."""
        x_arr = np.asarray(x, dtype=float)
        logit = float(np.dot(self.beta_1[: x_arr.size], x_arr)) + self.beta_2 * u
        acceptance = _logistic(logit)
        d_acceptance_du = acceptance * (1.0 - acceptance) * self.beta_2
        loss = float(np.dot(self.beta_3[: x_arr.size], x_arr))
        revenue = self.beta_4 * u
        return float(d_acceptance_du * (loss - revenue) - acceptance * self.beta_4)


__all__ = ["FixedRegressionObjective"]
