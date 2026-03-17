"""Planted convex logistic objective with known optimum."""

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
class PlantedLogisticObjective(Objective):
    """Convex logistic objective with known optimum $u^*$ for algorithm validation.

    $L(u; x) = \\log(1 + e^z) - p^*(x) z$ where $z = \\alpha u + \\beta^\\top x + b$.
    """

    policy: Policy
    alpha: float
    beta: np.ndarray
    bias: float
    u_star: float

    def __post_init__(self) -> None:
        alpha = float(self.alpha)
        beta = np.asarray(self.beta, dtype=float)
        bias = float(self.bias)
        u_star = float(self.u_star)
        if alpha == 0.0:
            raise ValueError("alpha must be nonzero for a unique optimum.")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "u_star", u_star)

    @classmethod
    def from_parameters(
        cls,
        policy: Policy,
        alpha: float,
        beta: np.ndarray | Sequence[float],
        bias: float,
        u_star: float,
    ) -> "PlantedLogisticObjective":
        """Create objective from parameter values."""
        return cls(
            policy=policy,
            alpha=float(alpha),
            beta=np.asarray(beta, dtype=float),
            bias=float(bias),
            u_star=float(u_star),
        )

    def optimal_u(self) -> float:
        """Return the planted optimal action value."""
        return float(self.u_star)

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
        beta_x = x_array @ self.beta[: x_array.shape[1]]
        z = self.alpha * u_array + beta_x + self.bias
        z_star = self.alpha * self.u_star + beta_x + self.bias
        p_star = _logistic_batch(z_star)
        return np.logaddexp(0.0, z) - p_star * z

    def _grad_u_batch(self, x_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
        """Compute gradient w.r.t. u for batch of (x, u) pairs."""
        beta_x = x_array @ self.beta[: x_array.shape[1]]
        z = self.alpha * u_array + beta_x + self.bias
        z_star = self.alpha * self.u_star + beta_x + self.bias
        p = _logistic_batch(z)
        p_star = _logistic_batch(z_star)
        return self.alpha * (p - p_star)

    # Scalar methods for single-sample evaluation (used in some tests/visualizations)
    def value_scalar(self, x: StateVector, u: float) -> float:
        """Compute objective value for a single (x, u) pair."""
        x_arr = np.asarray(x, dtype=float)
        z = self.alpha * u + float(np.dot(self.beta[: x_arr.size], x_arr)) + self.bias
        p_star = _logistic(self.alpha * self.u_star + float(np.dot(self.beta[: x_arr.size], x_arr)) + self.bias)
        return float(np.logaddexp(0.0, z) - p_star * z)

    def grad_u_scalar(self, x: StateVector, u: float) -> float:
        """Compute gradient w.r.t. u for a single (x, u) pair."""
        x_arr = np.asarray(x, dtype=float)
        z = self.alpha * u + float(np.dot(self.beta[: x_arr.size], x_arr)) + self.bias
        z_star = self.alpha * self.u_star + float(np.dot(self.beta[: x_arr.size], x_arr)) + self.bias
        p = _logistic(z)
        p_star = _logistic(z_star)
        return float(self.alpha * (p - p_star))


__all__ = ["PlantedLogisticObjective"]
