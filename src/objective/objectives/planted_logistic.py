"""Planted convex logistic objective with known optimum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from objective._math import _sigmoid
from objective.base import Objective, Policy
from objective.utils import _theta_grad_from_u_grad


@dataclass(frozen=True)
class PlantedLogisticObjective(Objective):
    """Convex logistic objective with known optimum $$u^*$$ for algorithm validation.

    $$L(u; x) = \\log(1 + e^z) - p^*(x) z$$ where $$z = \\alpha u + \\beta^\\top x + b$$.
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
        u_batch = self.policy.value(theta, x_arr)
        values = self._value_batch(x_arr, u_batch)
        return float(np.mean(values))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        """Compute theta-gradient via chain rule: df/dtheta = df/du * du/dtheta."""
        x_arr = np.asarray(x_batch, dtype=float)
        if x_arr.ndim != 2:
            raise ValueError("x_batch must be a 2D array.")
        theta_arr = np.asarray(theta, dtype=float)
        u_batch = self.policy.value(theta_arr, x_arr)
        grad_u = self._grad_u_batch(x_arr, u_batch)
        return _theta_grad_from_u_grad(self.policy, theta_arr, x_arr, grad_u)

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
        p_star = _sigmoid(z_star)
        return np.logaddexp(0.0, z) - p_star * z

    def _grad_u_batch(self, x_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
        """Compute gradient w.r.t. u for batch of (x, u) pairs."""
        beta_x = x_array @ self.beta[: x_array.shape[1]]
        z = self.alpha * u_array + beta_x + self.bias
        z_star = self.alpha * self.u_star + beta_x + self.bias
        p = _sigmoid(z)
        p_star = _sigmoid(z_star)
        return self.alpha * (p - p_star)


__all__ = ["PlantedLogisticObjective"]
