"""Gradient-method implementations for the class-based optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from optimization.base import Optimization


class GradientMethod:
    """Base interface for theta-gradient estimators."""

    name = "gradient"

    def setup(self, optimizer: "Optimization", theta0: np.ndarray) -> None:
        del optimizer, theta0

    def theta_and_u_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        raise NotImplementedError


class FirstOrderGradient(GradientMethod):
    """Exact objective u-gradient chained through the policy."""

    name = "first-order"

    def theta_and_u_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        _, x_batch, phi_batch_values, _, grad_batch_fn = optimizer.batch_context(indices)
        u_vals = optimizer.policy_u_batch(theta, x_batch, phi_batch_values)
        grad_u_vals = grad_batch_fn(u_vals)
        grad_theta = optimizer.theta_grad_from_u_grad(theta, phi_batch_values, u_vals, grad_u_vals)
        return grad_theta, grad_u_vals


class GaussSteinGradient(GradientMethod):
    """Value-only Stein estimator for u-gradients chained to theta."""

    name = "gauss-stein"

    def __init__(self) -> None:
        self._eps_base: np.ndarray | None = None

    def setup(self, optimizer: "Optimization", theta0: np.ndarray) -> None:
        del theta0
        if optimizer.n_grad_samples <= 0:
            raise ValueError("n_grad_samples must be positive.")
        if optimizer.sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        self._eps_base = optimizer.rng.normal(
            0.0,
            1.0,
            size=(optimizer.n_grad_samples, optimizer.n_total),
        ).astype(float)

    def theta_and_u_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self._eps_base is None:
            raise ValueError("GaussSteinGradient.setup must be called before solve.")
        _, x_batch, phi_batch_values, value_batch_fn, _ = optimizer.batch_context(indices)
        u_vals = optimizer.policy_u_batch(theta, x_batch, phi_batch_values)
        eps_values = self._eps_base[:, indices]
        accum = np.zeros_like(u_vals, dtype=float)
        for eps in eps_values:
            values = value_batch_fn(u_vals + optimizer.sigma * eps)
            accum += values * eps
        grad_u_vals = accum / float(eps_values.shape[0]) / max(optimizer.sigma, 1e-8)
        grad_theta = optimizer.theta_grad_from_u_grad(theta, phi_batch_values, u_vals, grad_u_vals)
        return grad_theta, grad_u_vals


class SPSAGradient(GradientMethod):
    """Two-sided SPSA estimator directly in theta-space."""

    name = "spsa"

    def __init__(self) -> None:
        self._delta_base: np.ndarray | None = None

    def setup(self, optimizer: "Optimization", theta0: np.ndarray) -> None:
        if optimizer.n_grad_samples <= 0:
            raise ValueError("n_grad_samples must be positive.")
        if optimizer.sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        self._delta_base = optimizer.rng.choice(
            np.asarray([-1.0, 1.0], dtype=float),
            size=(optimizer.n_grad_samples, theta0.size),
        )

    def theta_and_u_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self._delta_base is None:
            raise ValueError("SPSAGradient.setup must be called before solve.")
        grad_theta = np.zeros_like(theta, dtype=float)
        for delta in self._delta_base:
            value_plus = optimizer.objective_on_indices(theta + optimizer.sigma * delta, indices)
            value_minus = optimizer.objective_on_indices(theta - optimizer.sigma * delta, indices)
            grad_theta += ((value_plus - value_minus) / (2.0 * optimizer.sigma)) * delta
        return grad_theta / float(self._delta_base.shape[0]), None


__all__ = [
    "GradientMethod",
    "FirstOrderGradient",
    "GaussSteinGradient",
    "SPSAGradient",
]
