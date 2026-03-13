"""Gradient-method implementations for the class-based optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optimization.helpers import objective_grad_on_indices, objective_value_on_indices

if TYPE_CHECKING:
    from optimization.base import Optimization


class GradientMethod:
    """Base interface for theta-gradient estimators."""

    name = "gradient"

    def setup(self, optimizer: "Optimization", theta0: np.ndarray) -> None:
        del optimizer, theta0

    def theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


class FirstOrderGradient(GradientMethod):
    """Exact theta-gradient from objective.grad."""

    name = "first-order"

    def theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        return objective_grad_on_indices(
            optimizer.objective,
            optimizer.x_array,
            optimizer.n_total,
            theta,
            indices,
        )


class GaussSteinGradient(GradientMethod):
    """Theta-space Gaussian-Stein estimator using value-only queries."""

    name = "gauss-stein"

    def __init__(self) -> None:
        self._eps_base: np.ndarray | None = None

    def setup(self, optimizer: "Optimization", theta0: np.ndarray) -> None:
        if optimizer.n_grad_samples <= 0:
            raise ValueError("n_grad_samples must be positive.")
        if optimizer.sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        self._eps_base = optimizer.rng.normal(
            0.0,
            1.0,
            size=(optimizer.n_grad_samples, theta0.size),
        ).astype(float)

    def theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        if self._eps_base is None:
            raise ValueError("GaussSteinGradient.setup must be called before solve.")
        accum = np.zeros_like(theta, dtype=float)
        for eps in self._eps_base:
            value = objective_value_on_indices(
                optimizer.objective,
                optimizer.x_array,
                optimizer.n_total,
                theta + optimizer.sigma * eps,
                indices,
            )
            accum += value * eps
        return accum / float(self._eps_base.shape[0]) / max(optimizer.sigma, 1e-8)


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

    def theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        if self._delta_base is None:
            raise ValueError("SPSAGradient.setup must be called before solve.")
        grad_theta = np.zeros_like(theta, dtype=float)
        for delta in self._delta_base:
            value_plus = objective_value_on_indices(
                optimizer.objective,
                optimizer.x_array,
                optimizer.n_total,
                theta + optimizer.sigma * delta,
                indices,
            )
            value_minus = objective_value_on_indices(
                optimizer.objective,
                optimizer.x_array,
                optimizer.n_total,
                theta - optimizer.sigma * delta,
                indices,
            )
            grad_theta += ((value_plus - value_minus) / (2.0 * optimizer.sigma)) * delta
        return grad_theta / float(self._delta_base.shape[0])


__all__ = [
    "GradientMethod",
    "FirstOrderGradient",
    "GaussSteinGradient",
    "SPSAGradient",
]
