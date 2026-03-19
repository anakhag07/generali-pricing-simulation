"""Gradient-method implementations for the class-based optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

import numpy as np

from optimization.helpers import (
    objective_grad_on_indices,
    objective_value_on_indices,
    x_batch,
)

if TYPE_CHECKING:
    from optimization.base import Optimization


class GradientMethod:
    """Base interface for theta-gradient estimators used by the optimizer."""

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
    """Exact theta-gradient: $$\\nabla_\\theta J$$ from ``objective.grad``."""

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
    """Stein estimator: $$\\hat{g} = \\mathbb{E}[J(\\theta + \\sigma\\varepsilon)\\varepsilon]/\\sigma$$."""

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
    """SPSA estimator: $$\\hat{g} = (J(\\theta+\\sigma\\Delta) - J(\\theta-\\sigma\\Delta))\\Delta / 2\\sigma$$."""

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


def _action_objective_values(objective: object, x_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
    """Compute per-sample action-level objective values ``M(x_i, u_i)``."""
    u_arr = np.asarray(u_array, dtype=float).reshape(-1)
    if u_arr.shape != (x_array.shape[0],):
        raise ValueError("u_array must have shape (n_samples,).")

    value_batch_fn = getattr(objective, "_value_batch", None)
    if callable(value_batch_fn):
        values = np.asarray(value_batch_fn(x_array, u_arr), dtype=float)
        if values.shape != (x_array.shape[0],):
            raise ValueError("objective._value_batch(x_array, u_array) must return shape (n_samples,).")
        return values

    value_at_u_fn = getattr(objective, "value_at_u", None)
    if callable(value_at_u_fn):
        value_at_u_typed = cast(Callable[[np.ndarray, float], float], value_at_u_fn)
        values = np.empty(x_array.shape[0], dtype=float)
        for idx, u_val in enumerate(u_arr):
            values[idx] = float(value_at_u_typed(x_array[idx : idx + 1], float(u_val)))
        return values

    raise ValueError(
        "SteinDifferenceGradient requires objective._value_batch(x_array, u_array) or "
        "objective.value_at_u(x_batch, u)."
    )


class SteinDifferenceGradient(GradientMethod):
    """Stein-SPSA estimator in action-space mapped to theta by chain rule.

    $$\\hat{g}_{u,i} = \\frac{1}{m}\\sum_{j=1}^m \\frac{M(x_i, u_i+\\sigma w_j)-M(x_i, u_i-\\sigma w_j)}{2\\sigma} w_j$$
    and
    $$\\hat{g}_\\theta = \\frac{1}{n}\\sum_{i=1}^n \\hat{g}_{u,i} \\, \\nabla_\\theta \\pi_\\theta(x_i).$$
    """

    name = "stein-difference"

    def __init__(self) -> None:
        self._w_base: np.ndarray | None = None

    def setup(self, optimizer: "Optimization", theta0: np.ndarray) -> None:
        del theta0
        if optimizer.n_grad_samples <= 0:
            raise ValueError("n_grad_samples must be positive.")
        if optimizer.sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        self._w_base = optimizer.rng.normal(0.0, 1.0, size=optimizer.n_grad_samples).astype(float)

    def theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        if self._w_base is None:
            raise ValueError("SteinDifferenceGradient.setup must be called before solve.")

        policy = getattr(optimizer.objective, "policy", None)
        policy_value = getattr(policy, "value", None)
        policy_grad = getattr(policy, "grad", None)
        if not callable(policy_value) or not callable(policy_grad):
            raise ValueError("SteinDifferenceGradient requires objective.policy with value(...) and grad(...).")

        theta_arr = np.asarray(theta, dtype=float)
        x_arr = x_batch(optimizer.x_array, indices, optimizer.n_total)
        u_arr = np.asarray(policy_value(theta_arr, x_arr), dtype=float).reshape(-1)
        if u_arr.shape != (x_arr.shape[0],):
            raise ValueError("policy.value(theta, x_batch) must return shape (n_samples,).")

        grad_pi = np.asarray(policy_grad(theta_arr, x_arr), dtype=float)
        if grad_pi.shape != (x_arr.shape[0], theta_arr.size):
            raise ValueError("policy.grad(theta, x_batch) must return shape (n_samples, theta_dim).")

        sigma = optimizer.sigma
        grad_u = np.zeros(x_arr.shape[0], dtype=float)
        for w_j in self._w_base:
            values_plus = _action_objective_values(optimizer.objective, x_arr, u_arr + sigma * w_j)
            values_minus = _action_objective_values(optimizer.objective, x_arr, u_arr - sigma * w_j)
            grad_u += ((values_plus - values_minus) / (2.0 * sigma)) * w_j
        grad_u /= float(self._w_base.shape[0])

        return np.mean(grad_u[:, None] * grad_pi, axis=0)


__all__ = [
    "GradientMethod",
    "FirstOrderGradient",
    "GaussSteinGradient",
    "SPSAGradient",
    "SteinDifferenceGradient",
]
