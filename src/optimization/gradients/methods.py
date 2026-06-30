"""Gradient-method implementations for the class-based optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, cast

import numpy as np

from objective.utils import _policy_value, _theta_grad_from_u_grad
from optimization.helpers import (
    finite_difference_theta_grad,
    objective_grad_on_indices,
    objective_value_on_indices,
    x_batch,
)

if TYPE_CHECKING:
    from optimization.base import Optimization


# ---------------------------------------------------------------------------
# Shared u-space infrastructure
# ---------------------------------------------------------------------------


def _action_objective_values(objective: object, x_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
    """Compute per-sample action-level objective values ``M(x_i, u_i)``.

    Tries ``objective._value_batch(x_array, u_array)`` first, then falls back to
    calling ``objective.value_at_u(x_batch, u)`` per sample.
    """
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
        "U-space perturbation requires objective._value_batch(x_array, u_array) or "
        "objective.value_at_u(x_batch, u)."
    )


def _action_objective_values_many(objective: object, x_array: np.ndarray, u_matrix: np.ndarray) -> np.ndarray:
    """Compute action-level objective values for many action vectors."""
    u_arr = np.asarray(u_matrix, dtype=float)
    if u_arr.ndim != 2 or u_arr.shape[1] != x_array.shape[0]:
        raise ValueError("u_matrix must have shape (n_evaluations, n_samples).")

    value_batch_many_fn = getattr(objective, "_value_batch_many", None)
    if callable(value_batch_many_fn):
        values = np.asarray(value_batch_many_fn(x_array, u_arr), dtype=float)
        if values.shape != u_arr.shape:
            raise ValueError("objective._value_batch_many(x_array, u_matrix) must return u_matrix shape.")
        return values

    return np.vstack([_action_objective_values(objective, x_array, u_row) for u_row in u_arr])


def _u_space_policy_setup(
    optimizer: "Optimization",
    theta: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(x_arr, u_arr)`` for u-space gradient methods.

    Evaluates the policy at the current ``theta`` to get actions. The chain
    rule back to theta-space is handled later through a VJP-style policy hook.
    """
    policy = getattr(optimizer.objective, "policy", None)
    if policy is None or not callable(getattr(policy, "value", None)):
        raise ValueError("U-space perturbation requires objective.policy with value().")
    theta_arr = np.asarray(theta, dtype=float)
    x_arr = x_batch(optimizer.x_array, indices, optimizer.n_total)
    u_arr = _policy_value(optimizer.objective, theta_arr, x_arr).reshape(-1)
    return x_arr, u_arr


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class GradientMethod:
    """Base interface for theta-gradient estimators used by the optimizer."""

    name = "gradient"

    def setup(self, optimizer: "Optimization", theta0: np.ndarray) -> None:
        del optimizer, theta0

    def advance_rng(self, optimizer: "Optimization", theta: np.ndarray) -> None:
        del optimizer, theta

    def theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Estimator classes
# ---------------------------------------------------------------------------


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


class FiniteDifferenceGradient(GradientMethod):
    """Central finite-difference estimator.

    - **theta-space**: $$\\sum_{k=1}^d \\frac{J(\\theta+\\sigma e_k)-J(\\theta-\\sigma e_k)}{2\\sigma} e_k$$
      — ``2 * dim(theta)`` objective evaluations per step.
    - **u-space**: $$\\frac{M(x, u+\\sigma)-M(x, u-\\sigma)}{2\\sigma}$$ per sample, chain-ruled to theta
      — only 2 evaluations per step regardless of ``dim(theta)``.
    """

    name = "finite-difference"

    def setup(self, optimizer: "Optimization", theta0: np.ndarray) -> None:
        del theta0
        if optimizer.sigma <= 0.0:
            raise ValueError("sigma must be positive.")

    def theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        if optimizer.perturbation_space == "u":
            return self._u_grad(optimizer, theta, indices)
        return self._theta_grad(optimizer, theta, indices)

    def _theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        return finite_difference_theta_grad(
            lambda theta_eval: objective_value_on_indices(
                optimizer.objective,
                optimizer.x_array,
                optimizer.n_total,
                theta_eval,
                indices,
            ),
            theta,
            method="central",
            step=optimizer.sigma,
        )

    def _u_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        x_arr, u_arr = _u_space_policy_setup(optimizer, theta, indices)
        sigma = optimizer.sigma
        values = _action_objective_values_many(
            optimizer.objective,
            x_arr,
            np.vstack([u_arr + sigma, u_arr - sigma]),
        )
        values_plus = values[0]
        values_minus = values[1]
        grad_u = (values_plus - values_minus) / (2.0 * sigma)
        return _theta_grad_from_u_grad(optimizer.objective, theta, x_arr, grad_u)


class GaussSteinGradient(GradientMethod):
    """Gaussian Stein (score-function) estimator.

    - **theta-space**: $$\\hat{g} = \\frac{1}{m}\\sum_j J(\\theta+\\sigma\\varepsilon_j)\\varepsilon_j / \\sigma$$,
      $$\\varepsilon_j \\sim \\mathcal{N}(0, I^d)$$ — one-sided, ``n_grad_samples`` evaluations.
    - **u-space**: same estimator applied to actions, chain-ruled to theta.
    """

    name = "gauss-stein"

    def setup(self, optimizer: "Optimization", theta0: np.ndarray) -> None:
        del theta0
        if optimizer.n_grad_samples <= 0:
            raise ValueError("n_grad_samples must be positive.")
        if optimizer.sigma <= 0.0:
            raise ValueError("sigma must be positive.")

    def theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        if optimizer.perturbation_space == "u":
            return self._u_grad(optimizer, theta, indices)
        return self._theta_grad(optimizer, theta, indices)

    def advance_rng(self, optimizer: "Optimization", theta: np.ndarray) -> None:
        if optimizer.perturbation_space == "u":
            optimizer.rng.normal(0.0, 1.0, size=optimizer.n_grad_samples)
            return
        optimizer.rng.normal(0.0, 1.0, size=(optimizer.n_grad_samples, theta.size))

    def _theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        eps_samples = optimizer.rng.normal(
            0.0, 1.0, size=(optimizer.n_grad_samples, theta.size)
        ).astype(float)
        accum = np.zeros_like(theta, dtype=float)
        for eps in eps_samples:
            value = objective_value_on_indices(
                optimizer.objective,
                optimizer.x_array,
                optimizer.n_total,
                theta + optimizer.sigma * eps,
                indices,
            )
            accum += value * eps
        return accum / float(eps_samples.shape[0]) / max(optimizer.sigma, 1e-8)

    def _u_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        x_arr, u_arr = _u_space_policy_setup(optimizer, theta, indices)
        w_samples = optimizer.rng.normal(0.0, 1.0, size=optimizer.n_grad_samples).astype(float)
        values = _action_objective_values_many(
            optimizer.objective,
            x_arr,
            u_arr[None, :] + optimizer.sigma * w_samples[:, None],
        )
        grad_u = np.mean(values * w_samples[:, None], axis=0) / max(optimizer.sigma, 1e-8)
        return _theta_grad_from_u_grad(optimizer.objective, theta, x_arr, grad_u)


class SPSAGradient(GradientMethod):
    """SPSA estimator using Rademacher perturbations.

    - **theta-space**: $$\\hat{g} = \\frac{1}{m}\\sum_j \\frac{J(\\theta+\\sigma\\Delta_j)-J(\\theta-\\sigma\\Delta_j)}{2\\sigma}\\Delta_j$$,
      $$\\Delta_j \\sim \\{\\pm 1\\}^d$$ — two-sided, ``2 * n_grad_samples`` evaluations.
    - **u-space**: same estimator applied to actions, chain-ruled to theta.
    """

    name = "spsa"

    def setup(self, optimizer: "Optimization", theta0: np.ndarray) -> None:
        del theta0
        if optimizer.n_grad_samples <= 0:
            raise ValueError("n_grad_samples must be positive.")
        if optimizer.sigma <= 0.0:
            raise ValueError("sigma must be positive.")

    def theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        if optimizer.perturbation_space == "u":
            return self._u_grad(optimizer, theta, indices)
        return self._theta_grad(optimizer, theta, indices)

    def advance_rng(self, optimizer: "Optimization", theta: np.ndarray) -> None:
        choices = np.asarray([-1.0, 1.0], dtype=float)
        if optimizer.perturbation_space == "u":
            optimizer.rng.choice(choices, size=optimizer.n_grad_samples)
            return
        optimizer.rng.choice(choices, size=(optimizer.n_grad_samples, theta.size))

    def _theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        delta_samples = optimizer.rng.choice(
            np.asarray([-1.0, 1.0], dtype=float),
            size=(optimizer.n_grad_samples, theta.size),
        )
        grad = np.zeros_like(theta, dtype=float)
        for delta in delta_samples:
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
            grad += ((value_plus - value_minus) / (2.0 * optimizer.sigma)) * delta
        return grad / float(delta_samples.shape[0])

    def _u_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        x_arr, u_arr = _u_space_policy_setup(optimizer, theta, indices)
        delta_samples = optimizer.rng.choice(
            np.asarray([-1.0, 1.0], dtype=float), size=optimizer.n_grad_samples
        )
        values_plus = _action_objective_values_many(
            optimizer.objective,
            x_arr,
            u_arr[None, :] + optimizer.sigma * delta_samples[:, None],
        )
        values_minus = _action_objective_values_many(
            optimizer.objective,
            x_arr,
            u_arr[None, :] - optimizer.sigma * delta_samples[:, None],
        )
        grad_u = np.mean(
            ((values_plus - values_minus) / (2.0 * optimizer.sigma)) * delta_samples[:, None],
            axis=0,
        )
        return _theta_grad_from_u_grad(optimizer.objective, theta, x_arr, grad_u)


class SteinDifferenceGradient(GradientMethod):
    """Stein-difference estimator using two-sided Gaussian perturbations.

    - **theta-space**: $$\\hat{g} = \\frac{1}{m}\\sum_j \\frac{J(\\theta+\\sigma\\varepsilon_j)-J(\\theta-\\sigma\\varepsilon_j)}{2\\sigma}\\varepsilon_j$$,
      $$\\varepsilon_j \\sim \\mathcal{N}(0, I^d)$$ — two-sided Gaussian in theta, ``2 * n_grad_samples`` evaluations.
    - **u-space**: $$\\hat{g}_{u,i} = \\frac{1}{m}\\sum_j \\frac{M(x_i,u_i+\\sigma w_j)-M(x_i,u_i-\\sigma w_j)}{2\\sigma} w_j$$,
      $$w_j \\sim \\mathcal{N}(0,1)$$, chain-ruled to theta via $$\\nabla_\\theta \\pi_\\theta(x_i)$$.
    """

    name = "stein-difference"

    def setup(self, optimizer: "Optimization", theta0: np.ndarray) -> None:
        del theta0
        if optimizer.n_grad_samples <= 0:
            raise ValueError("n_grad_samples must be positive.")
        if optimizer.sigma <= 0.0:
            raise ValueError("sigma must be positive.")

    def theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        if optimizer.perturbation_space == "u":
            return self._u_grad(optimizer, theta, indices)
        return self._theta_grad(optimizer, theta, indices)

    def advance_rng(self, optimizer: "Optimization", theta: np.ndarray) -> None:
        if optimizer.perturbation_space == "u":
            optimizer.rng.normal(0.0, 1.0, size=optimizer.n_grad_samples)
            return
        optimizer.rng.normal(0.0, 1.0, size=(optimizer.n_grad_samples, theta.size))

    def _theta_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        eps_samples = optimizer.rng.normal(
            0.0, 1.0, size=(optimizer.n_grad_samples, theta.size)
        ).astype(float)
        grad = np.zeros_like(theta, dtype=float)
        for eps in eps_samples:
            value_plus = objective_value_on_indices(
                optimizer.objective, optimizer.x_array, optimizer.n_total,
                theta + optimizer.sigma * eps, indices,
            )
            value_minus = objective_value_on_indices(
                optimizer.objective, optimizer.x_array, optimizer.n_total,
                theta - optimizer.sigma * eps, indices,
            )
            grad += ((value_plus - value_minus) / (2.0 * optimizer.sigma)) * eps
        return grad / float(eps_samples.shape[0])

    def _u_grad(
        self,
        optimizer: "Optimization",
        theta: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        x_arr, u_arr = _u_space_policy_setup(optimizer, theta, indices)
        sigma = optimizer.sigma
        w_samples = optimizer.rng.normal(0.0, 1.0, size=optimizer.n_grad_samples).astype(float)
        values_plus = _action_objective_values_many(
            optimizer.objective,
            x_arr,
            u_arr[None, :] + sigma * w_samples[:, None],
        )
        values_minus = _action_objective_values_many(
            optimizer.objective,
            x_arr,
            u_arr[None, :] - sigma * w_samples[:, None],
        )
        grad_u = np.mean(((values_plus - values_minus) / (2.0 * sigma)) * w_samples[:, None], axis=0)
        return _theta_grad_from_u_grad(optimizer.objective, theta, x_arr, grad_u)


__all__ = [
    "GradientMethod",
    "FirstOrderGradient",
    "FiniteDifferenceGradient",
    "GaussSteinGradient",
    "SPSAGradient",
    "SteinDifferenceGradient",
]
