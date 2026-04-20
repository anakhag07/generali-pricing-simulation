"""Isolated math tests for gradient estimators on a known quadratic objective.

Uses f(u; x) = (u - c)^2 with LinearPolicy u = theta^T phi(x) so that the
analytical theta-gradient is computable. Verifies that stochastic estimators
converge to the true gradient with sufficient samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from objective._math import _sigmoid
from objective.base import Objective, Policy, sample_states
from objective.policy import LinearPolicy
from objective.utils import _theta_grad_from_u_grad


# ---------------------------------------------------------------------------
# Tiny quadratic objective for testing: f(u; x) = (u - c)^2
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _QuadraticObjective(Objective):
    """Test objective: f(u; x) = (u - c)^2 with known gradient 2(u - c)."""
    policy: Policy
    c: float = 0.0

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        u = self.policy.value(theta, x_batch)
        return float(np.mean((u - self.c) ** 2))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        u = self.policy.value(theta, x_batch)
        grad_u = 2.0 * (u - self.c)
        return _theta_grad_from_u_grad(self.policy, theta, x_batch, grad_u)

    def value_at_u(self, x_batch: np.ndarray, u: float) -> float:
        return float((u - self.c) ** 2)

    def _value_batch(self, x_array: np.ndarray, u_array: np.ndarray) -> np.ndarray:
        return (u_array - self.c) ** 2


def _setup(state_dim=2, n_samples=200, seed=42):
    """Create a small quadratic problem with LinearPolicy."""
    rng = np.random.default_rng(seed)
    policy = LinearPolicy()
    obj = _QuadraticObjective(policy=policy, c=0.3)
    theta = rng.normal(size=state_dim + 1) * 0.1
    x_batch = sample_states(rng, n_samples, state_dim)
    true_grad = obj.grad(theta, x_batch)
    return obj, policy, theta, x_batch, true_grad


# ---------------------------------------------------------------------------
# Gauss-Stein convergence
# ---------------------------------------------------------------------------

def test_gauss_stein_converges_to_true_grad():
    """With large n_grad_samples, Gauss-Stein approaches the true gradient."""
    obj, policy, theta, x_batch, true_grad = _setup()
    sigma = 0.01
    n_samples_grad = 5000  # one-sided estimator needs more samples
    rng = np.random.default_rng(99)

    # One-sided Gauss-Stein in theta-space
    eps_all = rng.normal(0, 1, size=(n_samples_grad, theta.size))
    accum = np.zeros_like(theta)
    for eps in eps_all:
        val = obj.value(theta + sigma * eps, x_batch)
        accum += val * eps
    estimate = accum / n_samples_grad / sigma

    np.testing.assert_allclose(estimate, true_grad, atol=0.3)


# ---------------------------------------------------------------------------
# SPSA convergence
# ---------------------------------------------------------------------------

def test_spsa_converges_to_true_grad():
    """With large n_grad_samples, SPSA approaches the true gradient."""
    obj, policy, theta, x_batch, true_grad = _setup()
    sigma = 0.01
    n_samples_grad = 2000
    rng = np.random.default_rng(99)

    delta_all = rng.choice(np.array([-1.0, 1.0]), size=(n_samples_grad, theta.size))
    accum = np.zeros_like(theta)
    for delta in delta_all:
        vp = obj.value(theta + sigma * delta, x_batch)
        vm = obj.value(theta - sigma * delta, x_batch)
        accum += ((vp - vm) / (2.0 * sigma)) * delta
    estimate = accum / n_samples_grad

    np.testing.assert_allclose(estimate, true_grad, atol=0.15)


# ---------------------------------------------------------------------------
# Stein-Difference convergence
# ---------------------------------------------------------------------------

def test_stein_difference_converges_to_true_grad():
    """With large n_grad_samples, Stein-Difference approaches the true gradient."""
    obj, policy, theta, x_batch, true_grad = _setup()
    sigma = 0.01
    n_samples_grad = 2000
    rng = np.random.default_rng(99)

    eps_all = rng.normal(0, 1, size=(n_samples_grad, theta.size))
    accum = np.zeros_like(theta)
    for eps in eps_all:
        vp = obj.value(theta + sigma * eps, x_batch)
        vm = obj.value(theta - sigma * eps, x_batch)
        accum += ((vp - vm) / (2.0 * sigma)) * eps
    estimate = accum / n_samples_grad

    np.testing.assert_allclose(estimate, true_grad, atol=0.15)


# ---------------------------------------------------------------------------
# FD: u-space vs theta-space agreement
# ---------------------------------------------------------------------------

def test_fd_u_space_matches_theta_space():
    """U-space and theta-space central FD agree on the quadratic objective."""
    obj, policy, theta, x_batch, true_grad = _setup()
    sigma = 1e-5

    # Theta-space FD (coordinate-wise)
    fd_theta = np.zeros_like(theta)
    for i in range(theta.size):
        e = np.zeros_like(theta)
        e[i] = 1.0
        fd_theta[i] = (obj.value(theta + sigma * e, x_batch) - obj.value(theta - sigma * e, x_batch)) / (2.0 * sigma)

    # U-space FD (per-sample, chain-ruled)
    u_arr = policy.value(theta, x_batch)
    grad_pi = policy.grad(theta, x_batch)
    vp = obj._value_batch(x_batch, u_arr + sigma)
    vm = obj._value_batch(x_batch, u_arr - sigma)
    grad_u = (vp - vm) / (2.0 * sigma)
    fd_u = np.mean(grad_u[:, None] * grad_pi, axis=0)

    np.testing.assert_allclose(fd_u, fd_theta, atol=1e-4)


# ---------------------------------------------------------------------------
# SPSA variance decreases with more samples
# ---------------------------------------------------------------------------

def test_spsa_variance_decreases_with_samples():
    """Variance of SPSA estimate decreases as n_grad_samples increases."""
    obj, policy, theta, x_batch, true_grad = _setup()
    sigma = 0.01

    def spsa_estimate(n_samples, seed):
        rng = np.random.default_rng(seed)
        delta_all = rng.choice(np.array([-1.0, 1.0]), size=(n_samples, theta.size))
        accum = np.zeros_like(theta)
        for delta in delta_all:
            vp = obj.value(theta + sigma * delta, x_batch)
            vm = obj.value(theta - sigma * delta, x_batch)
            accum += ((vp - vm) / (2.0 * sigma)) * delta
        return accum / n_samples

    # Collect multiple estimates and compute variance of the norm
    n_trials = 20
    errors_small = [np.linalg.norm(spsa_estimate(10, s) - true_grad) for s in range(n_trials)]
    errors_large = [np.linalg.norm(spsa_estimate(500, s) - true_grad) for s in range(n_trials)]

    assert np.var(errors_large) < np.var(errors_small)
