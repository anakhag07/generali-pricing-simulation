"""Verify MLPPolicy.grad matches a per-coordinate finite-difference Jacobian."""

from __future__ import annotations

import numpy as np
import pytest

from objective.policy import (
    IdentityFeatureMap,
    MLPPolicy,
    QuadraticFeatureMap,
    mlp_init_theta,
)


def _fd_jacobian(policy: MLPPolicy, theta: np.ndarray, x_array: np.ndarray, *, step: float = 1e-6) -> np.ndarray:
    n_samples = x_array.shape[0]
    fd = np.zeros((n_samples, theta.size), dtype=float)
    for idx in range(theta.size):
        direction = np.zeros_like(theta)
        direction[idx] = 1.0
        upper = policy.value(theta + step * direction, x_array)
        lower = policy.value(theta - step * direction, x_array)
        fd[:, idx] = (upper - lower) / (2.0 * step)
    return fd


def test_mlp_grad_matches_finite_difference_identity_map() -> None:
    rng = np.random.default_rng(0)
    state_dim = 4
    hidden = 6
    policy = MLPPolicy(feature_map=IdentityFeatureMap(), hidden=hidden)
    theta = mlp_init_theta(rng, d_in=state_dim, hidden=hidden)
    x_array = rng.normal(size=(3, state_dim))

    analytical = policy.grad(theta, x_array)
    fd = _fd_jacobian(policy, theta, x_array, step=1e-6)

    np.testing.assert_allclose(analytical, fd, atol=1e-6)


def test_mlp_grad_matches_finite_difference_quadratic_map() -> None:
    rng = np.random.default_rng(1)
    state_dim = 3
    hidden = 5
    policy = MLPPolicy(feature_map=QuadraticFeatureMap(), hidden=hidden)
    theta = mlp_init_theta(
        rng, d_in=QuadraticFeatureMap().output_dim(state_dim), hidden=hidden
    )
    x_array = rng.normal(size=(2, state_dim))

    analytical = policy.grad(theta, x_array)
    fd = _fd_jacobian(policy, theta, x_array, step=1e-6)

    np.testing.assert_allclose(analytical, fd, atol=1e-6)


def test_mlp_grad_zero_theta_is_finite_and_nonzero_in_first_layer() -> None:
    """At theta=0 the network output is constant (z3=0, u=0), but the Jacobian
    is well-defined: du/dz3 = -0.25 propagates back through layers.
    Layers 1 and 2 receive zero gradient because their downstream weights are
    zero, but the W3 / b3 columns are not zero (h2 = tanh(0) = 0 makes dW3
    zero, but db3 is -0.25).
    """
    state_dim = 2
    hidden = 3
    policy = MLPPolicy(feature_map=IdentityFeatureMap(), hidden=hidden)
    theta = np.zeros(policy.theta_dim(state_dim), dtype=float)
    x_array = np.array([[0.5, -0.3]], dtype=float)

    grad = policy.grad(theta, x_array)
    assert grad.shape == (1, policy.theta_dim(state_dim))
    assert np.all(np.isfinite(grad))
    # b3 lives at the very last index; its gradient is du/dz3 = -sigma*(1-sigma) at z=0 = -0.25.
    assert grad[0, -1] == pytest.approx(-0.25)


def test_mlp_init_theta_size_matches_policy_theta_dim() -> None:
    rng = np.random.default_rng(7)
    for state_dim, hidden in [(1, 4), (5, 8), (12, 16)]:
        policy = MLPPolicy(hidden=hidden)
        theta = mlp_init_theta(rng, d_in=state_dim, hidden=hidden)
        assert theta.size == policy.theta_dim(state_dim)
        assert np.all(np.isfinite(theta))


def test_mlp_init_theta_glorot_breaks_hidden_unit_symmetry() -> None:
    rng = np.random.default_rng(3)
    state_dim = 4
    hidden = 8
    theta = mlp_init_theta(rng, d_in=state_dim, hidden=hidden)
    # First layer weights occupy the first state_dim*hidden entries.
    W1 = theta[: state_dim * hidden].reshape(state_dim, hidden)
    # Columns (hidden units) must not be all-equal — that would be the symmetry
    # we explicitly want random init to avoid.
    column_stds = W1.std(axis=0)
    assert np.all(column_stds > 0.0)
