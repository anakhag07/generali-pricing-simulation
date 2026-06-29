"""Policy weighted-gradient VJP tests."""

from __future__ import annotations

import numpy as np
import pytest

from objective.policy import (
    ConstantPolicy,
    LinearPolicy,
    MLPPolicy,
    QuadraticFeatureMap,
    SoftmaxPolicy,
    mlp_init_theta,
)
from objective.utils import _theta_grad_from_u_grad


@pytest.mark.parametrize(
    ("policy", "state_dim", "theta"),
    [
        (ConstantPolicy(), 3, np.array([0.2], dtype=float)),
        (LinearPolicy(), 3, np.array([0.2, -0.1, 0.3, 0.4], dtype=float)),
        (SoftmaxPolicy(), 3, np.array([0.2, -0.1, 0.3, 0.4], dtype=float)),
        (
            LinearPolicy(feature_map=QuadraticFeatureMap()),
            2,
            np.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1], dtype=float),
        ),
        (
            SoftmaxPolicy(feature_map=QuadraticFeatureMap(), action_low=-0.1, action_high=0.2),
            2,
            np.array([0.2, -0.1, 0.3, 0.4, -0.2, 0.1], dtype=float),
        ),
    ],
)
def test_weighted_grad_matches_full_jacobian(policy, state_dim: int, theta: np.ndarray) -> None:
    rng = np.random.default_rng(100 + state_dim)
    x_batch = rng.normal(size=(7, state_dim))
    weights = rng.normal(size=x_batch.shape[0])

    result = policy.weighted_grad(theta, x_batch, weights)
    expected = weights @ policy.grad(theta, x_batch)

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_mlp_weighted_grad_matches_full_jacobian() -> None:
    rng = np.random.default_rng(123)
    state_dim = 4
    hidden = 5
    policy = MLPPolicy(hidden=hidden)
    theta = mlp_init_theta(rng, d_in=state_dim, hidden=hidden)
    x_batch = rng.normal(size=(6, state_dim))
    weights = rng.normal(size=x_batch.shape[0])

    result = policy.weighted_grad(theta, x_batch, weights)
    expected = weights @ policy.grad(theta, x_batch)

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_theta_grad_from_u_grad_uses_weighted_grad_hook() -> None:
    class WeightedOnlyPolicy:
        def value(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
            del theta
            return np.zeros(x_batch.shape[0], dtype=float)

        def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
            del theta, x_batch
            raise AssertionError("full Jacobian should not be materialized")

        def weighted_grad(
            self,
            theta: np.ndarray,
            x_batch: np.ndarray,
            weights: np.ndarray,
        ) -> np.ndarray:
            del theta, x_batch
            return np.asarray([np.sum(weights), 2.0 * np.sum(weights)], dtype=float)

    x_batch = np.zeros((4, 2), dtype=float)
    grad_u = np.array([1.0, -0.5, 0.25, 2.0], dtype=float)

    result = _theta_grad_from_u_grad(WeightedOnlyPolicy(), np.zeros(2), x_batch, grad_u)
    expected = np.array([np.sum(grad_u), 2.0 * np.sum(grad_u)], dtype=float) / grad_u.size

    np.testing.assert_allclose(result, expected)
