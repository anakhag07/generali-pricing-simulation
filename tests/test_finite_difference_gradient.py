from __future__ import annotations

import numpy as np

from objective import FixedRegressionObjective
from objective.policy import LinearPolicy
from optimization import FiniteDifferenceGradient, Optimization


def _build_objective() -> FixedRegressionObjective:
    return FixedRegressionObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=[0.2],
        beta_2=-0.4,
        beta_3=[0.1],
        beta_4=0.3,
    )


def test_finite_difference_theta_grad_matches_exact_gradient() -> None:
    objective = _build_objective()
    x_samples = np.array([[1.0], [-0.5]], dtype=float)
    theta = np.asarray([0.2, 0.3], dtype=float)
    optimizer = Optimization(
        objective,
        x_samples,
        FiniteDifferenceGradient(),
        algorithm="l-bfgs-b",
        t_steps=5,
        n_grad_samples=1,
        sigma=1e-5,
    )

    indices = np.arange(optimizer.n_total, dtype=int)
    optimizer.gradient.setup(optimizer, theta)
    grad_fd = optimizer.gradient.theta_grad(optimizer, theta, indices)
    grad_exact = objective.grad(theta, x_samples)

    assert np.allclose(grad_fd, grad_exact, rtol=1e-4, atol=1e-4)


def test_finite_difference_theta_grad_is_deterministic() -> None:
    objective = _build_objective()
    x_samples = np.array([[1.0], [-0.5]], dtype=float)
    theta = np.asarray([0.2, 0.3], dtype=float)
    optimizer = Optimization(
        objective,
        x_samples,
        FiniteDifferenceGradient(),
        algorithm="l-bfgs-b",
        t_steps=5,
        n_grad_samples=1,
        sigma=1e-3,
    )

    indices = np.arange(optimizer.n_total, dtype=int)
    optimizer.gradient.setup(optimizer, theta)
    grad_a = optimizer.gradient.theta_grad(optimizer, theta, indices)
    grad_b = optimizer.gradient.theta_grad(optimizer, theta, indices)

    assert np.allclose(grad_a, grad_b)
