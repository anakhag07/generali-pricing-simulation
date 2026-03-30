from __future__ import annotations

import numpy as np

from objective import FixedRegressionObjective
from objective.policy import LinearPolicy
from optimization import (
    FiniteDifferenceGradient,
    FirstOrderGradient,
    GaussSteinGradient,
    Optimization,
    SPSAGradient,
    SteinDifferenceGradient,
)


def _build_objective() -> FixedRegressionObjective:
    return FixedRegressionObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=[0.2],
        beta_2=-0.4,
        beta_3=[0.1],
        beta_4=0.3,
    )


def _build_optimizer(gradient: object, seed: int) -> Optimization:
    return Optimization(
        _build_objective(),
        np.array([[1.0], [-0.5]], dtype=float),
        gradient,
        algorithm="l-bfgs-b",
        t_steps=5,
        n_grad_samples=4,
        sigma=0.1,
        rng=np.random.default_rng(seed),
    )


def test_first_order_theta_grad_is_stable_across_calls() -> None:
    optimizer = _build_optimizer(FirstOrderGradient(), seed=0)
    theta = np.asarray([0.2, 0.3], dtype=float)
    indices = np.arange(optimizer.n_total, dtype=int)

    optimizer.gradient.setup(optimizer, theta)
    grad_a = optimizer.gradient.theta_grad(optimizer, theta, indices)
    grad_b = optimizer.gradient.theta_grad(optimizer, theta, indices)

    assert np.allclose(grad_a, grad_b)


def test_finite_difference_theta_grad_is_stable_across_calls() -> None:
    optimizer = _build_optimizer(FiniteDifferenceGradient(), seed=5)
    theta = np.asarray([0.2, 0.3], dtype=float)
    indices = np.arange(optimizer.n_total, dtype=int)

    optimizer.gradient.setup(optimizer, theta)
    grad_a = optimizer.gradient.theta_grad(optimizer, theta, indices)
    grad_b = optimizer.gradient.theta_grad(optimizer, theta, indices)

    assert np.allclose(grad_a, grad_b)


def test_gauss_stein_theta_grad_resamples_each_call() -> None:
    optimizer = _build_optimizer(GaussSteinGradient(), seed=11)
    theta = np.asarray([0.2, 0.3], dtype=float)
    indices = np.arange(optimizer.n_total, dtype=int)

    optimizer.gradient.setup(optimizer, theta)
    grad_a = optimizer.gradient.theta_grad(optimizer, theta, indices)
    grad_b = optimizer.gradient.theta_grad(optimizer, theta, indices)

    assert not np.allclose(grad_a, grad_b)


def test_spsa_theta_grad_resamples_each_call() -> None:
    optimizer = _build_optimizer(SPSAGradient(), seed=17)
    theta = np.asarray([0.2, 0.3], dtype=float)
    indices = np.arange(optimizer.n_total, dtype=int)

    optimizer.gradient.setup(optimizer, theta)
    grad_a = optimizer.gradient.theta_grad(optimizer, theta, indices)
    grad_b = optimizer.gradient.theta_grad(optimizer, theta, indices)

    assert not np.allclose(grad_a, grad_b)


def test_stein_difference_theta_grad_resamples_each_call() -> None:
    optimizer = _build_optimizer(SteinDifferenceGradient(), seed=23)
    theta = np.asarray([0.2, 0.3], dtype=float)
    indices = np.arange(optimizer.n_total, dtype=int)

    optimizer.gradient.setup(optimizer, theta)
    grad_a = optimizer.gradient.theta_grad(optimizer, theta, indices)
    grad_b = optimizer.gradient.theta_grad(optimizer, theta, indices)

    assert not np.allclose(grad_a, grad_b)
