from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from objective import FixedRegressionObjective
from objective.policy import LinearPolicy
from optimization import (
    FiniteDifferenceGradient,
    FirstOrderGradient,
    GaussSteinGradient,
    Optimization,
    SteinDifferenceGradient,
)


def _build_theta_objective() -> FixedRegressionObjective:
    """Build a simple theta-space objective with LinearPolicy for testing."""
    return FixedRegressionObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=[0.2],
        beta_2=-0.4,
        beta_3=[0.1],
        beta_4=0.3,
    )


def test_optimization_first_order_reduces_objective() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = np.array([[1.0], [-0.5]], dtype=float)
    objective = _build_theta_objective()

    optimizer = Optimization(
        objective,
        x_samples,
        FirstOrderGradient(),
        algorithm="l-bfgs-b",
        t_steps=25,
        n_grad_samples=4,
        sigma=0.1,
    )
    theta_final, trace = optimizer.solve(theta_start)

    assert trace.objective_values
    assert trace.objective_values[-1] <= trace.objective_values[0] + 1e-10
    assert theta_final.shape == theta_start.shape


def test_optimization_finite_difference_reduces_objective() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = np.array([[1.0], [-0.5]], dtype=float)
    objective = _build_theta_objective()

    optimizer = Optimization(
        objective,
        x_samples,
        FiniteDifferenceGradient(),
        algorithm="l-bfgs-b",
        t_steps=25,
        n_grad_samples=1,
        sigma=1e-3,
    )
    theta_final, trace = optimizer.solve(theta_start)

    assert trace.objective_values
    assert trace.objective_values[-1] <= trace.objective_values[0] + 1e-10
    assert theta_final.shape == theta_start.shape


def test_optimization_invalid_algorithm_raises() -> None:
    objective = _build_theta_objective()
    x_samples = np.array([[1.0]], dtype=float)
    optimizer = Optimization(
        objective,
        x_samples,
        FirstOrderGradient(),
        algorithm="adam",
        t_steps=5,
        n_grad_samples=2,
        sigma=0.1,
    )

    try:
        optimizer.solve(np.asarray([0.1, 0.2], dtype=float))
    except ValueError as exc:
        assert "Unsupported algorithm" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported algorithm")


def test_optimization_gauss_stein_is_seed_deterministic() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = np.array([[1.0], [-0.5]], dtype=float)
    objective = _build_theta_objective()

    optimizer_a = Optimization(
        objective,
        x_samples,
        GaussSteinGradient(),
        algorithm="l-bfgs-b",
        t_steps=20,
        n_grad_samples=8,
        sigma=0.1,
        rng=np.random.default_rng(11),
    )
    optimizer_b = Optimization(
        objective,
        x_samples,
        GaussSteinGradient(),
        algorithm="l-bfgs-b",
        t_steps=20,
        n_grad_samples=8,
        sigma=0.1,
        rng=np.random.default_rng(11),
    )

    theta_a, trace_a = optimizer_a.solve(theta_start)
    theta_b, trace_b = optimizer_b.solve(theta_start)

    assert np.allclose(theta_a, theta_b)
    assert np.allclose(trace_a.objective_values, trace_b.objective_values)


def test_optimization_stein_difference_is_seed_deterministic() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = np.array([[1.0], [-0.5]], dtype=float)
    objective = _build_theta_objective()

    optimizer_a = Optimization(
        objective,
        x_samples,
        SteinDifferenceGradient(),
        algorithm="l-bfgs-b",
        t_steps=20,
        n_grad_samples=8,
        sigma=0.1,
        rng=np.random.default_rng(29),
    )
    optimizer_b = Optimization(
        objective,
        x_samples,
        SteinDifferenceGradient(),
        algorithm="l-bfgs-b",
        t_steps=20,
        n_grad_samples=8,
        sigma=0.1,
        rng=np.random.default_rng(29),
    )

    theta_a, trace_a = optimizer_a.solve(theta_start)
    theta_b, trace_b = optimizer_b.solve(theta_start)

    assert np.allclose(theta_a, theta_b)
    assert np.allclose(trace_a.objective_values, trace_b.objective_values)


def test_scipy_callback_reuses_last_optimizer_gradient_for_recording() -> None:
    class CountingGradient(FirstOrderGradient):
        def __init__(self) -> None:
            self.calls = 0

        def theta_grad(self, optimizer: Optimization, theta: np.ndarray, indices: np.ndarray) -> np.ndarray:
            self.calls += 1
            return super().theta_grad(optimizer, theta, indices)

    def fake_minimize(fun, *, x0, jac, callback, **_kwargs):
        fun(x0)
        jac(x0)
        callback(x0)
        return SimpleNamespace(x=x0, status=0, success=True, message="ok")

    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = np.array([[1.0], [-0.5]], dtype=float)
    objective = _build_theta_objective()
    gradient = CountingGradient()

    optimizer = Optimization(
        objective,
        x_samples,
        gradient,
        algorithm="l-bfgs-b",
        t_steps=5,
        n_grad_samples=1,
        sigma=0.1,
        minimize_fn=fake_minimize,
    )
    optimizer.solve(theta_start)

    assert gradient.calls == 2
