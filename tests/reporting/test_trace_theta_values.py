import numpy as np

from objective import FixedRegressionObjective
from objective.policy import LinearPolicy
from optimization.solvers import (
    run_finite_difference_minimize,
    run_first_order_minimize,
    run_gauss_stein_minimize,
    run_stein_difference_minimize,
)


def _build_theta_objective() -> FixedRegressionObjective:
    """Build a simple theta-space objective for testing."""
    return FixedRegressionObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=[0.2],
        beta_2=-0.4,
        beta_3=[0.1],
        beta_4=0.3,
    )


def test_run_first_order_records_theta_values() -> None:
    theta_start = np.array([0.1, 0.2], dtype=float)
    x_samples = np.array([[1.0]], dtype=float)
    objective = _build_theta_objective()
    _, trace = run_first_order_minimize(
        theta_start,
        x_samples,
        objective,
        t_steps=3,
        n_grad_samples=2,
        sigma=0.1,
    )
    assert trace.theta_values is not None
    assert len(trace.theta_values) >= 1
    assert np.allclose(trace.theta_values[0], theta_start)
    assert len(trace.steps) == len(trace.theta_values)
    assert trace.step_sizes is None


def test_run_first_order_constant_records_step_sizes() -> None:
    theta_start = np.array([0.1, 0.2], dtype=float)
    x_samples = np.array([[1.0]], dtype=float)
    objective = _build_theta_objective()
    _, trace = run_first_order_minimize(
        theta_start,
        x_samples,
        objective,
        t_steps=3,
        n_grad_samples=2,
        sigma=0.1,
        algorithm="constant",
        step_size=0.05,
    )
    assert trace.step_sizes is not None
    assert len(trace.steps) == len(trace.step_sizes)
    assert np.isnan(trace.step_sizes[0])
    assert np.allclose(trace.step_sizes[1:], 0.05)


def test_run_first_order_armijo_records_step_sizes() -> None:
    theta_start = np.array([0.1, 0.2], dtype=float)
    x_samples = np.array([[1.0]], dtype=float)
    objective = _build_theta_objective()
    _, trace = run_first_order_minimize(
        theta_start,
        x_samples,
        objective,
        t_steps=3,
        n_grad_samples=2,
        sigma=0.1,
        algorithm="armijo",
        step_size=0.1,
    )
    assert trace.step_sizes is not None
    assert len(trace.steps) == len(trace.step_sizes)
    assert np.isnan(trace.step_sizes[0])
    assert np.all(np.asarray(trace.step_sizes[1:], dtype=float) > 0.0)
    assert np.all(np.asarray(trace.step_sizes[1:], dtype=float) <= 0.1)


def test_run_finite_difference_records_theta_values() -> None:
    theta_start = np.array([0.1, 0.2], dtype=float)
    x_samples = np.array([[1.0]], dtype=float)
    objective = _build_theta_objective()
    _, trace = run_finite_difference_minimize(
        theta_start,
        x_samples,
        objective,
        t_steps=3,
        n_grad_samples=1,
        sigma=1e-3,
    )
    assert trace.theta_values is not None
    assert len(trace.theta_values) >= 1
    assert np.allclose(trace.theta_values[0], theta_start)
    assert len(trace.steps) == len(trace.theta_values)
    assert trace.step_sizes is None


def test_run_gauss_stein_records_theta_values() -> None:
    theta_start = np.array([0.1, 0.2], dtype=float)
    x_samples = np.array([[1.0]], dtype=float)
    objective = _build_theta_objective()
    rng = np.random.default_rng(0)
    _, trace = run_gauss_stein_minimize(
        theta_start,
        x_samples,
        objective,
        rng,
        t_steps=3,
        n_grad_samples=2,
        sigma=0.1,
    )
    assert trace.theta_values is not None
    assert len(trace.theta_values) >= 1
    assert np.allclose(trace.theta_values[0], theta_start)
    assert len(trace.steps) == len(trace.theta_values)
    assert trace.step_sizes is None


def test_run_stein_difference_records_theta_values() -> None:
    theta_start = np.array([0.1, 0.2], dtype=float)
    x_samples = np.array([[1.0]], dtype=float)
    objective = _build_theta_objective()
    rng = np.random.default_rng(0)
    _, trace = run_stein_difference_minimize(
        theta_start,
        x_samples,
        objective,
        rng,
        t_steps=3,
        n_grad_samples=2,
        sigma=0.1,
    )
    assert trace.theta_values is not None
    assert len(trace.theta_values) >= 1
    assert np.allclose(trace.theta_values[0], theta_start)
    assert len(trace.steps) == len(trace.theta_values)
    assert trace.step_sizes is None
