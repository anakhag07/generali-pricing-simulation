"""Tests for early stopping based on gradient norm."""

import numpy as np

from objective.base import StateVector
from objective import FixedRegressionObjective
from objective.policy import LinearPolicy
from optimization.solvers import run_first_order_minimize, run_gauss_stein_minimize


def _build_theta_objective() -> FixedRegressionObjective:
    """Build a simple theta-space objective for testing."""
    return FixedRegressionObjective.from_parameters(
        policy=LinearPolicy(),
        beta_1=[0.2],
        beta_2=-0.4,
        beta_3=[0.1],
        beta_4=0.3,
    )


def test_first_order_early_stops_on_grad_norm() -> None:
    theta_start = np.array([0.1, 0.2], dtype=float)
    x_samples = [StateVector(values=np.asarray([1.0], dtype=float))]
    objective = _build_theta_objective()
    _, trace = run_first_order_minimize(
        theta_start,
        x_samples,
        objective,
        t_steps=5,
        n_grad_samples=2,
        sigma=0.1,
        grad_norm_tol=1e6,
    )
    assert len(trace.steps) <= 3
    assert trace.theta_values is not None
    assert len(trace.theta_values) == len(trace.steps)


def test_gauss_stein_early_stops_on_grad_norm() -> None:
    theta_start = np.array([0.1, 0.2], dtype=float)
    x_samples = [StateVector(values=np.asarray([1.0], dtype=float))]
    objective = _build_theta_objective()
    rng = np.random.default_rng(0)
    _, trace = run_gauss_stein_minimize(
        theta_start,
        x_samples,
        objective,
        rng,
        t_steps=5,
        n_grad_samples=2,
        sigma=0.1,
        grad_norm_tol=1e6,
    )
    assert len(trace.steps) <= 3
    assert trace.theta_values is not None
    assert len(trace.theta_values) == len(trace.steps)
