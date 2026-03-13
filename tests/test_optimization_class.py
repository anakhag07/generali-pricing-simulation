from __future__ import annotations

import numpy as np

from objective.base import ObjectiveResult, StateVector
from model.policy import POLICY_LINEAR
from optimization import FirstOrderGradient, GaussSteinGradient, Optimization


class SimpleObjective:
    def value(self, x: StateVector, u: float) -> float:
        return u**2

    def grad_u(self, x: StateVector, u: float) -> float:
        return 2.0 * u

    def evaluate(self, x: StateVector, u: float) -> ObjectiveResult:
        return ObjectiveResult(value=self.value(x, u), grad_u=self.grad_u(x, u))


def test_optimization_first_order_reduces_objective() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = [StateVector(values=[1.0]), StateVector(values=[-0.5])]
    objective = SimpleObjective()

    optimizer = Optimization(
        objective,
        POLICY_LINEAR,
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


def test_optimization_invalid_algorithm_raises() -> None:
    objective = SimpleObjective()
    x_samples = [StateVector(values=[1.0])]
    optimizer = Optimization(
        objective,
        POLICY_LINEAR,
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
    x_samples = [StateVector(values=[1.0]), StateVector(values=[-0.5])]
    objective = SimpleObjective()

    optimizer_a = Optimization(
        objective,
        POLICY_LINEAR,
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
        POLICY_LINEAR,
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
