import numpy as np

from objective.base import ObjectiveResult, StateVector
from experiments.helpers import run_first_order, run_zeroth_order
from optimization.policy import POLICY_LINEAR


class SimpleObjective:
    def value(self, x: StateVector, u: float) -> float:
        return u**2

    def grad_u(self, x: StateVector, u: float) -> float:
        return 2.0 * u

    def evaluate(self, x: StateVector, u: float) -> ObjectiveResult:
        return ObjectiveResult(value=self.value(x, u), grad_u=self.grad_u(x, u))


def test_run_first_order_records_theta_values() -> None:
    theta_start = np.array([0.1, 0.2], dtype=float)
    x_samples = [StateVector(values=[1.0])]
    objective = SimpleObjective()
    rng = np.random.default_rng(0)
    _, trace = run_first_order(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        rng,
        t_steps=3,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=2,
        sigma=0.1,
        true_grad_u_fn=objective.grad_u,
    )
    assert trace.theta_values is not None
    assert len(trace.theta_values) == 4
    assert np.allclose(trace.theta_values[0], theta_start)
    assert trace.step_sizes is not None
    assert len(trace.step_sizes) == 3
    assert np.allclose(trace.step_sizes, [0.01, 0.01, 0.01])


def test_run_zeroth_order_records_theta_values() -> None:
    theta_start = np.array([0.1, 0.2], dtype=float)
    x_samples = [StateVector(values=[1.0])]
    objective = SimpleObjective()
    rng = np.random.default_rng(0)
    _, trace = run_zeroth_order(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        rng,
        t_steps=3,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=2,
        sigma=0.1,
        true_grad_u_fn=objective.grad_u,
    )
    assert trace.theta_values is not None
    assert len(trace.theta_values) == 4
    assert np.allclose(trace.theta_values[0], theta_start)
    assert trace.step_sizes is not None
    assert len(trace.step_sizes) == 3
    assert np.allclose(trace.step_sizes, [0.01, 0.01, 0.01])
