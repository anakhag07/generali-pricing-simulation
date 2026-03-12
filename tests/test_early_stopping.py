import numpy as np

from objective.base import ObjectiveResult, StateVector
from experiments.helpers import run_first_order, run_gauss_stein
from model.policy import POLICY_LINEAR


class SimpleObjective:
    def value(self, x: StateVector, u: float) -> float:
        return u**2

    def grad_u(self, x: StateVector, u: float) -> float:
        return 2.0 * u

    def evaluate(self, x: StateVector, u: float) -> ObjectiveResult:
        return ObjectiveResult(value=self.value(x, u), grad_u=self.grad_u(x, u))


def test_first_order_early_stops_on_grad_norm() -> None:
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
        t_steps=5,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=2,
        sigma=0.1,
        grad_norm_tol=1e6,
    )
    assert len(trace.steps) <= 2
    assert trace.theta_values is not None
    assert len(trace.theta_values) == len(trace.steps)


def test_gauss_stein_early_stops_on_grad_norm() -> None:
    theta_start = np.array([0.1, 0.2], dtype=float)
    x_samples = [StateVector(values=[1.0])]
    objective = SimpleObjective()
    rng = np.random.default_rng(0)
    _, trace = run_gauss_stein(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        rng,
        t_steps=5,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=2,
        sigma=0.1,
        grad_norm_tol=1e6,
    )
    assert len(trace.steps) <= 2
    assert trace.theta_values is not None
    assert len(trace.theta_values) == len(trace.steps)
