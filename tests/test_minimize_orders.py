import numpy as np

from objective.base import ObjectiveResult, StateVector
from experiments.helpers import run_first_order, run_zeroth_order
from model.policy import POLICY_LINEAR


class SimpleObjective:
    def value(self, x: StateVector, u: float) -> float:
        return u**2

    def grad_u(self, x: StateVector, u: float) -> float:
        return 2.0 * u

    def evaluate(self, x: StateVector, u: float) -> ObjectiveResult:
        return ObjectiveResult(value=self.value(x, u), grad_u=self.grad_u(x, u))


def test_first_order_minimize_reduces_objective() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = [StateVector(values=[1.0]), StateVector(values=[-0.5])]
    objective = SimpleObjective()
    rng = np.random.default_rng(0)

    theta_final, trace = run_first_order(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        rng,
        t_steps=25,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=4,
        sigma=0.1,
    )
    assert trace.objective_values
    assert trace.objective_values[-1] <= trace.objective_values[0] + 1e-10
    assert theta_final.shape == theta_start.shape


def test_zeroth_order_minimize_is_seed_deterministic() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = [StateVector(values=[1.0]), StateVector(values=[-0.5])]
    objective = SimpleObjective()

    theta_a, trace_a = run_zeroth_order(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        np.random.default_rng(11),
        t_steps=20,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=8,
        sigma=0.1,
    )
    theta_b, trace_b = run_zeroth_order(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        np.random.default_rng(11),
        t_steps=20,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=8,
        sigma=0.1,
    )

    assert np.allclose(theta_a, theta_b)
    assert np.allclose(trace_a.objective_values, trace_b.objective_values)
