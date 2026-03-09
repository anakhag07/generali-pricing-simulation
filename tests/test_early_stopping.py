import numpy as np

from data.models import ObjectiveResult, StateVector
import experiments.helpers as helpers
from experiments.helpers import run_first_order, run_lbfgs_theta, run_zeroth_order
from optimization.policy import POLICY_LINEAR


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
    assert len(trace.steps) == 0
    assert trace.theta_values is not None
    assert len(trace.theta_values) == 1


def test_zeroth_order_early_stops_on_grad_norm() -> None:
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
        t_steps=5,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=2,
        sigma=0.1,
        grad_norm_tol=1e6,
    )
    assert len(trace.steps) == 0
    assert trace.theta_values is not None
    assert len(trace.theta_values) == 1


def test_run_lbfgs_theta_passes_grad_norm_tol(monkeypatch) -> None:
    captured = {}

    def fake_minimize(fun, x0, jac, method, options, callback):
        captured["options"] = options

        class Result:
            x = np.asarray(x0, dtype=float)

        return Result()

    monkeypatch.setattr(helpers, "minimize", fake_minimize)

    objective = SimpleObjective()
    theta_start = np.array([1.0, 0.2], dtype=float)
    x_samples = [StateVector(values=[1.0])]
    run_lbfgs_theta(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        maxiter=5,
        grad_norm_tol=1e-3,
    )
    assert captured["options"]["gtol"] == 1e-3
