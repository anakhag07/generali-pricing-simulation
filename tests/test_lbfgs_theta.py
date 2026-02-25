from __future__ import annotations

import numpy as np

from data.models import ObjectiveResult, StateVector
from experiments.helpers import run_lbfgs_theta
from optimization.policy import POLICY_LINEAR, policy_u


class QuadraticObjective:
    def value(self, x: StateVector, u: float) -> float:
        return u**2

    def grad_u(self, x: StateVector, u: float) -> float:
        return 2.0 * u

    def evaluate(self, x: StateVector, u: float) -> ObjectiveResult:
        return ObjectiveResult(value=self.value(x, u), grad_u=self.grad_u(x, u))


def test_run_lbfgs_theta_reduces_objective() -> None:
    x_samples = [StateVector(values=[1.0, -0.5]), StateVector(values=[0.5, 0.25])]
    theta_start = np.asarray([0.5, 0.2, -0.1], dtype=float)
    objective_model = QuadraticObjective()

    start_values = [
        objective_model.value(x, policy_u(theta_start, x, kind=POLICY_LINEAR)) for x in x_samples
    ]
    value_start = float(np.mean(start_values))

    theta_lbfgs, value_lbfgs = run_lbfgs_theta(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective_model,
        maxiter=50,
    )

    assert theta_lbfgs.shape == theta_start.shape
    assert value_lbfgs <= value_start + 1e-10
