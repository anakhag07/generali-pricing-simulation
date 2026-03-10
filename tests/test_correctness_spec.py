from __future__ import annotations

import math

import pytest

from objective.base import ObjectiveResult, StateVector
from objective.fixed_objective import FixedRegressionObjective
from experiments.config import CorrectnessSpec, ExperimentConfig
from experiments.defaults import default_policy_spec
from experiments.helpers import resolve_true_grad_u_fn


class DummyObjectiveNoGrad:
    def value(self, _x: StateVector, _u: float) -> float:
        return 0.0

    def evaluate(self, _x: StateVector, _u: float) -> ObjectiveResult:
        return ObjectiveResult(value=0.0, grad_u=0.0)


def test_correctness_exact_requires_grad_u() -> None:
    with pytest.raises(ValueError, match="grad_u"):
        ExperimentConfig(
            state_dim=1,
            objective_model=DummyObjectiveNoGrad(),
            policy_spec=default_policy_spec(1),
            n_samples=1,
            step_rule="constant",
            correctness=CorrectnessSpec(gradient_source="exact"),
        )


def test_resolve_true_grad_numdiff_matches_exact() -> None:
    objective = FixedRegressionObjective.from_parameters(
        beta_1=[0.2],
        beta_2=-0.4,
        beta_3=[0.3],
        beta_4=0.5,
    )
    correctness = CorrectnessSpec(
        gradient_source="numdiff",
        numdiff_method="central",
        numdiff_step=1e-5,
    )
    true_grad_fn = resolve_true_grad_u_fn(objective, correctness)
    assert true_grad_fn is not None

    x = StateVector(values=[0.8])
    u = 1.1
    grad_exact = objective.grad_u(x, u)
    grad_numdiff = true_grad_fn(x, u)
    assert math.isclose(grad_numdiff, grad_exact, rel_tol=1e-4, abs_tol=1e-4)


def test_resolve_true_grad_none_returns_none() -> None:
    objective = FixedRegressionObjective.from_parameters(
        beta_1=[0.2],
        beta_2=-0.4,
        beta_3=[0.3],
        beta_4=0.5,
    )
    correctness = CorrectnessSpec(gradient_source="none")
    assert resolve_true_grad_u_fn(objective, correctness) is None


def test_numdiff_batch_unsupported_for_theta_grad() -> None:
    objective = FixedRegressionObjective.from_parameters(
        beta_1=[0.2],
        beta_2=-0.4,
        beta_3=[0.3],
        beta_4=0.5,
    )
    with pytest.raises(ValueError, match="numdiff_aggregate='batch'"):
        ExperimentConfig(
            state_dim=1,
            objective_model=objective,
            policy_spec=default_policy_spec(1),
            n_samples=1,
            step_rule="constant",
            correctness=CorrectnessSpec(
                gradient_source="numdiff",
                numdiff_aggregate="batch",
            ),
        )
