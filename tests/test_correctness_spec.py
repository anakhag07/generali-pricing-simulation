from __future__ import annotations

import numpy as np
import pytest

from experiments.config import CorrectnessSpec, ExperimentConfig
from experiments.defaults import default_theta0
from experiments.helpers import resolve_true_grad_theta_fn
from objective.composed import PolicyObjective
from objective.fixed_objective import FixedRegressionObjective
from objective.policy import SoftmaxPolicy


class DummyThetaObjectiveNoGrad:
    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        del x_batch
        return float(np.sum(theta**2))


def _build_theta_objective() -> PolicyObjective:
    action = FixedRegressionObjective.from_parameters(
        beta_1=[0.2],
        beta_2=-0.4,
        beta_3=[0.3],
        beta_4=0.5,
    )
    return PolicyObjective(action_objective=action, policy=SoftmaxPolicy())


def test_correctness_exact_requires_theta_grad() -> None:
    with pytest.raises(ValueError, match="objective must implement grad"):
        ExperimentConfig(
            state_dim=1,
            n_samples=1,
            step_rule="constant",
            objective=DummyThetaObjectiveNoGrad(),
            theta0=default_theta0(1),
            correctness=CorrectnessSpec(gradient_source="exact"),
        )


def test_resolve_true_grad_numdiff_matches_exact() -> None:
    objective = _build_theta_objective()
    correctness = CorrectnessSpec(
        gradient_source="numdiff",
        numdiff_method="central",
        numdiff_step=1e-5,
    )
    true_grad_fn = resolve_true_grad_theta_fn(objective, correctness)
    assert true_grad_fn is not None

    theta = default_theta0(1)
    x_batch = np.asarray([[0.8]], dtype=float)
    grad_exact = objective.grad(theta, x_batch)
    grad_numdiff = true_grad_fn(theta, x_batch)
    assert np.allclose(grad_numdiff, grad_exact, rtol=1e-4, atol=1e-4)


def test_resolve_true_grad_none_returns_none() -> None:
    objective = _build_theta_objective()
    correctness = CorrectnessSpec(gradient_source="none")
    assert resolve_true_grad_theta_fn(objective, correctness) is None


def test_numdiff_batch_is_supported_for_theta_grad() -> None:
    objective = _build_theta_objective()
    config = ExperimentConfig(
        state_dim=1,
        n_samples=1,
        step_rule="constant",
        objective=objective,
        theta0=default_theta0(1),
        correctness=CorrectnessSpec(
            gradient_source="numdiff",
            numdiff_aggregate="batch",
        ),
    )
    assert config.correctness.numdiff_aggregate == "batch"
