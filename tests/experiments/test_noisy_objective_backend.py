from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from experiments import run as run_module
from objective.noise import NoisyObjective, NoNoise
from objective.objectives import ModelBasedObjective
from objective.policy import ConstantPolicy


def test_acceptance_controls_are_applied_inside_noisy_objective() -> None:
    objective = NoisyObjective(_glm_objective(), NoNoise())
    config = SimpleNamespace(
        acceptance_floor=0.8,
        acceptance_penalty_weight=None,
        acceptance_penalty_temperature=0.01,
        lagrangian_lambda=None,
    )

    updated = run_module._objective_with_acceptance_controls(objective, config)

    assert isinstance(updated, NoisyObjective)
    assert updated.base_objective.acceptance_floor == 0.8
    assert updated.base_objective.acceptance_penalty_weight is None
    assert objective.base_objective.acceptance_floor is None


@pytest.mark.parametrize("model_type", ["glm", "linear"])
def test_jax_backend_rewraps_noisy_glm_objective(monkeypatch, model_type: str) -> None:
    base = _glm_objective(model_type=model_type)
    prepared = SimpleNamespace(policy=base.policy, warmed_theta=None)

    def fake_prepare_jax_glm_objective(objective, x_samples, *, row_indices=None):
        assert objective is base
        assert row_indices is None
        prepared_batch = SimpleNamespace(x_array=np.asarray([[1.0], [2.0]], dtype=float))
        prepared.warmup = lambda theta: setattr(prepared, "warmed_theta", np.asarray(theta, dtype=float))
        return prepared, prepared_batch

    monkeypatch.setattr(run_module, "_prepare_jax_glm_objective", fake_prepare_jax_glm_objective)
    config = SimpleNamespace(
        compute_backend="jax",
        step_rule="trust-constr",
        batch_size=None,
        enabled_estimators=("finite_difference",),
    )
    theta_initial = np.asarray([0.0], dtype=float)

    optimizer_objective, optimizer_x = run_module._optimizer_backend_objective(
        config,
        NoisyObjective(base, NoNoise()),
        np.asarray([[0.0]], dtype=float),
        theta_initial,
        row_indices=None,
    )

    assert isinstance(optimizer_objective, NoisyObjective)
    assert optimizer_objective.base_objective is prepared
    np.testing.assert_allclose(optimizer_x, [[1.0], [2.0]])
    np.testing.assert_allclose(prepared.warmed_theta, theta_initial)


def _glm_objective(*, model_type: str = "glm") -> ModelBasedObjective:
    policy = ConstantPolicy()
    return ModelBasedObjective(
        policy=policy,
        acceptance_model=SimpleNamespace(model_type=model_type),
        loss_model=SimpleNamespace(),
        acceptance_state_cols=("x",),
        loss_cols=("x",),
        premium_col="x",
    )
