"""Prepared GLM objective parity tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from data.loader import (
    ACCEPTANCE_STATE_COLS,
    LOSS_FEATURE_COLS,
    PREMIUM_COL,
    extract_glm_u_coef,
    load_model_artifacts,
    load_x_frame,
)
from objective.objectives.model_based import ModelBasedObjective
from objective.objectives.prepared_glm import (
    PreparedGLMBatch,
    PreparedGLMObjective,
    prepare_glm_batch,
    prepare_glm_objective,
)
from objective.policy import SoftmaxPolicy


def _make_glm_objective(n_rows: int = 30) -> tuple[ModelBasedObjective, object, np.ndarray]:
    acc_model, loss_model = load_model_artifacts("glm")
    x = load_x_frame("glm", n_rows=n_rows, seed=123)
    policy = SoftmaxPolicy(action_low=-0.1, action_high=0.2)
    objective = ModelBasedObjective(
        policy=policy,
        acceptance_model=acc_model,
        loss_model=loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=PREMIUM_COL,
        u_coef=extract_glm_u_coef(acc_model),
    )
    theta = np.array([0.05] + [0.01] * (objective.policy_theta_dim() - 1), dtype=float)
    return objective, x, theta


def test_prepare_glm_batch_materializes_numeric_arrays() -> None:
    objective, x, _ = _make_glm_objective(n_rows=12)
    row_indices = np.arange(x.shape[0], dtype=int) + 100

    batch = prepare_glm_batch(objective, x, row_indices=row_indices)

    assert isinstance(batch, PreparedGLMBatch)
    assert batch.n_rows == x.shape[0]
    assert batch.policy_feature_dim == objective.policy_input_dim()
    assert batch.x_array.shape == (x.shape[0], 3 + objective.policy_input_dim())
    np.testing.assert_array_equal(batch.row_indices, row_indices)


def test_prepared_glm_value_grad_and_metrics_match_model_based_objective() -> None:
    objective, x, theta = _make_glm_objective(n_rows=30)
    prepared, batch = prepare_glm_objective(objective, x)

    assert isinstance(prepared, PreparedGLMObjective)
    assert prepared.policy_theta_dim() == objective.policy_theta_dim()
    assert prepared.value(theta, batch.x_array) == pytest.approx(objective.value(theta, x))
    assert prepared.base_value(theta, batch.x_array) == pytest.approx(objective.base_value(theta, x))
    assert prepared.mean_acceptance(theta, batch.x_array) == pytest.approx(
        objective.mean_acceptance(theta, x)
    )
    assert prepared.value_at_u(batch.x_array, 0.03) == pytest.approx(objective.value_at_u(x, 0.03))
    assert prepared.base_value_at_u(batch.x_array, 0.03) == pytest.approx(
        objective.base_value_at_u(x, 0.03)
    )
    np.testing.assert_allclose(prepared.grad(theta, batch.x_array), objective.grad(theta, x))
    np.testing.assert_allclose(
        prepared.mean_acceptance_grad(theta, batch.x_array),
        objective.mean_acceptance_grad(theta, x),
    )
    assert prepared._step_metrics(theta, batch.x_array) == pytest.approx(
        objective._step_metrics(theta, x)
    )


def test_prepared_glm_supports_minibatch_slices() -> None:
    objective, x, theta = _make_glm_objective(n_rows=30)
    prepared, batch = prepare_glm_objective(objective, x)
    indices = np.array([0, 3, 4, 10, 17], dtype=int)

    x_slice = x.iloc[indices].reset_index(drop=True)
    prepared_slice = batch.x_array[indices]

    assert prepared.value(theta, prepared_slice) == pytest.approx(objective.value(theta, x_slice))
    np.testing.assert_allclose(prepared.grad(theta, prepared_slice), objective.grad(theta, x_slice))


def test_prepared_glm_constraint_terms_match_model_based_objective() -> None:
    objective, x, theta = _make_glm_objective(n_rows=30)
    baseline_acceptance = objective.mean_acceptance(theta, x)
    constrained = replace(
        objective,
        acceptance_floor=baseline_acceptance + 0.01,
        acceptance_penalty_weight=100.0,
        acceptance_penalty_temperature=1e-4,
    )
    prepared, batch = prepare_glm_objective(constrained, x)

    assert prepared.value(theta, batch) == pytest.approx(constrained.value(theta, x))
    np.testing.assert_allclose(prepared.grad(theta, batch), constrained.grad(theta, x))


def test_prepared_glm_rejects_incompatible_batch_shape() -> None:
    objective, x, _ = _make_glm_objective(n_rows=5)
    prepared, batch = prepare_glm_objective(objective, x)

    with pytest.raises(ValueError, match="columns"):
        prepared.value(np.zeros(prepared.policy_theta_dim(), dtype=float), batch.x_array[:, :-1])
