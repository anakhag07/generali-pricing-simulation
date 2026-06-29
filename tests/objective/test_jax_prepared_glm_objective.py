"""JAX prepared GLM objective parity tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from data.loader import (  # noqa: E402
    ACCEPTANCE_STATE_COLS,
    LOSS_FEATURE_COLS,
    PREMIUM_COL,
    extract_glm_u_coef,
    load_model_artifacts,
    load_x_frame,
)
from objective.objectives.jax_prepared_glm import prepare_jax_glm_objective  # noqa: E402
from objective.objectives.model_based import ModelBasedObjective  # noqa: E402
from objective.objectives.prepared_glm import prepare_glm_objective  # noqa: E402
from objective.policy import SoftmaxPolicy  # noqa: E402


def _make_glm_objective(n_rows: int = 24) -> tuple[ModelBasedObjective, object, np.ndarray]:
    acc_model, loss_model = load_model_artifacts("glm")
    x = load_x_frame("glm", n_rows=n_rows, seed=321)
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
    theta = np.array([0.03] + [0.01] * (objective.policy_theta_dim() - 1), dtype=float)
    return objective, x, theta


def test_jax_prepared_glm_matches_model_based_and_numpy_prepared() -> None:
    objective, x, theta = _make_glm_objective()
    prepared, batch = prepare_glm_objective(objective, x)
    jax_objective, jax_batch = prepare_jax_glm_objective(objective, x)

    jax_objective.warmup(theta)

    np.testing.assert_allclose(jax_batch.x_array, batch.x_array)
    assert jax_objective.policy_theta_dim() == objective.policy_theta_dim()
    np.testing.assert_allclose(jax_objective.policy_value(theta, jax_batch.x_array), objective.policy_value(theta, x))
    assert jax_objective.value(theta, jax_batch.x_array) == pytest.approx(objective.value(theta, x), rel=1e-10)
    assert jax_objective.base_value(theta, jax_batch.x_array) == pytest.approx(
        prepared.base_value(theta, batch.x_array), rel=1e-10
    )
    assert jax_objective.mean_acceptance(theta, jax_batch.x_array) == pytest.approx(
        objective.mean_acceptance(theta, x), rel=1e-10
    )
    assert jax_objective.value_at_u(jax_batch.x_array, 0.02) == pytest.approx(
        objective.value_at_u(x, 0.02), rel=1e-10
    )
    u_arr = np.linspace(-0.05, 0.08, x.shape[0])
    np.testing.assert_allclose(
        jax_objective._value_batch(jax_batch.x_array, u_arr),
        objective._value_batch(x, u_arr),
        rtol=1e-10,
    )
    u_matrix = np.vstack([u_arr, u_arr + 0.01])
    np.testing.assert_allclose(
        jax_objective._value_batch_many(jax_batch.x_array, u_matrix),
        np.vstack([objective._value_batch(x, u_row) for u_row in u_matrix]),
        rtol=1e-10,
    )
    weights = np.linspace(-0.25, 0.5, x.shape[0])
    np.testing.assert_allclose(
        jax_objective.policy_weighted_grad(theta, jax_batch.x_array, weights),
        prepared.policy_weighted_grad(theta, batch.x_array, weights),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(jax_objective.grad(theta, jax_batch.x_array), objective.grad(theta, x), rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(
        jax_objective.mean_acceptance_grad(theta, jax_batch.x_array),
        objective.mean_acceptance_grad(theta, x),
        rtol=1e-9,
        atol=1e-8,
    )


def test_jax_scipy_adapter_constraint_margin_shapes() -> None:
    objective, x, theta = _make_glm_objective()
    floor = objective.mean_acceptance(theta, x) - 0.01
    constrained = replace(objective, acceptance_floor=floor)
    jax_objective, batch = prepare_jax_glm_objective(constrained, x)
    adapter = jax_objective.scipy_adapter()

    assert adapter.objective_value(theta) == pytest.approx(constrained.value(theta, x), rel=1e-10)
    np.testing.assert_allclose(adapter.objective_grad(theta), constrained.grad(theta, x), rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(adapter.constraint(theta), np.asarray([constrained.mean_acceptance(theta, x) - floor]))
    assert adapter.constraint_jac(theta).shape == (1, theta.size)
    np.testing.assert_allclose(
        adapter.constraint_jac(theta)[0],
        constrained.mean_acceptance_grad(theta, x),
        rtol=1e-9,
        atol=1e-8,
    )
    assert jax_objective.constraint_margin(theta) == pytest.approx(
        jax_objective.mean_acceptance(theta, batch.x_array) - floor
    )
