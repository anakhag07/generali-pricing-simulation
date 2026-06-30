"""JAX prepared GLM objective parity tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

import objective.objectives.jax_prepared_glm as jax_glm  # noqa: E402
from data.loader import (  # noqa: E402
    ACCEPTANCE_STATE_COLS,
    LOSS_FEATURE_COLS,
    PREMIUM_COL,
    extract_glm_u_coef,
    load_model_artifacts,
    load_x_frame,
)
from objective.objectives.jax_prepared_glm import (  # noqa: E402
    JaxPreparedGLMObjective,
    prepare_jax_glm_objective,
)
from objective.objectives.model_based import ModelBasedObjective  # noqa: E402
from objective.objectives.prepared_glm import (  # noqa: E402
    PreparedGLMBatch,
    PreparedGLMObjective,
    prepare_glm_objective,
)
from objective.policy import (  # noqa: E402
    CallableFeatureMap,
    CubicFeatureMap,
    LinearPolicy,
    QuadraticFeatureMap,
    QuarticFeatureMap,
    SoftmaxPolicy,
)


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


def _callable_feature_values(x_batch: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            x_batch[:, 0] + 2.0 * x_batch[:, 1],
            x_batch[:, 0] * x_batch[:, 1],
            x_batch[:, 1] ** 2,
        ]
    )


def _synthetic_prepared_batch() -> PreparedGLMBatch:
    rng = np.random.default_rng(2026)
    policy_features = rng.normal(size=(11, 2))
    return PreparedGLMBatch.from_arrays(
        base_logit=0.15 + 0.2 * policy_features[:, 0] - 0.1 * policy_features[:, 1],
        loss=115.0 + 4.0 * policy_features[:, 0] + 2.5 * policy_features[:, 1],
        premium=95.0 + policy_features[:, 0],
        policy_features=policy_features,
        u_coef=-2.75,
    )


def _theta_for_policy(policy: object, state_dim: int) -> np.ndarray:
    return np.linspace(-0.08, 0.09, policy.theta_dim(state_dim), dtype=float)


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


@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(LinearPolicy(feature_map=QuadraticFeatureMap()), id="linear-quadratic"),
        pytest.param(LinearPolicy(feature_map=CubicFeatureMap()), id="linear-cubic"),
        pytest.param(LinearPolicy(feature_map=QuarticFeatureMap()), id="linear-quartic"),
        pytest.param(
            LinearPolicy(
                feature_map=CallableFeatureMap(
                    _callable_feature_values,
                    feature_dim=3,
                    name="custom3",
                )
            ),
            id="linear-callable",
        ),
        pytest.param(
            SoftmaxPolicy(feature_map=QuadraticFeatureMap(), action_low=-0.1, action_high=0.2),
            id="softmax-quadratic",
        ),
        pytest.param(
            SoftmaxPolicy(feature_map=CubicFeatureMap(), action_low=-0.1, action_high=0.2),
            id="softmax-cubic",
        ),
        pytest.param(
            SoftmaxPolicy(feature_map=QuarticFeatureMap(), action_low=-0.1, action_high=0.2),
            id="softmax-quartic",
        ),
        pytest.param(
            SoftmaxPolicy(
                feature_map=CallableFeatureMap(
                    _callable_feature_values,
                    feature_dim=3,
                    name="custom3",
                ),
                action_low=-0.1,
                action_high=0.2,
            ),
            id="softmax-callable",
        ),
    ],
)
def test_jax_prepared_glm_feature_maps_match_numpy_prepared(policy) -> None:
    batch = _synthetic_prepared_batch()
    cpu_objective = PreparedGLMObjective(
        policy=policy,
        policy_feature_dim=batch.policy_feature_dim,
        u_coef=batch.u_coef,
    )
    jax_objective = JaxPreparedGLMObjective(
        policy=policy,
        x_array=batch.x_array,
        u_coef=batch.u_coef,
    )
    theta = _theta_for_policy(policy, batch.policy_feature_dim)
    weights = np.linspace(-0.3, 0.4, batch.n_rows)

    jax_objective.warmup(theta)

    assert jax_objective.policy_theta_dim() == cpu_objective.policy_theta_dim()
    assert jax_objective.policy_theta_dim() == theta.size
    np.testing.assert_allclose(
        jax_objective.policy_value(theta, batch.x_array),
        cpu_objective.policy_value(theta, batch.x_array),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        jax_objective.policy_grad(theta, batch.x_array),
        cpu_objective.policy_grad(theta, batch.x_array),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        jax_objective.policy_weighted_grad(theta, batch.x_array, weights),
        cpu_objective.policy_weighted_grad(theta, batch.x_array, weights),
        rtol=1e-10,
        atol=1e-10,
    )
    assert jax_objective.value(theta, batch.x_array) == pytest.approx(
        cpu_objective.value(theta, batch.x_array), rel=1e-10, abs=1e-10
    )
    assert jax_objective.base_value(theta, batch.x_array) == pytest.approx(
        cpu_objective.base_value(theta, batch.x_array), rel=1e-10, abs=1e-10
    )
    assert jax_objective.mean_acceptance(theta, batch.x_array) == pytest.approx(
        cpu_objective.mean_acceptance(theta, batch.x_array), rel=1e-10, abs=1e-10
    )
    np.testing.assert_allclose(
        jax_objective.grad(theta, batch.x_array),
        cpu_objective.grad(theta, batch.x_array),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        jax_objective.mean_acceptance_grad(theta, batch.x_array),
        cpu_objective.mean_acceptance_grad(theta, batch.x_array),
        rtol=1e-9,
        atol=1e-9,
    )


def test_jax_prepared_glm_kernels_take_design_runtime_argument() -> None:
    batch = _synthetic_prepared_batch()
    policy = LinearPolicy(feature_map=QuadraticFeatureMap())
    objective = JaxPreparedGLMObjective(
        policy=policy,
        x_array=batch.x_array,
        u_coef=batch.u_coef,
    )
    theta = _theta_for_policy(policy, batch.policy_feature_dim)
    theta_jax = objective._theta_key_and_jax(theta)[1]

    baseline = np.asarray(
        objective._policy_u_jit(
            theta_jax,
            objective._base_logit_jax,
            objective._design_jax,
        ),
        dtype=float,
    )
    zero_design = jnp.zeros_like(objective._design_jax)
    altered = np.asarray(
        objective._policy_u_jit(theta_jax, objective._base_logit_jax, zero_design),
        dtype=float,
    )
    altered_objective, altered_objective_grad = objective._objective_value_and_grad_jit(
        theta_jax,
        objective._base_logit_jax,
        objective._loss_jax,
        objective._premium_jax,
        zero_design,
    )
    altered_acceptance, altered_acceptance_grad = objective._mean_acceptance_value_and_grad_jit(
        theta_jax,
        objective._base_logit_jax,
        zero_design,
    )

    assert not np.allclose(baseline, altered)
    np.testing.assert_allclose(altered, np.zeros(batch.n_rows), atol=1e-12)
    assert float(altered_objective) != pytest.approx(objective.value(theta, batch.x_array))
    assert float(altered_acceptance) != pytest.approx(
        objective.mean_acceptance(theta, batch.x_array)
    )
    np.testing.assert_allclose(altered_objective_grad, np.zeros(theta.size), atol=1e-12)
    np.testing.assert_allclose(altered_acceptance_grad, np.zeros(theta.size), atol=1e-12)


def test_prepared_design_memory_summary_counts_design_bytes() -> None:
    design = np.zeros((4, 7), dtype=np.float64)
    summary = jax_glm._prepared_design_memory_summary(design)

    assert summary["design_shape"] == (4, 7)
    assert summary["design_dtype"] == "float64"
    assert summary["design_nbytes"] == design.nbytes
    assert summary["design_gb"] == pytest.approx(design.nbytes / 1e9)
    assert jax_glm._prepared_design_memory_summary(None)["design_nbytes"] == 0


def test_prepared_design_memory_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jax_glm, "_LARGE_DESIGN_WARN_BYTES", 1)

    with pytest.warns(RuntimeWarning, match="large policy design matrix"):
        jax_glm._warn_if_large_prepared_design(np.zeros((2, 3), dtype=float))


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
