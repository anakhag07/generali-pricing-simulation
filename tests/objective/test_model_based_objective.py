"""Tests for ModelBasedObjective."""

from dataclasses import replace

import numpy as np
import pytest

from objective.utils import _mean_action
from optimization.helpers import finite_difference_theta_grad


def _make_glm_objective(n_rows=20):
    """Create a ModelBasedObjective with GLM models and a small x_batch fixture."""
    from data.loader import (
        ACCEPTANCE_STATE_COLS,
        LOSS_FEATURE_COLS,
        extract_glm_u_coef,
        load_model_artifacts,
        load_x_array,
    )
    from objective.objectives.model_based import ModelBasedObjective
    from objective.policy import SoftmaxPolicy

    acc_model, loss_model = load_model_artifacts("glm")
    u_coef = extract_glm_u_coef(acc_model)
    x = load_x_array("glm", n_rows=n_rows, seed=123)
    obj = ModelBasedObjective(
        policy=SoftmaxPolicy(),
        acceptance_model=acc_model,
        loss_model=loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=9,
        u_coef=u_coef,
    )
    return obj, x, obj.policy_theta_dim()


def test_value_returns_scalar():
    obj, x, theta_dim = _make_glm_objective()
    theta = np.zeros(theta_dim, dtype=float)
    val = obj.value(theta, x)
    assert isinstance(val, float)
    assert np.isfinite(val)


def test_value_records_eval_counts() -> None:
    obj, x, theta_dim = _make_glm_objective()
    theta = np.zeros(theta_dim, dtype=float)

    obj.reset_eval_counts()
    obj.value(theta, x)
    counts = obj.eval_counts()

    assert counts["objective_value_calls"] == 1
    assert counts["objective_value_calls_rows"] == x.shape[0]
    assert counts["acceptance_predict_calls"] >= 1
    assert counts["loss_predict_calls"] >= 1


def test_grad_shape():
    obj, x, theta_dim = _make_glm_objective()
    theta = np.zeros(theta_dim, dtype=float)
    grad = obj.grad(theta, x)
    assert grad.shape == theta.shape
    assert np.all(np.isfinite(grad))


def test_value_at_u_returns_scalar():
    obj, x, _ = _make_glm_objective()
    val = obj.value_at_u(x, 0.0)
    assert isinstance(val, float)
    assert np.isfinite(val)


def test_value_at_u_consistent_with_value():
    """value_at_u should match value when policy is ConstantPolicy at that u."""
    from objective.policy import ConstantPolicy
    from data.loader import (
        ACCEPTANCE_STATE_COLS,
        LOSS_FEATURE_COLS,
        extract_glm_u_coef,
        load_model_artifacts,
        load_x_array,
    )
    from objective.objectives.model_based import ModelBasedObjective

    acc_model, loss_model = load_model_artifacts("glm")
    u_coef = extract_glm_u_coef(acc_model)
    x = load_x_array("glm", n_rows=20, seed=123)

    obj = ModelBasedObjective(
        policy=ConstantPolicy(),
        acceptance_model=acc_model,
        loss_model=loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=9,
        u_coef=u_coef,
    )
    u_val = 0.0
    theta = np.array([u_val], dtype=float)
    assert abs(obj.value(theta, x) - obj.value_at_u(x, u_val)) < 1e-8


def test_value_at_u_uses_shifted_revenue_multiplier() -> None:
    from objective.objectives.model_based import ModelBasedObjective
    from objective.policy import ConstantPolicy

    class ConstantAcceptanceModel:
        def predict_proba(self, x_frame):
            churn = np.full(len(x_frame), 0.25, dtype=float)
            return np.column_stack([1.0 - churn, churn])

    class ConstantLossModel:
        def predict(self, x_frame):
            return np.full(len(x_frame), 10.0, dtype=float)

    x = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
        ],
        dtype=float,
    )
    obj = ModelBasedObjective(
        policy=ConstantPolicy(),
        acceptance_model=ConstantAcceptanceModel(),
        loss_model=ConstantLossModel(),
        acceptance_state_cols=tuple(f"x{i}" for i in range(10)),
        loss_cols=tuple(f"x{i}" for i in range(9)),
        premium_col=9,
        u_coef=0.0,
    )

    expected_value = 0.75 * (10.0 - 3.0)
    assert obj.value_at_u(x, 0.0) == pytest.approx(expected_value)


def test_analytical_vs_fd_grad_glm():
    """Analytical GLM gradient should closely match central FD gradient."""
    from data.loader import (
        ACCEPTANCE_STATE_COLS,
        LOSS_FEATURE_COLS,
        extract_glm_u_coef,
        load_model_artifacts,
        load_x_array,
    )
    from objective.objectives.model_based import ModelBasedObjective
    from objective.policy import SoftmaxPolicy

    acc_model, loss_model = load_model_artifacts("glm")
    u_coef = extract_glm_u_coef(acc_model)
    x = load_x_array("glm", n_rows=30, seed=123)
    policy = SoftmaxPolicy()
    theta = np.zeros(policy.theta_dim(acc_model.policy_feature_dim()), dtype=float)

    obj_analytical = ModelBasedObjective(
        policy=policy,
        acceptance_model=acc_model,
        loss_model=loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=9,
        u_coef=u_coef,
    )
    obj_fd = ModelBasedObjective(
        policy=policy,
        acceptance_model=acc_model,
        loss_model=loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=9,
        u_coef=None,  # numerical FD
    )

    grad_a = obj_analytical.grad(theta, x)
    grad_fd = obj_fd.grad(theta, x)
    # Should agree within 5% relative tolerance on the dominant components
    np.testing.assert_allclose(grad_a, grad_fd, rtol=0.05, atol=1e-3)


def test_x_batch_must_be_2d():
    obj, _, theta_dim = _make_glm_objective()
    theta = np.zeros(theta_dim, dtype=float)
    with pytest.raises(ValueError):
        obj.value(theta, np.zeros(12))


def test_policy_hooks_use_acceptance_preprocessor() -> None:
    obj, x, _ = _make_glm_objective(n_rows=5)
    theta = np.zeros(obj.policy_theta_dim(), dtype=float)

    processed = obj._policy_features(x)
    direct_u = obj.policy.value(theta, processed)
    hook_u = obj.policy_value(theta, x)
    hook_grad = obj.policy_grad(theta, x)
    direct_grad = obj.policy.grad(theta, processed)

    assert processed.shape[1] == obj.acceptance_model.policy_feature_dim()
    assert np.allclose(hook_u, direct_u)
    assert np.allclose(hook_grad, direct_grad)


def test_policy_hooks_can_use_independent_policy_preprocessor() -> None:
    from data.loader import ACCEPTANCE_STATE_COLS
    from objective.policy_preprocessing import fit_policy_feature_preprocessor

    obj, x, _ = _make_glm_objective(n_rows=20)
    x_policy = x[:, : len(ACCEPTANCE_STATE_COLS)]
    policy_preprocessor = fit_policy_feature_preprocessor(x_policy, pca_dim=4)
    obj = replace(
        obj,
        policy_preprocessor=policy_preprocessor,
        policy_feature_cols=tuple(ACCEPTANCE_STATE_COLS),
    )
    theta = np.zeros(obj.policy_theta_dim(), dtype=float)

    processed = obj._policy_features(x)
    direct_u = obj.policy.value(theta, processed)

    assert processed.shape == (x.shape[0], 4)
    assert obj.policy_input_dim() == 4
    np.testing.assert_allclose(processed, policy_preprocessor.transform(x_policy))
    np.testing.assert_allclose(obj.policy_value(theta, x), direct_u)


def test_policy_preprocessor_does_not_change_black_box_acceptance_path() -> None:
    from data.loader import ACCEPTANCE_STATE_COLS
    from objective.policy_preprocessing import fit_policy_feature_preprocessor

    obj, x, _ = _make_glm_objective(n_rows=20)
    x_policy = x[:, : len(ACCEPTANCE_STATE_COLS)]
    policy_preprocessor = fit_policy_feature_preprocessor(x_policy, pca_dim=4)
    obj_with_policy_preprocessing = replace(
        obj,
        policy_preprocessor=policy_preprocessor,
        policy_feature_cols=tuple(ACCEPTANCE_STATE_COLS),
    )

    np.testing.assert_allclose(
        obj.mean_acceptance_at_u(x, 0.0),
        obj_with_policy_preprocessing.mean_acceptance_at_u(x, 0.0),
    )


def test_policy_hooks_support_quadratic_feature_map() -> None:
    from dataclasses import replace

    from objective.policy import QuadraticFeatureMap, SoftmaxPolicy

    obj, x, _ = _make_glm_objective(n_rows=5)
    quadratic_policy = SoftmaxPolicy(feature_map=QuadraticFeatureMap())
    obj = replace(obj, policy=quadratic_policy)
    theta = np.zeros(obj.policy_theta_dim(), dtype=float)

    processed = obj._policy_features(x)
    direct_u = quadratic_policy.value(theta, processed)
    direct_grad = quadratic_policy.grad(theta, processed)

    assert theta.size == quadratic_policy.theta_dim(processed.shape[1])
    np.testing.assert_allclose(obj.policy_value(theta, x), direct_u)
    np.testing.assert_allclose(obj.policy_grad(theta, x), direct_grad)


def test_mean_acceptance_grad_matches_fd() -> None:
    obj, x, theta_dim = _make_glm_objective(n_rows=30)
    theta = np.array([0.4] + [0.01] * (theta_dim - 1), dtype=float)

    grad = obj.mean_acceptance_grad(theta, x)
    grad_fd = finite_difference_theta_grad(
        lambda theta_eval: obj.mean_acceptance(theta_eval, x),
        theta,
        method="central",
        step=1e-6,
    )
    np.testing.assert_allclose(grad, grad_fd, rtol=1e-4, atol=1e-6)


def test_step_metrics_match_objective_components() -> None:
    obj, x, theta_dim = _make_glm_objective(n_rows=30)
    theta = np.array([0.4] + [0.01] * (theta_dim - 1), dtype=float)

    metrics = obj._step_metrics(theta, x)
    u_batch = obj._clip_u(obj.policy_value(theta, x))
    premium = x[:, obj.premium_col]

    assert metrics["mean_acceptance"] == pytest.approx(obj.mean_acceptance(theta, x))
    assert metrics["projected_loss"] == pytest.approx(float(np.mean(obj._loss_prediction(x))))
    assert metrics["projected_revenue"] == pytest.approx(float(np.mean((u_batch + 1.0) * premium)))


def test_mean_action_uses_clipped_u_when_bounds_present() -> None:
    from data.loader import ACCEPTANCE_STATE_COLS, LOSS_FEATURE_COLS, load_model_artifacts, load_x_array
    from objective.objectives.model_based import ModelBasedObjective
    from objective.policy import LinearPolicy

    acc_model, loss_model = load_model_artifacts("xgb")
    x = load_x_array("xgb", n_rows=30, seed=123)
    obj = ModelBasedObjective(
        policy=LinearPolicy(),
        acceptance_model=acc_model,
        loss_model=loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=9,
        u_coef=None,
        u_bounds=(-0.05, 0.5),
    )
    theta = np.array([10.0] + [0.0] * acc_model.policy_feature_dim(), dtype=float)

    assert _mean_action(obj, theta, x) == pytest.approx(0.5)


def test_acceptance_penalty_raises_value_when_floor_is_higher() -> None:
    obj, x, theta_dim = _make_glm_objective(n_rows=30)
    theta = np.array([0.4] + [0.01] * (theta_dim - 1), dtype=float)
    baseline_acceptance = obj.mean_acceptance(theta, x)

    constrained = replace(
        obj,
        acceptance_floor=baseline_acceptance + 0.01,
        acceptance_penalty_weight=100.0,
        acceptance_penalty_temperature=1e-4,
    )
    assert constrained.value(theta, x) > obj.value(theta, x)


def test_constrained_grad_matches_fd() -> None:
    obj, x, theta_dim = _make_glm_objective(n_rows=30)
    theta = np.array([0.4] + [0.01] * (theta_dim - 1), dtype=float)
    baseline_acceptance = obj.mean_acceptance(theta, x)
    constrained = replace(
        obj,
        acceptance_floor=baseline_acceptance + 0.01,
        acceptance_penalty_weight=100.0,
        acceptance_penalty_temperature=1e-4,
    )

    grad = constrained.grad(theta, x)
    grad_fd = finite_difference_theta_grad(
        lambda theta_eval: constrained.value(theta_eval, x),
        theta,
        method="central",
        step=1e-6,
    )
    np.testing.assert_allclose(grad, grad_fd, rtol=1e-4, atol=1e-6)


def test_lagrangian_value_matches_base_plus_lambda_gap() -> None:
    from data.loader import load_mean_observed_acceptance

    obj, x, theta_dim = _make_glm_objective(n_rows=30)
    theta = np.array([0.4] + [0.01] * (theta_dim - 1), dtype=float)
    floor = load_mean_observed_acceptance("glm")
    lagrangian = replace(
        obj,
        acceptance_floor=floor,
        lagrangian_lambda=2.0,
    )

    base_value = obj.base_value(theta, x)
    mean_acceptance = obj.mean_acceptance(theta, x)
    expected = base_value + 2.0 * (floor - mean_acceptance)

    assert lagrangian.value(theta, x) == pytest.approx(expected)


def test_lagrangian_grad_matches_fd() -> None:
    from data.loader import load_mean_observed_acceptance

    obj, x, theta_dim = _make_glm_objective(n_rows=30)
    theta = np.array([0.4] + [0.01] * (theta_dim - 1), dtype=float)
    lagrangian = replace(
        obj,
        acceptance_floor=load_mean_observed_acceptance("glm"),
        lagrangian_lambda=2.0,
    )

    grad = lagrangian.grad(theta, x)
    grad_fd = finite_difference_theta_grad(
        lambda theta_eval: lagrangian.value(theta_eval, x),
        theta,
        method="central",
        step=1e-6,
    )
    np.testing.assert_allclose(grad, grad_fd, rtol=1e-4, atol=1e-6)
