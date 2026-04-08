"""Tests for ModelBasedObjective."""

import numpy as np
import pytest


def _make_glm_objective(n_rows=20):
    """Create a ModelBasedObjective with GLM models and a small x_batch fixture."""
    from data.loader import (
        ACCEPTANCE_STATE_COLS,
        FEATURE_COLS_GLM,
        LOSS_FEATURE_COLS,
        extract_glm_u_coef,
        load_model_artifacts,
        load_x_array,
    )
    from objective.objectives.model_based import ModelBasedObjective
    from objective.policy import FeatureProcessedPolicy, SoftmaxPolicy

    acc_model, loss_model = load_model_artifacts("glm")
    u_coef = extract_glm_u_coef(acc_model)
    policy = FeatureProcessedPolicy(
        policy=SoftmaxPolicy(),
        raw_feature_cols=tuple(FEATURE_COLS_GLM),
        preprocess_feature_cols=acc_model.x_feature_cols,
        preprocessor=acc_model.preprocessor,
    )
    x = load_x_array("glm", n_rows=n_rows)
    obj = ModelBasedObjective(
        policy=policy,
        acceptance_model=acc_model,
        loss_model=loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=9,
        u_coef=u_coef,
    )
    return obj, x, acc_model.policy_feature_dim()


def test_value_returns_scalar():
    obj, x, policy_dim = _make_glm_objective()
    theta = np.array([0.4] + [0.01] * policy_dim, dtype=float)
    val = obj.value(theta, x)
    assert isinstance(val, float)
    assert np.isfinite(val)


def test_grad_shape():
    obj, x, policy_dim = _make_glm_objective()
    theta = np.array([0.4] + [0.01] * policy_dim, dtype=float)
    grad = obj.grad(theta, x)
    assert grad.shape == theta.shape
    assert np.all(np.isfinite(grad))


def test_value_at_u_returns_scalar():
    obj, x, _ = _make_glm_objective()
    val = obj.value_at_u(x, 1.1)
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
    x = load_x_array("glm", n_rows=20)

    obj = ModelBasedObjective(
        policy=ConstantPolicy(),
        acceptance_model=acc_model,
        loss_model=loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=9,
        u_coef=u_coef,
    )
    u_val = 1.1
    theta = np.array([u_val], dtype=float)
    assert abs(obj.value(theta, x) - obj.value_at_u(x, u_val)) < 1e-8


def test_analytical_vs_fd_grad_glm():
    """Analytical GLM gradient should closely match central FD gradient."""
    from data.loader import (
        ACCEPTANCE_STATE_COLS,
        FEATURE_COLS_GLM,
        LOSS_FEATURE_COLS,
        extract_glm_u_coef,
        load_model_artifacts,
        load_x_array,
    )
    from objective.objectives.model_based import ModelBasedObjective
    from objective.policy import FeatureProcessedPolicy, SoftmaxPolicy

    acc_model, loss_model = load_model_artifacts("glm")
    u_coef = extract_glm_u_coef(acc_model)
    x = load_x_array("glm", n_rows=30)
    policy = FeatureProcessedPolicy(
        policy=SoftmaxPolicy(),
        raw_feature_cols=tuple(FEATURE_COLS_GLM),
        preprocess_feature_cols=acc_model.x_feature_cols,
        preprocessor=acc_model.preprocessor,
    )
    theta = np.array([0.4] + [0.01] * acc_model.policy_feature_dim(), dtype=float)

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
    obj, _, policy_dim = _make_glm_objective()
    theta = np.array([0.4] + [0.01] * policy_dim, dtype=float)
    with pytest.raises(ValueError):
        obj.value(theta, np.zeros(12))
