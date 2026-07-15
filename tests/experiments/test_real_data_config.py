"""Tests for real-data experiment config factory presets."""

from __future__ import annotations

import numpy as np
import pytest


def _cfg(name: str, **overrides):
    from experiments.configs import get_config

    return get_config(
        name,
        overrides={
            "n_samples": 25,
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
            **overrides,
        },
    )


@pytest.mark.parametrize("kwargs", [{}, {"n_samples": None}])
def test_real_data_config_uses_all_eligible_rows_when_n_samples_omitted_or_none(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
) -> None:
    import pandas as pd

    import experiments.configs.real_data_factory as factory
    from data.loader import FEATURE_COLS_GLM

    eligible = np.asarray([4, 1, 8], dtype=int)

    def fail_sample(*args, **kwargs):
        raise AssertionError("n_samples=None should not call sampled row selection")

    def fake_load_x_frame(model_type, n_rows=5000, *, row_indices=None, seed=None):
        del n_rows, seed
        assert model_type == "glm"
        np.testing.assert_array_equal(row_indices, eligible)
        return pd.DataFrame(
            np.zeros((eligible.size, len(FEATURE_COLS_GLM)), dtype=float),
            columns=FEATURE_COLS_GLM,
        )

    monkeypatch.setattr(factory, "eligible_csv_row_indices", lambda model_type: eligible.copy())
    monkeypatch.setattr(factory, "sample_csv_row_indices", fail_sample)
    monkeypatch.setattr(factory, "load_x_frame", fake_load_x_frame)
    monkeypatch.setattr(factory, "load_model_artifacts", lambda model_type: (object(), object()))
    monkeypatch.setattr(factory, "extract_glm_u_coef", lambda acceptance_model: -1.0)

    cfg = factory.build_real_data_config(
        model_type="glm",
        plot=False,
        verbose=False,
        wandb_enabled=False,
        **kwargs,
    )

    assert cfg.n_samples == eligible.size
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (eligible.size, len(FEATURE_COLS_GLM))
    np.testing.assert_array_equal(cfg.x_fixed_row_indices, eligible)


@pytest.mark.parametrize(
    "name",
    ["real_data_glm_base", "real_data_xgb_base", "real_data_xgb_logit_spline_base"],
)
def test_real_data_base_configs_load(name):
    cfg = _cfg(name)
    assert cfg is not None
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape[0] == cfg.n_samples
    assert cfg.x_fixed_row_indices is not None
    assert cfg.x_fixed_row_indices.shape == (cfg.n_samples,)
    assert np.unique(cfg.x_fixed_row_indices).shape == (cfg.n_samples,)


def test_old_real_data_preset_names_are_removed() -> None:
    from experiments.configs import get_config, list_configs

    assert "real_data_glm_softmax_policy_base" not in list_configs()
    with pytest.raises(ValueError, match="Unknown experiment config"):
        get_config("real_data_glm_softmax_policy_base")


def test_glm_base_default_shape_and_first_order():
    cfg = _cfg("real_data_glm_base")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 19)
    assert cfg.state_dim == 19
    assert "first_order" in cfg.enabled_estimators


def test_xgb_base_default_shape_and_no_first_order():
    cfg = _cfg("real_data_xgb_base")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 19)
    assert cfg.state_dim == 19
    assert "first_order" not in cfg.enabled_estimators


def test_xgb_logit_spline_base_is_covered_bounded_and_first_order() -> None:
    from objective.policy import SoftmaxPolicy

    cfg = _cfg("real_data_xgb_logit_spline_base")

    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 20)
    assert list(cfg.x_fixed.columns)[0] == "id"
    assert cfg.state_dim == 19
    assert cfg.objective.u_bounds == (0.0, 0.16)
    assert "first_order" in cfg.enabled_estimators
    assert isinstance(cfg.objective.policy, SoftmaxPolicy)
    assert cfg.objective.policy.action_low == pytest.approx(0.0)
    assert cfg.objective.policy.action_high == pytest.approx(0.16)
    acceptance = cfg.objective.mean_acceptance(cfg.theta0, cfg.x_fixed)
    assert 0.0 <= acceptance <= 1.0


def test_xgb_logit_spline_uses_all_200_covered_rows_by_default() -> None:
    from experiments.configs import get_config

    cfg = get_config(
        "real_data_xgb_logit_spline_base",
        overrides={"plot": False, "verbose": False, "wandb_enabled": False},
    )

    assert cfg.n_samples == 200
    assert cfg.x_fixed.shape == (200, 20)
    assert cfg.x_fixed["id"].nunique() == 200


def test_xgb_logit_spline_rejects_jax_backend() -> None:
    with pytest.raises(ValueError, match="only compute_backend='numpy'"):
        _cfg("real_data_xgb_logit_spline_base", compute_backend="jax")


@pytest.mark.parametrize(
    ("policy_kind", "feature_order", "policy_type", "feature_map_type", "theta0_is_none"),
    [
        ("linear", "linear", "linear", "identity", True),
        ("linear", "quadratic", "linear", "quadratic", True),
        ("linear", "cubic", "linear", "cubic", True),
        ("linear", "quartic", "linear", "quartic", True),
        ("softmax", "linear", "softmax", "identity", False),
        ("softmax", "quadratic", "softmax", "quadratic", False),
        ("softmax", "cubic", "softmax", "cubic", False),
        ("softmax", "quartic", "softmax", "quartic", False),
    ],
)
def test_glm_policy_feature_overrides(
    policy_kind,
    feature_order,
    policy_type,
    feature_map_type,
    theta0_is_none,
):
    from objective.policy import (
        CubicFeatureMap,
        IdentityFeatureMap,
        LinearPolicy,
        QuadraticFeatureMap,
        QuarticFeatureMap,
        SoftmaxPolicy,
    )

    cfg = _cfg(
        "real_data_glm_base",
        policy_kind=policy_kind,
        feature_order=feature_order,
    )
    policy = cfg.objective.policy
    policy_classes = {"linear": LinearPolicy, "softmax": SoftmaxPolicy}
    feature_map_classes = {
        "identity": IdentityFeatureMap,
        "quadratic": QuadraticFeatureMap,
        "cubic": CubicFeatureMap,
        "quartic": QuarticFeatureMap,
    }

    assert isinstance(policy, policy_classes[policy_type])
    assert isinstance(policy.feature_map, feature_map_classes[feature_map_type])
    assert (cfg.theta0 is None) is theta0_is_none
    if cfg.theta0 is not None:
        assert cfg.theta0.size == cfg.objective.policy_theta_dim()


def test_glm_constant_policy_override():
    from objective.policy import ConstantPolicy

    cfg = _cfg("real_data_glm_base", policy_kind="constant", seed=8)
    assert isinstance(cfg.objective.policy, ConstantPolicy)
    assert cfg.theta0 is not None
    assert cfg.theta0.shape == (1,)


def test_glm_softmax_action_bounds_override_sets_policy_bounds() -> None:
    from objective.policy import SoftmaxPolicy

    cfg = _cfg(
        "real_data_glm_base",
        policy_kind="softmax",
        softmax_action_bounds=(-0.1, 0.2),
    )

    policy = cfg.objective.policy
    assert isinstance(policy, SoftmaxPolicy)
    assert policy.action_low == pytest.approx(-0.1)
    assert policy.action_high == pytest.approx(0.2)
    assert cfg.theta0 is not None
    u_batch = cfg.objective.policy_value(cfg.theta0, cfg.x_fixed)
    assert np.all(u_batch > -0.1)
    assert np.all(u_batch < 0.2)
    np.testing.assert_allclose(u_batch, 0.05)


def test_glm_softmax_initial_u_uses_action_scale() -> None:
    cfg = _cfg(
        "real_data_glm_base",
        policy_kind="softmax",
        softmax_action_bounds=(-0.1, 0.2),
        initial_u=0.0,
    )

    assert cfg.theta0 is not None
    u_batch = cfg.objective.policy_value(cfg.theta0, cfg.x_fixed)
    np.testing.assert_allclose(u_batch, 0.0, atol=1e-12)


def test_glm_softmax_initial_u_must_be_inside_bounds() -> None:
    with pytest.raises(ValueError, match="initial_u"):
        _cfg(
            "real_data_glm_base",
            policy_kind="softmax",
            softmax_action_bounds=(-0.1, 0.2),
            initial_u=0.2,
        )


def test_softmax_action_bounds_rejected_for_non_softmax_policy() -> None:
    with pytest.raises(ValueError, match="softmax_action_bounds"):
        _cfg(
            "real_data_glm_base",
            policy_kind="linear",
            softmax_action_bounds=(-0.1, 0.2),
        )


def test_glm_u_coef_override_sets_acceptance_coefficient() -> None:
    cfg = _cfg("real_data_glm_base", u_coef=-5.0)
    assert cfg.objective.u_coef == pytest.approx(-5.0)


def test_glm_observed_loss_source_override_adds_y_g_loss() -> None:
    from data.loader import LOSS_TARGET_COL

    cfg = _cfg("real_data_glm_base", loss_source="observed")

    assert cfg.objective.loss_source == "observed"
    assert cfg.objective.observed_loss_col == LOSS_TARGET_COL
    assert cfg.x_fixed is not None
    assert cfg.state_dim == 19
    assert cfg.x_fixed.shape == (cfg.n_samples, cfg.state_dim + 1)
    assert LOSS_TARGET_COL in cfg.x_fixed.columns
    assert cfg.to_dict()["objective"]["loss_source"] == "observed"


def test_invalid_real_data_loss_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="loss_source"):
        _cfg("real_data_glm_base", loss_source="historical")


def test_xgb_u_coef_override_is_rejected() -> None:
    with pytest.raises(ValueError, match="GLM acceptance"):
        _cfg("real_data_xgb_base", u_coef=-5.0)


def test_glm_mlp_policy_override_has_mlp_policy_and_first_order():
    from objective.policy import IdentityFeatureMap, MLPPolicy

    cfg = _cfg("real_data_glm_base", policy_kind="mlp")
    policy = cfg.objective.policy
    assert isinstance(policy, MLPPolicy)
    assert isinstance(policy.feature_map, IdentityFeatureMap)
    assert policy.hidden == 16
    assert cfg.enabled_estimators == ("first_order", "spsa", "stein_difference")
    assert cfg.theta0 is not None
    assert cfg.theta0.size == cfg.objective.policy_theta_dim()


def test_glm_no_pca_policy_preprocessing_override() -> None:
    cfg = _cfg(
        "real_data_glm_base",
        policy_kind="softmax",
        feature_order="quartic",
        policy_preprocessing="no_pca",
    )

    assert cfg.objective.policy_preprocessor is not None
    assert cfg.objective.policy_feature_cols is None
    assert cfg.objective.policy_input_dim() == cfg.objective.policy_preprocessor.output_dim_
    assert cfg.theta0 is not None
    assert cfg.theta0.size == cfg.objective.policy_theta_dim()


def test_glm_trust_constraint_override_sets_acceptance_floor():
    cfg = _cfg(
        "real_data_glm_base",
        constraint_mode="trust_constr",
        enabled_estimators=("first_order", "constant"),
    )
    assert cfg.step_rule == "trust-constr"
    assert cfg.acceptance_floor is not None
    assert cfg.acceptance_penalty_weight is None
    assert cfg.initial_constr_penalty == 1.0
    assert cfg.batch_size is None
    assert cfg.enabled_estimators == ("first_order", "constant")


def test_glm_jax_backend_override_keeps_trust_constr_solver():
    from objective.policy import QuadraticFeatureMap

    cfg = _cfg(
        "real_data_glm_base",
        constraint_mode="trust_constr",
        feature_order="quadratic",
        compute_backend="jax",
        enabled_estimators=("first_order", "finite_difference", "stein_difference"),
    )

    assert cfg.compute_backend == "jax"
    assert cfg.step_rule == "trust-constr"
    assert cfg.acceptance_floor is not None
    assert cfg.enabled_estimators == ("first_order", "finite_difference", "stein_difference")
    assert isinstance(cfg.objective.policy.feature_map, QuadraticFeatureMap)
    assert cfg.theta0 is not None
    assert cfg.theta0.size == cfg.objective.policy_theta_dim()


def test_glm_lagrangian_override_sets_scalarization():
    cfg = _cfg(
        "real_data_glm_base",
        constraint_mode="lagrangian",
        n_samples=20,
        t_steps=50,
        n_grad_samples=8,
        lagrangian_lambda=250.0,
    )
    assert cfg.step_rule == "l-bfgs-b"
    assert cfg.acceptance_floor is not None
    assert cfg.lagrangian_lambda == 250.0
    assert cfg.acceptance_penalty_weight is None


def test_xgb_linear_policy_initial_action_override_is_constant_0_0():
    cfg = _cfg("real_data_xgb_base", policy_kind="linear", seed=8, initial_u=0.0)
    assert cfg.x_fixed is not None
    assert cfg.theta0 is not None
    u_batch = cfg.objective.policy_value(cfg.theta0, cfg.x_fixed)
    assert np.allclose(u_batch, 0.0)


def test_xgb_penalty_constraint_override():
    cfg = _cfg(
        "real_data_xgb_base",
        policy_kind="linear",
        constraint_mode="penalty",
        seed=8,
        initial_u=0.2,
    )
    assert cfg.step_rule == "l-bfgs-b"
    assert cfg.acceptance_floor is not None
    assert cfg.acceptance_penalty_weight == 1e4
    assert cfg.acceptance_penalty_temperature == 0.05
    assert cfg.objective.u_bounds == (-0.05, 0.5)
    assert cfg.theta0 is not None
    assert cfg.theta0[0] == 0.2
