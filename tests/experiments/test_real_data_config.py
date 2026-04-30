"""Tests for real-data experiment config presets."""

from dataclasses import replace

import numpy as np
import pytest


@pytest.mark.parametrize("name", [
    "real_data_glm_constant_policy_base",
    "real_data_glm_constant_policy_trust_region_constr",
    "real_data_glm_linear_policy_base",
    "real_data_glm_linear_policy_trust_region_constr",
    "real_data_glm_mlp_policy_base",
    "real_data_glm_softmax_policy_base",
    "real_data_glm_softmax_policy_lagrangian_small",
    "real_data_glm_softmax_policy_quadratic_base",
    "real_data_glm_softmax_policy_trust_region_constr",
    "real_data_xgb_base",
    "real_data_xgb_linear_acceptance_floor_base",
    "real_data_xgb_linear_policy_base",
    "real_data_xgb_softmax_policy_base",
])
def test_config_loads(name):
    from experiments.configs import get_config
    cfg = get_config(name)
    assert cfg is not None


def test_glm_softmax_policy_base_x_fixed_shape():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_softmax_policy_base")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 12)
    assert cfg.state_dim == 12


def test_glm_softmax_policy_trust_region_constr_x_fixed_shape():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_softmax_policy_trust_region_constr")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 12)
    assert cfg.state_dim == 12


def test_glm_softmax_policy_lagrangian_small_x_fixed_shape():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_softmax_policy_lagrangian_small")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 12)
    assert cfg.state_dim == 12


def test_glm_linear_policy_base_x_fixed_shape():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_policy_base")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 12)
    assert cfg.state_dim == 12


def test_glm_constant_policy_base_x_fixed_shape():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_constant_policy_base")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 12)
    assert cfg.state_dim == 12


def test_glm_constant_policy_trust_region_constr_x_fixed_shape():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_constant_policy_trust_region_constr")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 12)
    assert cfg.state_dim == 12


def test_xgb_base_x_fixed_shape():
    from experiments.configs import get_config
    cfg = get_config("real_data_xgb_base")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 10)
    assert cfg.state_dim == 10


def test_xgb_linear_acceptance_floor_base_x_fixed_shape():
    from experiments.configs import get_config

    cfg = get_config("real_data_xgb_linear_acceptance_floor_base")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 10)
    assert cfg.state_dim == 10


def test_xgb_linear_policy_base_x_fixed_shape():
    from experiments.configs import get_config

    cfg = get_config("real_data_xgb_linear_policy_base")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 10)
    assert cfg.state_dim == 10


def test_xgb_softmax_policy_base_x_fixed_shape():
    from experiments.configs import get_config

    cfg = get_config("real_data_xgb_softmax_policy_base")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 10)
    assert cfg.state_dim == 10


@pytest.mark.parametrize("name", [
    "real_data_glm_constant_policy_base",
    "real_data_glm_constant_policy_trust_region_constr",
    "real_data_glm_linear_policy_base",
    "real_data_glm_linear_policy_trust_region_constr",
    "real_data_glm_mlp_policy_base",
    "real_data_glm_softmax_policy_base",
    "real_data_glm_softmax_policy_lagrangian_small",
    "real_data_glm_softmax_policy_quadratic_base",
    "real_data_glm_softmax_policy_trust_region_constr",
    "real_data_xgb_base",
    "real_data_xgb_linear_acceptance_floor_base",
    "real_data_xgb_linear_policy_base",
    "real_data_xgb_softmax_policy_base",
])
def test_real_data_configs_store_sampled_row_indices(name):
    from experiments.configs import get_config

    cfg = get_config(name)
    assert cfg.x_fixed is not None
    assert cfg.x_fixed_row_indices is not None
    assert cfg.x_fixed.shape[0] == cfg.n_samples
    assert cfg.x_fixed_row_indices.shape == (cfg.n_samples,)
    assert np.unique(cfg.x_fixed_row_indices).shape == (cfg.n_samples,)


def test_glm_softmax_policy_base_has_first_order():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_softmax_policy_base")
    assert "first_order" in cfg.enabled_estimators


def test_glm_softmax_policy_base_uses_quadratic_feature_map():
    from experiments.configs import get_config
    from objective.policy import QuadraticFeatureMap

    cfg = get_config("real_data_glm_softmax_policy_base")
    policy = cfg.objective.policy
    assert isinstance(policy.feature_map, QuadraticFeatureMap)
    assert cfg.theta0 is not None
    assert cfg.theta0.size == cfg.objective.policy_theta_dim()


def test_glm_mlp_policy_base_has_mlp_policy_and_first_order():
    from experiments.configs import get_config
    from objective.policy import IdentityFeatureMap, MLPPolicy

    cfg = get_config("real_data_glm_mlp_policy_base")
    policy = cfg.objective.policy
    assert isinstance(policy, MLPPolicy)
    assert isinstance(policy.feature_map, IdentityFeatureMap)
    assert policy.hidden == 16
    assert "first_order" in cfg.enabled_estimators
    assert cfg.enabled_estimators == ("first_order", "spsa", "stein_difference")
    assert cfg.theta0 is not None
    assert cfg.theta0.size == cfg.objective.policy_theta_dim()
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (cfg.n_samples, 12)
    assert cfg.state_dim == 12
    assert cfg.step_rule == "l-bfgs-b"
    assert cfg.acceptance_floor is None


def test_glm_linear_policy_base_has_first_order():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_policy_base")
    assert "first_order" in cfg.enabled_estimators


def test_glm_linear_policy_base_theta0_is_resolved_at_runtime():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_policy_base")
    assert cfg.theta0 is None


def test_glm_constant_policy_base_is_unconstrained() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_constant_policy_base")
    assert cfg.step_rule == "l-bfgs-b"
    assert cfg.acceptance_floor is None
    assert cfg.acceptance_penalty_weight is None
    assert cfg.enabled_estimators == (
        "first_order",
        "finite_difference",
        "spsa",
        "stein_difference",
    )


def test_glm_softmax_policy_base_is_unconstrained() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_softmax_policy_base")
    assert cfg.step_rule == "l-bfgs-b"
    assert cfg.acceptance_floor is None
    assert cfg.enabled_estimators == (
        "first_order",
        "finite_difference",
        "spsa",
        "stein_difference",
    )


def test_glm_softmax_policy_trust_region_constr_sets_floor_from_csv_mean():
    from data.loader import load_mean_observed_acceptance
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_softmax_policy_trust_region_constr")
    assert cfg.acceptance_floor == pytest.approx(load_mean_observed_acceptance("glm"))


def test_glm_softmax_policy_lagrangian_small_sets_floor_from_csv_mean():
    from data.loader import load_mean_observed_acceptance
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_softmax_policy_lagrangian_small")
    assert cfg.acceptance_floor == pytest.approx(load_mean_observed_acceptance("glm"))
    assert cfg.lagrangian_lambda == pytest.approx(2.0)


def test_glm_softmax_policy_lagrangian_small_uses_all_estimators() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_softmax_policy_lagrangian_small")
    assert cfg.step_rule == "l-bfgs-b"
    assert cfg.enabled_estimators == (
        "first_order",
        "finite_difference",
        "gauss_stein",
        "spsa",
        "stein_difference",
    )


def test_glm_linear_policy_trust_region_constr_sets_floor_from_csv_mean():
    from data.loader import load_mean_observed_acceptance
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_policy_trust_region_constr")
    assert cfg.acceptance_floor == pytest.approx(load_mean_observed_acceptance("glm"))


def test_glm_constant_policy_trust_region_constr_sets_floor_from_csv_mean():
    from data.loader import load_mean_observed_acceptance
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_constant_policy_trust_region_constr")
    assert cfg.acceptance_floor == pytest.approx(load_mean_observed_acceptance("glm"))


def test_xgb_linear_acceptance_floor_base_sets_floor_from_csv_mean():
    from data.loader import load_mean_observed_acceptance
    from experiments.configs import get_config

    cfg = get_config("real_data_xgb_linear_acceptance_floor_base")
    assert cfg.acceptance_floor == pytest.approx(load_mean_observed_acceptance("xgb"))


def test_glm_softmax_policy_trust_region_constr_uses_trust_constr_with_selected_estimators():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_softmax_policy_trust_region_constr")
    assert cfg.step_rule == "trust-constr"
    assert cfg.enabled_estimators == (
        "first_order",
        "finite_difference",
        "spsa",
        "stein_difference",
    )


def test_glm_linear_policy_trust_region_constr_uses_trust_constr_with_selected_estimators():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_policy_trust_region_constr")
    assert cfg.step_rule == "trust-constr"
    assert cfg.enabled_estimators == (
        "first_order",
        "finite_difference",
        "spsa",
        "stein_difference",
    )


def test_glm_constant_policy_trust_region_constr_uses_trust_constr_with_selected_estimators():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_constant_policy_trust_region_constr")
    assert cfg.step_rule == "trust-constr"
    assert cfg.enabled_estimators == (
        "first_order",
        "finite_difference",
        "spsa",
        "stein_difference",
    )


def test_glm_constant_policy_trust_region_constr_initial_action_is_constant_0_0():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_constant_policy_trust_region_constr")
    assert cfg.x_fixed is not None
    u_batch = cfg.objective.policy_value(cfg.theta0, cfg.x_fixed)
    assert np.allclose(u_batch, 0.0)


def test_glm_constant_policy_base_initial_action_is_constant_0_0():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_constant_policy_base")
    assert cfg.x_fixed is not None
    u_batch = cfg.objective.policy_value(cfg.theta0, cfg.x_fixed)
    assert np.allclose(u_batch, 0.0)


def test_xgb_base_no_first_order():
    from experiments.configs import get_config
    cfg = get_config("real_data_xgb_base")
    assert "first_order" not in cfg.enabled_estimators


def test_xgb_linear_acceptance_floor_base_no_first_order():
    from experiments.configs import get_config

    cfg = get_config("real_data_xgb_linear_acceptance_floor_base")
    assert "first_order" not in cfg.enabled_estimators


@pytest.mark.parametrize("name", [
    "real_data_xgb_linear_policy_base",
    "real_data_xgb_softmax_policy_base",
])
def test_xgb_unconstrained_policy_bases_no_first_order(name):
    from experiments.configs import get_config

    cfg = get_config(name)
    assert "first_order" not in cfg.enabled_estimators


@pytest.mark.parametrize("name", [
    "real_data_xgb_linear_policy_base",
    "real_data_xgb_softmax_policy_base",
])
def test_xgb_unconstrained_policy_bases_are_unconstrained(name):
    from experiments.configs import get_config

    cfg = get_config(name)
    assert cfg.step_rule == "l-bfgs-b"
    assert cfg.acceptance_floor is None
    assert cfg.acceptance_penalty_weight is None
    assert cfg.enabled_estimators == (
        "finite_difference",
        "spsa",
        "stein_difference",
    )


def test_xgb_linear_acceptance_floor_base_initial_action_is_constant_0_2():
    from experiments.configs import get_config

    cfg = get_config("real_data_xgb_linear_acceptance_floor_base")
    assert cfg.x_fixed is not None
    u_batch = cfg.objective.policy_value(cfg.theta0, cfg.x_fixed)
    assert np.allclose(u_batch, 0.2)


def test_xgb_linear_policy_base_initial_action_is_constant_0_0():
    from experiments.configs import get_config

    cfg = get_config("real_data_xgb_linear_policy_base")
    assert cfg.x_fixed is not None
    u_batch = cfg.objective.policy_value(cfg.theta0, cfg.x_fixed)
    assert np.allclose(u_batch, 0.0)


@pytest.mark.parametrize("name", [
    "real_data_glm_constant_policy_base",
    "real_data_glm_constant_policy_trust_region_constr",
    "real_data_glm_linear_policy_base",
    "real_data_glm_linear_policy_trust_region_constr",
    "real_data_glm_mlp_policy_base",
    "real_data_glm_softmax_policy_base",
    "real_data_glm_softmax_policy_lagrangian_small",
    "real_data_glm_softmax_policy_trust_region_constr",
    "real_data_xgb_base",
    "real_data_xgb_linear_acceptance_floor_base",
    "real_data_xgb_linear_policy_base",
    "real_data_xgb_softmax_policy_base",
])
def test_real_data_configs_disable_correctness_gradients(name):
    from experiments.configs import get_config

    cfg = get_config(name)
    assert cfg.correctness.gradient_source == "none"


@pytest.mark.parametrize("name", [
    "real_data_glm_constant_policy_base",
    "real_data_glm_constant_policy_trust_region_constr",
    "real_data_glm_linear_policy_base",
    "real_data_glm_linear_policy_trust_region_constr",
    "real_data_glm_mlp_policy_base",
    "real_data_glm_softmax_policy_base",
    "real_data_glm_softmax_policy_lagrangian_small",
    "real_data_glm_softmax_policy_trust_region_constr",
    "real_data_xgb_base",
    "real_data_xgb_linear_acceptance_floor_base",
    "real_data_xgb_linear_policy_base",
    "real_data_xgb_softmax_policy_base",
])
def test_real_data_configs_enable_verbose_and_wandb(name):
    from experiments.configs import get_config

    cfg = get_config(name)
    assert cfg.verbose is True
    assert cfg.wandb_enabled is True


def test_x_fixed_validation_wrong_dim():
    """ExperimentConfig raises when x_fixed columns don't match state_dim."""
    from experiments.configs import get_config
    from experiments.config import ExperimentConfig

    cfg = get_config("real_data_glm_softmax_policy_base")

    with pytest.raises(ValueError, match="x_fixed"):
        ExperimentConfig(
            **{
                **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__ if k != "x_fixed"},
                "x_fixed": np.zeros((5, 3)),  # wrong n_cols for state_dim=12
            }
        )


def test_x_fixed_none_still_works_for_synthetic_config():
    """Existing synthetic configs (x_fixed=None) are unaffected."""
    from experiments.configs import get_config
    cfg = get_config("fixed_regression_base")
    assert cfg.x_fixed is None


def test_glm_linear_policy_base_estimators_move_and_agree_on_small_run():
    from experiments.configs import get_config
    from experiments.run import run_experiment

    cfg = get_config("real_data_glm_linear_policy_base")
    assert cfg.x_fixed is not None
    small_cfg = replace(
        cfg,
        n_samples=250,
        x_fixed=cfg.x_fixed[:250],
        x_fixed_row_indices=cfg.x_fixed_row_indices[:250],
        t_steps=50,
    )

    result = run_experiment(small_cfg, step_reporter=None)
    first = result.results["first_order"]
    fd = result.results["finite_difference"]
    spsa = result.results["spsa"]

    assert result.traces["first_order"].optimizer_status == 0
    assert result.traces["finite_difference"].optimizer_status == 0
    assert result.traces["spsa"].optimizer_status == 0
    assert result.config.theta0 is not None
    assert not np.allclose(first.theta, result.config.theta0)
    assert first.u == pytest.approx(fd.u, rel=5e-4, abs=5e-4)
    assert first.u == pytest.approx(spsa.u, rel=5e-4, abs=5e-4)


def test_glm_linear_policy_trust_region_constr_enforces_constraint_on_small_run():
    from experiments.configs import get_config
    from experiments.run import run_experiment

    cfg = get_config("real_data_glm_linear_policy_trust_region_constr")
    assert cfg.x_fixed is not None
    small_cfg = replace(
        cfg,
        n_samples=100,
        x_fixed=cfg.x_fixed[:100],
        x_fixed_row_indices=cfg.x_fixed_row_indices[:100],
        t_steps=20,
    )

    result = run_experiment(small_cfg, step_reporter=None)
    first = result.results["first_order"]
    fd = result.results["finite_difference"]

    assert result.traces["first_order"].optimizer_status is not None
    assert result.traces["finite_difference"].optimizer_status is not None
    assert first.mean_acceptance is not None
    assert fd.mean_acceptance is not None
    assert first.mean_acceptance >= small_cfg.acceptance_floor - 0.03
    assert fd.mean_acceptance >= small_cfg.acceptance_floor - 0.03
    for name in ("spsa", "stein_difference"):
        estimator = result.results[name]
        assert estimator.mean_acceptance is not None
        assert estimator.constraint_violation is not None
