"""Tests for real-data experiment config presets."""

import numpy as np
import pytest


@pytest.mark.parametrize("name", [
    "real_data_glm_base",
    "real_data_glm_linear_base",
    "real_data_xgb_base",
])
def test_config_loads(name):
    from experiments.configs import get_config
    cfg = get_config(name)
    assert cfg is not None


def test_glm_base_x_fixed_shape():
    from experiments.configs import get_config
    cfg = get_config("real_data_glm_base")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (5000, 12)
    assert cfg.state_dim == 12


def test_glm_linear_base_x_fixed_shape():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_base")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (5000, 12)
    assert cfg.state_dim == 12


def test_xgb_base_x_fixed_shape():
    from experiments.configs import get_config
    cfg = get_config("real_data_xgb_base")
    assert cfg.x_fixed is not None
    assert cfg.x_fixed.shape == (5000, 10)
    assert cfg.state_dim == 10


def test_glm_base_has_first_order():
    from experiments.configs import get_config
    cfg = get_config("real_data_glm_base")
    assert "first_order" in cfg.enabled_estimators


def test_glm_linear_base_has_first_order():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_base")
    assert "first_order" in cfg.enabled_estimators


def test_glm_linear_base_initial_action_is_constant_1_1():
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_base")
    assert cfg.x_fixed is not None
    u_batch = cfg.objective.policy.value(cfg.theta0, cfg.x_fixed)
    assert np.allclose(u_batch, 1.1)


def test_xgb_base_no_first_order():
    from experiments.configs import get_config
    cfg = get_config("real_data_xgb_base")
    assert "first_order" not in cfg.enabled_estimators


@pytest.mark.parametrize("name", [
    "real_data_glm_base",
    "real_data_glm_linear_base",
    "real_data_xgb_base",
])
def test_real_data_configs_disable_correctness_gradients(name):
    from experiments.configs import get_config

    cfg = get_config(name)
    assert cfg.correctness.gradient_source == "none"


@pytest.mark.parametrize("name", [
    "real_data_glm_base",
    "real_data_glm_linear_base",
    "real_data_xgb_base",
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
    cfg = get_config("real_data_glm_base")

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
