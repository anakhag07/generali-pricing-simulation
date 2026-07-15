from __future__ import annotations

import numpy as np
import pytest

from experiments.config import ExperimentConfig
from experiments.configs import get_config, list_configs
from objective import QuadraticObjective


def test_get_config_returns_config() -> None:
    config = get_config("fixed_regression_base")
    assert isinstance(config, ExperimentConfig)


def test_get_config_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown experiment config"):
        get_config("does-not-exist")


def test_list_configs_includes_defaults() -> None:
    configs = list_configs()
    assert "first_order_runs_diff_starts" in configs
    assert "fixed_regression_base" in configs
    assert "planted_logistic_base" in configs
    assert "quadratic_base" in configs
    assert "real_data_glm_base" in configs
    assert "real_data_xgb_base" in configs
    assert "real_data_glm_softmax_policy_base" not in configs


def test_quadratic_config_dimension_override_sets_objective_and_fixed_norm_start() -> None:
    config = get_config("quadratic_base", overrides={"dimension": 7, "plot": False})

    assert isinstance(config.objective, QuadraticObjective)
    assert config.objective.dimension == 7
    assert config.theta0 is not None
    assert config.theta0.shape == (7,)
    assert np.linalg.norm(config.theta0) == pytest.approx(1.0)
    assert config.objective.value(config.theta0, config.x_fixed) == pytest.approx(0.5)
    assert config.perturbation_space == "theta"
    assert "constant" not in config.enabled_estimators


def test_get_config_accepts_real_data_builder_overrides() -> None:
    config = get_config(
        "real_data_glm_base",
        overrides={
            "policy_kind": "linear",
            "feature_order": "quadratic",
            "n_samples": 20,
            "plot": False,
            "wandb_enabled": False,
        },
    )
    assert isinstance(config, ExperimentConfig)
    assert config.n_samples == 20
    assert config.plot is False
