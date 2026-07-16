from __future__ import annotations

import pytest

from experiments.config import ExperimentConfig
from experiments.configs import get_config, list_configs


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
    assert "synthetic_quadratic_base" in configs
    assert "real_data_glm_base" in configs
    assert "real_data_xgb_base" in configs
    assert "real_data_xgb_logit_spline_base" in configs
    assert "real_data_glm_softmax_policy_base" not in configs



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
