from __future__ import annotations

from dataclasses import fields

from experiments.config import ExperimentConfig
from experiments.configs import list_configs
from experiments.configs import config_template


def test_config_template_not_registered_as_runtime_preset() -> None:
    assert "config_template" not in list_configs()


def test_config_template_covers_all_experiment_fields() -> None:
    expected_fields = {field.name for field in fields(ExperimentConfig)}
    template_fields = set(config_template.EXPERIMENT_CONFIG_TEMPLATE.keys())
    assert template_fields == expected_fields
    assert all(value is None for value in config_template.EXPERIMENT_CONFIG_TEMPLATE.values())


def test_config_template_objective_and_correctness_blocks_present() -> None:
    assert set(config_template.FIXED_REGRESSION_OBJECTIVE_TEMPLATE.keys()) == {
        "policy",
        "beta_1",
        "beta_2",
        "beta_3",
        "beta_4",
    }
    assert set(config_template.PLANTED_LOGISTIC_OBJECTIVE_TEMPLATE.keys()) == {
        "policy",
        "alpha",
        "beta",
        "bias",
        "u_star",
    }
    assert set(config_template.CORRECTNESS_TEMPLATE.keys()) == {
        "gradient_source",
        "numdiff_method",
        "numdiff_step",
        "numdiff_aggregate",
        "numdiff_bounds",
    }
