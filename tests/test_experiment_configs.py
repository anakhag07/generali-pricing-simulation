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
    assert "fixed_regression_base" in configs
    assert "planted_logistic_base" in configs
