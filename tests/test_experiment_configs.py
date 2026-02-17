from __future__ import annotations

import pytest

from experiments.config import ExperimentConfig
from experiments.configs import get_config, list_configs


def test_get_config_returns_config() -> None:
    config = get_config("custom")
    assert isinstance(config, ExperimentConfig)


def test_get_config_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown experiment config"):
        get_config("does-not-exist")


def test_list_configs_includes_defaults() -> None:
    configs = list_configs()
    assert "custom" in configs
    assert "baseline_test" in configs
