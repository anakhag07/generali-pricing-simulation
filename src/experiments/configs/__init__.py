"""Preset experiment configurations."""

from __future__ import annotations

from experiments.config import ExperimentConfig
from experiments.configs.baseline_fixed import CONFIG as baseline_fixed
from experiments.configs.baseline_test import CONFIG as baseline_test
from experiments.configs.custom import CONFIG as custom

_CONFIGS = {
    "baseline_fixed": baseline_fixed,
    "baseline_test": baseline_test,
    "custom": custom,
}


def list_configs() -> tuple[str, ...]:
    return tuple(_CONFIGS.keys())


def get_config(name: str) -> ExperimentConfig:
    try:
        return _CONFIGS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_CONFIGS.keys()))
        raise ValueError(f"Unknown experiment config '{name}'. Available: {available}.") from exc


__all__ = ["get_config", "list_configs"]
