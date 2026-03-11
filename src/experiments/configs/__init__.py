"""Preset experiment configurations."""

from __future__ import annotations

from experiments.config import ExperimentConfig
from experiments.configs.fixed_regression_base import CONFIG as fixed_regression_base
from experiments.configs.planted_logistic_base import CONFIG as planted_logistic_base

_CONFIGS = {
    "fixed_regression_base": fixed_regression_base,
    "planted_logistic_base": planted_logistic_base,
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
