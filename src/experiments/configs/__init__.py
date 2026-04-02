"""Preset experiment configurations."""

from __future__ import annotations

from importlib import import_module

from experiments.config import ExperimentConfig

_CONFIG_MODULES = {
    "first_order_runs_diff_starts": "experiments.configs.first_order_runs_diff_starts",
    "fixed_regression_base": "experiments.configs.fixed_regression_base",
    "planted_logistic_base": "experiments.configs.planted_logistic_base",
    "real_data_glm_base": "experiments.configs.real_data_glm_base",
    "real_data_xgb_base": "experiments.configs.real_data_xgb_base",
}

_CONFIG_CACHE: dict[str, ExperimentConfig] = {}


def list_configs() -> tuple[str, ...]:
    return tuple(_CONFIG_MODULES.keys())


def get_config(name: str) -> ExperimentConfig:
    try:
        module_name = _CONFIG_MODULES[name]
    except KeyError as exc:
        available = ", ".join(sorted(_CONFIG_MODULES.keys()))
        raise ValueError(f"Unknown experiment config '{name}'. Available: {available}.") from exc

    if name not in _CONFIG_CACHE:
        module = import_module(module_name)
        _CONFIG_CACHE[name] = module.CONFIG
    return _CONFIG_CACHE[name]


__all__ = ["get_config", "list_configs"]
