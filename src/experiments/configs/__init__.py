"""Preset experiment configurations."""

from __future__ import annotations

from dataclasses import fields, replace
from importlib import import_module
from typing import Any, Mapping

from experiments.config import ExperimentConfig
from experiments.configs.quadratic_base import build_quadratic_config
from experiments.configs.real_data_factory import build_real_data_config
from experiments.configs.synthetic_ladder import build_synthetic_ladder_config

_CONFIG_MODULES = {
    "first_order_runs_diff_starts": "experiments.configs.first_order_runs_diff_starts",
    "fixed_regression_base": "experiments.configs.fixed_regression_base",
    "planted_logistic_base": "experiments.configs.planted_logistic_base",
}

_REAL_DATA_BASES: dict[str, dict[str, Any]] = {
    "real_data_glm_base": {"model_type": "glm"},
    "real_data_xgb_base": {"model_type": "xgb"},
    "real_data_xgb_logit_spline_base": {"model_type": "xgb_logit_spline"},
}

_QUADRATIC_BASES: dict[str, dict[str, Any]] = {
    "quadratic_base": {},
}

_SYNTHETIC_LADDER_BASES: dict[str, dict[str, Any]] = {
    "synthetic_quadratic_base": {"rung": "quadratic"},
    "synthetic_smoothed_nonconvex_base": {"rung": "smoothed_nonconvex"},
}

_CONFIG_CACHE: dict[str, ExperimentConfig] = {}


def list_configs() -> tuple[str, ...]:
    return tuple(
        [
            *_CONFIG_MODULES.keys(),
            *_QUADRATIC_BASES.keys(),
            *_SYNTHETIC_LADDER_BASES.keys(),
            *_REAL_DATA_BASES.keys(),
        ]
    )


def get_config(name: str, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    override_payload = dict(overrides or {})
    if name in _QUADRATIC_BASES:
        if not override_payload and name in _CONFIG_CACHE:
            return _CONFIG_CACHE[name]
        config = build_quadratic_config(**override_payload)
        if not override_payload:
            _CONFIG_CACHE[name] = config
        return config
    if name in _SYNTHETIC_LADDER_BASES:
        if not override_payload and name in _CONFIG_CACHE:
            return _CONFIG_CACHE[name]
        payload = dict(_SYNTHETIC_LADDER_BASES[name])
        payload.update(override_payload)
        config = build_synthetic_ladder_config(**payload)
        if not override_payload:
            _CONFIG_CACHE[name] = config
        return config
    real_data_payload = _real_data_payload(name)
    if real_data_payload is not None:
        real_data_payload.update(override_payload)
        if not override_payload and name in _CONFIG_CACHE:
            return _CONFIG_CACHE[name]
        config = build_real_data_config(**real_data_payload)
        if not override_payload:
            _CONFIG_CACHE[name] = config
        return config

    if name not in _CONFIG_MODULES:
        available = ", ".join(sorted(list_configs()))
        raise ValueError(f"Unknown experiment config '{name}'. Available: {available}.")

    if name not in _CONFIG_CACHE:
        module = import_module(_CONFIG_MODULES[name])
        _CONFIG_CACHE[name] = module.CONFIG
    config = _CONFIG_CACHE[name]
    if override_payload:
        valid_fields = {field.name for field in fields(ExperimentConfig)}
        unknown = sorted(key for key in override_payload if key not in valid_fields)
        if unknown:
            unknown_text = ", ".join(unknown)
            raise ValueError(f"Unknown config override fields: {unknown_text}.")
        return replace(config, **override_payload)
    return config


def _real_data_payload(name: str) -> dict[str, Any] | None:
    if name in _REAL_DATA_BASES:
        return dict(_REAL_DATA_BASES[name])
    return None


__all__ = ["get_config", "list_configs"]
