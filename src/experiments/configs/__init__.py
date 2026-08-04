"""Preset experiment configurations."""

from __future__ import annotations

from dataclasses import fields, replace
from importlib import import_module
from typing import Any, Callable, Mapping

from experiments.config import ExperimentConfig
from experiments.configs.real_data_factory import build_real_data_config
from experiments.configs.synthetic_ladder import build_synthetic_ladder_config

# Factory-backed presets: name -> (factory, base payload). The factory validates
# objective-specific axes (rung, model_type, ...) before standard ExperimentConfig
# override fields.
_FACTORY_BASES: dict[str, tuple[Callable[..., ExperimentConfig], dict[str, Any]]] = {
    "synthetic_quadratic_base": (build_synthetic_ladder_config, {"rung": "quadratic"}),
    "synthetic_smoothed_nonconvex_base": (
        build_synthetic_ladder_config,
        {"rung": "smoothed_nonconvex"},
    ),
    "real_data_glm_base": (build_real_data_config, {"model_type": "glm"}),
    "real_data_xgb_base": (build_real_data_config, {"model_type": "xgb"}),
    "real_data_xgb_logit_spline_base": (
        build_real_data_config,
        {"model_type": "xgb_logit_spline"},
    ),
    "real_data_glm_glm_20260728_base": (
        build_real_data_config,
        {
            "acceptance_model_type": "glm_20260527",
            "loss_model_type": "glm_20260527",
        },
    ),
    "real_data_monotone_spline_glm_20260728_base": (
        build_real_data_config,
        {
            "acceptance_model_type": "xgb_monotone_spline_20260728",
            "loss_model_type": "glm_20260527",
        },
    ),
    "real_data_xgb_glm_20260728_base": (
        build_real_data_config,
        {
            "acceptance_model_type": "xgb_20260728",
            "loss_model_type": "glm_20260527",
        },
    ),
    "real_data_xgb_xgb_20260728_base": (
        build_real_data_config,
        {
            "acceptance_model_type": "xgb_20260728",
            "loss_model_type": "xgb_20260728",
        },
    ),
}

# Module-backed presets exposing a module-level CONFIG.
_MODULE_BASES = {
    "fixed_regression_base": "experiments.configs.fixed_regression_base",
    "planted_logistic_base": "experiments.configs.planted_logistic_base",
    "zeroth_order_proof_base": "experiments.configs.zeroth_order_proof_base",
}

_CONFIG_CACHE: dict[str, ExperimentConfig] = {}


def list_configs() -> tuple[str, ...]:
    return tuple([*_MODULE_BASES.keys(), *_FACTORY_BASES.keys()])


def get_config(name: str, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    override_payload = dict(overrides or {})
    if name in _FACTORY_BASES:
        if not override_payload and name in _CONFIG_CACHE:
            return _CONFIG_CACHE[name]
        factory, base_payload = _FACTORY_BASES[name]
        config = factory(**{**base_payload, **override_payload})
        if not override_payload:
            _CONFIG_CACHE[name] = config
        return config

    if name not in _MODULE_BASES:
        available = ", ".join(sorted(list_configs()))
        raise ValueError(f"Unknown experiment config '{name}'. Available: {available}.")

    if name not in _CONFIG_CACHE:
        module = import_module(_MODULE_BASES[name])
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


__all__ = ["get_config", "list_configs"]
