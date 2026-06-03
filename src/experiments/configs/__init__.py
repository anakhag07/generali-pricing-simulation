"""Preset experiment configurations."""

from __future__ import annotations

from dataclasses import fields, replace
from importlib import import_module
from typing import Any, Mapping

from experiments.config import ExperimentConfig
from experiments.configs.real_data_factory import build_real_data_config

_CONFIG_MODULES = {
    "first_order_runs_diff_starts": "experiments.configs.first_order_runs_diff_starts",
    "fixed_regression_base": "experiments.configs.fixed_regression_base",
    "planted_logistic_base": "experiments.configs.planted_logistic_base",
}

_REAL_DATA_BASES: dict[str, dict[str, Any]] = {
    "real_data_glm_base": {"model_type": "glm"},
    "real_data_xgb_base": {"model_type": "xgb"},
}

_LEGACY_REAL_DATA_ALIASES: dict[str, dict[str, Any]] = {
    "real_data_glm_constant_policy_base": {"model_type": "glm", "policy_kind": "constant", "seed": 8},
    "real_data_glm_constant_policy_trust_region_constr": {
        "model_type": "glm",
        "policy_kind": "constant",
        "constraint_mode": "trust_constr",
        "seed": 8,
    },
    "real_data_glm_linear_policy_base": {"model_type": "glm", "policy_kind": "linear", "seed": 8},
    "real_data_glm_linear_policy_quadratic_base": {
        "model_type": "glm",
        "policy_kind": "linear",
        "feature_order": "quadratic",
        "seed": 8,
    },
    "real_data_glm_linear_policy_cubic_base": {
        "model_type": "glm",
        "policy_kind": "linear",
        "feature_order": "cubic",
        "seed": 8,
    },
    "real_data_glm_linear_policy_quartic_base": {
        "model_type": "glm",
        "policy_kind": "linear",
        "feature_order": "quartic",
        "seed": 8,
    },
    "real_data_glm_linear_policy_trust_region_constr": {
        "model_type": "glm",
        "policy_kind": "linear",
        "constraint_mode": "trust_constr",
        "seed": 8,
        "initial_u": 0.0,
    },
    "real_data_glm_mlp_policy_base": {"model_type": "glm", "policy_kind": "mlp"},
    "real_data_glm_softmax_policy_base": {
        "model_type": "glm",
        "policy_kind": "softmax",
        "n_samples": 194373,
        "enabled_estimators": ("first_order", "finite_difference", "spsa", "stein_difference"),
    },
    "real_data_glm_softmax_policy_quadratic_base": {
        "model_type": "glm",
        "policy_kind": "softmax",
        "feature_order": "quadratic",
    },
    "real_data_glm_softmax_policy_cubic_base": {
        "model_type": "glm",
        "policy_kind": "softmax",
        "feature_order": "cubic",
    },
    "real_data_glm_softmax_policy_quartic_base": {
        "model_type": "glm",
        "policy_kind": "softmax",
        "feature_order": "quartic",
    },
    "real_data_glm_softmax_policy_quartic_no_pca": {
        "model_type": "glm",
        "policy_kind": "softmax",
        "feature_order": "quartic",
        "policy_preprocessing": "no_pca",
    },
    "real_data_glm_softmax_policy_trust_region_constr": {
        "model_type": "glm",
        "policy_kind": "softmax",
        "constraint_mode": "trust_constr",
        "enabled_estimators": ("first_order", "finite_difference", "spsa", "stein_difference"),
        "constant_u_baselines": (0.0853,),
    },
    "real_data_glm_softmax_policy_lagrangian_small": {
        "model_type": "glm",
        "policy_kind": "softmax",
        "constraint_mode": "lagrangian",
        "n_samples": 250,
        "t_steps": 50,
        "n_grad_samples": 8,
        "lagrangian_lambda": 250.0,
    },
    "real_data_glm_softmax_policy_linear_no_pca_trust_region_constr": {
        "model_type": "glm",
        "policy_kind": "softmax",
        "policy_preprocessing": "no_pca",
        "constraint_mode": "trust_constr",
        "t_steps": 500,
        "enabled_estimators": ("first_order", "finite_difference", "stein_difference"),
    },
    "real_data_glm_softmax_policy_quadratic_no_pca_trust_region_constr": {
        "model_type": "glm",
        "policy_kind": "softmax",
        "feature_order": "quadratic",
        "policy_preprocessing": "no_pca",
        "constraint_mode": "trust_constr",
        "t_steps": 500,
        "enabled_estimators": ("first_order",),
    },
    "real_data_glm_linear_policy_quartic_no_pca_trust_region_constr": {
        "model_type": "glm",
        "policy_kind": "linear",
        "feature_order": "quartic",
        "policy_preprocessing": "no_pca",
        "constraint_mode": "trust_constr",
        "t_steps": 500,
        "enabled_estimators": ("first_order",),
    },
    "real_data_glm_mlp_policy_no_pca_trust_region_constr": {
        "model_type": "glm",
        "policy_kind": "mlp",
        "policy_preprocessing": "no_pca",
        "constraint_mode": "trust_constr",
        "t_steps": 500,
        "enabled_estimators": ("first_order",),
    },
    "real_data_xgb_linear_policy_base": {
        "model_type": "xgb",
        "policy_kind": "linear",
        "seed": 8,
        "initial_u": 0.0,
    },
    "real_data_xgb_softmax_policy_base": {"model_type": "xgb", "policy_kind": "softmax"},
    "real_data_xgb_softmax_policy_trust_region_constr": {
        "model_type": "xgb",
        "policy_kind": "softmax",
        "constraint_mode": "trust_constr",
    },
    "real_data_xgb_linear_acceptance_floor_base": {
        "model_type": "xgb",
        "policy_kind": "linear",
        "constraint_mode": "penalty",
        "seed": 8,
        "initial_u": 0.2,
        "n_grad_samples": 10,
    },
}

_CONFIG_CACHE: dict[str, ExperimentConfig] = {}


def list_configs() -> tuple[str, ...]:
    return tuple([*_CONFIG_MODULES.keys(), *_REAL_DATA_BASES.keys(), *_LEGACY_REAL_DATA_ALIASES.keys()])


def get_config(name: str, overrides: Mapping[str, Any] | None = None) -> ExperimentConfig:
    override_payload = dict(overrides or {})
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
    if name in _LEGACY_REAL_DATA_ALIASES:
        return dict(_LEGACY_REAL_DATA_ALIASES[name])
    return None


__all__ = ["get_config", "list_configs"]
