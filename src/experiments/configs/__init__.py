"""Preset experiment configurations."""

from __future__ import annotations

from importlib import import_module

from experiments.config import ExperimentConfig

_CONFIG_MODULES = {
    "first_order_runs_diff_starts": "experiments.configs.first_order_runs_diff_starts",
    "fixed_regression_base": "experiments.configs.fixed_regression_base",
    "real_data_glm_constant_policy_base": "experiments.configs.real_data_glm_constant_policy_base",
    "real_data_glm_constant_policy_trust_region_constr": "experiments.configs.real_data_glm_constant_policy_trust_region_constr",
    "planted_logistic_base": "experiments.configs.planted_logistic_base",
    "real_data_glm_linear_policy_base": "experiments.configs.real_data_glm_linear_policy_base",
    "real_data_glm_linear_policy_cubic_base": "experiments.configs.real_data_glm_linear_policy_cubic_base",
    "real_data_glm_linear_policy_quadratic_base": "experiments.configs.real_data_glm_linear_policy_quadratic_base",
    "real_data_glm_linear_policy_quartic_base": "experiments.configs.real_data_glm_linear_policy_quartic_base",
    "real_data_glm_linear_policy_trust_region_constr": "experiments.configs.real_data_glm_linear_policy_trust_region_constr",
    "real_data_glm_mlp_policy_base": "experiments.configs.real_data_glm_mlp_policy_base",
    "real_data_glm_softmax_policy_base": "experiments.configs.real_data_glm_softmax_policy_base",
    "real_data_glm_softmax_policy_cubic_base": "experiments.configs.real_data_glm_softmax_policy_cubic_base",
    "real_data_glm_softmax_policy_lagrangian_small": "experiments.configs.real_data_glm_softmax_policy_lagrangian_small",
    "real_data_glm_softmax_policy_quadratic_base": "experiments.configs.real_data_glm_softmax_policy_quadratic_base",
    "real_data_glm_softmax_policy_quartic_base": "experiments.configs.real_data_glm_softmax_policy_quartic_base",
    "real_data_glm_softmax_policy_quartic_no_pca": "experiments.configs.real_data_glm_softmax_policy_quartic_no_pca",
    "real_data_glm_softmax_policy_trust_region_constr": "experiments.configs.real_data_glm_softmax_policy_trust_region_constr",
    "real_data_xgb_base": "experiments.configs.real_data_xgb_base",
    "real_data_xgb_linear_acceptance_floor_base": "experiments.configs.real_data_xgb_linear_acceptance_floor_base",
    "real_data_xgb_linear_policy_base": "experiments.configs.real_data_xgb_linear_policy_base",
    "real_data_xgb_softmax_policy_base": "experiments.configs.real_data_xgb_softmax_policy_base",
    "real_data_xgb_softmax_policy_trust_region_constr": "experiments.configs.real_data_xgb_softmax_policy_trust_region_constr",
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
