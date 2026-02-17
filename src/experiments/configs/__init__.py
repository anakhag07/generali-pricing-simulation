"""Preset experiment configurations."""

from __future__ import annotations

from experiments.config import ExperimentConfig
from experiments.configs.baseline_fixed import CONFIG as baseline_fixed
from experiments.configs.baseline_stochastic import CONFIG as baseline_stochastic
from experiments.configs.custom import CONFIG as custom
from experiments.configs.smoke import CONFIG as smoke

_CONFIGS = {
    "baseline_fixed": baseline_fixed,
    "baseline_stochastic": baseline_stochastic,
    "custom": custom,
    "smoke": smoke,
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
