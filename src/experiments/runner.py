"""Legacy wrapper for experiment runner."""

from __future__ import annotations

from typing import Optional, Tuple

from experiments.config import ExperimentConfig
from experiments.run import run_experiment


def run_demo(config: Optional[ExperimentConfig] = None) -> Tuple[float, float, float, float]:
    return run_experiment(config)


__all__ = ["ExperimentConfig", "run_demo"]
