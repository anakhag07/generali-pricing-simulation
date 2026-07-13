"""Dimension-configurable strongly convex quadratic experiment preset."""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np

from experiments.config import ExperimentConfig
from objective import QuadraticObjective


DEFAULT_DIMENSION = 10


def build_quadratic_config(
    *,
    dimension: int = DEFAULT_DIMENSION,
    **overrides: Any,
) -> ExperimentConfig:
    """Build the quadratic preset, using ``dimension`` for theta and its fixed-norm start."""
    valid_fields = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(key for key in overrides if key not in valid_fields)
    if unknown:
        unknown_text = ", ".join(unknown)
        raise ValueError(f"Unknown quadratic config override fields: {unknown_text}.")

    objective = QuadraticObjective(dimension=dimension)
    theta0 = np.ones(objective.dimension, dtype=float) / np.sqrt(objective.dimension)
    payload: dict[str, Any] = {
        "seed": 7,
        "state_dim": 1,
        "n_samples": 1,
        "objective": objective,
        "theta0": theta0,
        "step_rule": "l-bfgs-b",
        "perturbation_space": "theta",
        "t_steps": 100,
        "step_size": 0.05,
        "sigma": 0.1,
        "n_grad_samples": 64,
        "enabled_estimators": (
            "first_order",
            "finite_difference",
            "gauss_stein",
            "stein_difference",
            "spsa",
        ),
        "plot": True,
        "verbose": False,
        "wandb_enabled": False,
    }
    payload.update(overrides)
    payload.setdefault(
        "x_fixed",
        np.zeros((int(payload["n_samples"]), int(payload["state_dim"])), dtype=float),
    )
    return ExperimentConfig(**payload)


CONFIG = build_quadratic_config()


__all__ = ["CONFIG", "DEFAULT_DIMENSION", "build_quadratic_config"]
