"""Synthetic ladder experiment presets: one factory, one rung per preset name.

Ladder runs are policy-free theta-space optimizations of `SyntheticFunction`
rungs with a dummy fixed state batch. The estimator set intentionally excludes
`gauss_stein` (one-sided score estimator; too high variance in practice) and
`stein_difference` runs in its two-sided theta-space mode — u-space
perturbations are rejected because ladder objectives have no action space.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

import numpy as np

from experiments.config import ExperimentConfig
from objective import IMPLEMENTED_SYNTHETIC_LADDER, SYNTHETIC_LADDER


DEFAULT_DIMENSION = 20
DEFAULT_FUNCTION_SEED = 7
DEFAULT_ESTIMATORS = (
    "first_order",
    "finite_difference",
    "spsa",
    "stein_difference",
)


def build_synthetic_ladder_config(
    *,
    rung: str = "quadratic",
    dimension: int = DEFAULT_DIMENSION,
    function_seed: int = DEFAULT_FUNCTION_SEED,
    function_params: Mapping[str, Any] | None = None,
    **overrides: Any,
) -> ExperimentConfig:
    """Build a ladder preset for ``rung``; ``function_params`` feed the rung's
    ``from_seed`` factory (e.g. ``condition_number``, ``n_bumps``) and remaining
    overrides are standard `ExperimentConfig` fields.

    ``theta0`` defaults to ``None`` so the runner draws it from the theta seed
    stream, keeping seed sweeps over initialization meaningful.
    """
    if rung not in SYNTHETIC_LADDER:
        available = ", ".join(sorted(SYNTHETIC_LADDER))
        raise ValueError(f"Unknown synthetic ladder rung '{rung}'. Available: {available}.")
    if rung not in IMPLEMENTED_SYNTHETIC_LADDER:
        implemented = ", ".join(IMPLEMENTED_SYNTHETIC_LADDER)
        raise ValueError(
            f"Synthetic ladder rung '{rung}' is a structural stub and cannot run yet. "
            f"Implemented rungs: {implemented}."
        )
    valid_fields = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(key for key in overrides if key not in valid_fields)
    if unknown:
        unknown_text = ", ".join(unknown)
        raise ValueError(f"Unknown synthetic ladder config override fields: {unknown_text}.")
    if overrides.get("perturbation_space", "theta") != "theta":
        raise ValueError(
            "Synthetic ladder objectives have no action space; perturbation_space must be 'theta'."
        )

    objective = SYNTHETIC_LADDER[rung].from_seed(
        int(function_seed), dim=int(dimension), **dict(function_params or {})
    )
    payload: dict[str, Any] = {
        "seed": 7,
        "state_dim": 1,
        "n_samples": 1,
        "objective": objective,
        "theta0": None,
        "step_rule": "l-bfgs-b",
        "perturbation_space": "theta",
        "t_steps": 500,
        "step_size": 0.05,
        "sigma": 0.1,
        "n_grad_samples": 64,
        "enabled_estimators": DEFAULT_ESTIMATORS,
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


__all__ = [
    "DEFAULT_DIMENSION",
    "DEFAULT_ESTIMATORS",
    "DEFAULT_FUNCTION_SEED",
    "build_synthetic_ladder_config",
]
