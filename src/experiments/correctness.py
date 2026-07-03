"""Gradient-correctness helpers for experiment diagnostics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from experiments.config import CorrectnessSpec
from objective.base import Objective
from optimization.helpers import finite_difference_theta_grad


TrueThetaGradFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def resolve_true_grad_theta_fn(
    objective: Objective,
    correctness: CorrectnessSpec,
) -> TrueThetaGradFn | None:
    """Return a theta-gradient proxy based on correctness settings."""
    if correctness.gradient_source == "none":
        return None
    if correctness.gradient_source == "exact":
        return lambda theta, x_batch: objective.grad(theta, x_batch)
    if correctness.gradient_source == "denoised_exact":
        denoised_objective = getattr(objective, "base_objective", objective)
        return lambda theta, x_batch: denoised_objective.grad(theta, x_batch)
    if correctness.gradient_source == "numdiff":
        return lambda theta, x_batch: finite_difference_theta_grad(
            lambda theta_eval: objective.value(theta_eval, x_batch),
            theta,
            method=correctness.numdiff_method,
            step=correctness.numdiff_step,
            bounds=correctness.numdiff_bounds,
        )
    raise ValueError(f"Unknown gradient_source '{correctness.gradient_source}'.")


__all__ = ["TrueThetaGradFn", "resolve_true_grad_theta_fn"]
