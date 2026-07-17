"""Gradient-correctness helpers for experiment diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from experiments.config import CorrectnessSpec
from objective.base import Objective
from objective.modifications import NoisyObjective
from optimization.helpers import finite_difference_theta_grad


TrueThetaGradFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _innermost_base_objective(objective: Objective) -> Objective:
    """Return the innermost wrapped objective, unwrapping any deterministic
    ``base_objective`` chain (e.g. ``NoisyObjective(BiasedObjective(base))``).

    A single wrapper resolves to its immediate base (unchanged behavior); a stack
    of wrappers resolves all the way to the clean, unbiased, noise-free objective
    so ``denoised_exact`` references the true first-order objective."""
    current = objective
    seen: set[int] = {id(current)}
    while True:
        base = getattr(current, "base_objective", None)
        if base is None or id(base) in seen:
            return current
        seen.add(id(base))
        current = base


def _noise_free_objective(objective: Objective) -> Objective:
    """Return a copy with only noise wrappers removed.

    Deterministic wrappers such as bias or regularization are preserved by
    replacing their ``base_objective`` with the recursively noise-free base.
    """
    if isinstance(objective, NoisyObjective):
        return _noise_free_objective(objective.base_objective)
    base = getattr(objective, "base_objective", None)
    if base is None:
        return objective
    noise_free_base = _noise_free_objective(base)
    if noise_free_base is base:
        return objective
    try:
        return replace(objective, base_objective=noise_free_base)
    except TypeError:
        return objective


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
        denoised_objective = _innermost_base_objective(objective)
        return lambda theta, x_batch: denoised_objective.grad(theta, x_batch)
    if correctness.gradient_source == "noise_free_exact":
        noise_free_objective = _noise_free_objective(objective)
        return lambda theta, x_batch: noise_free_objective.grad(theta, x_batch)
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
