"""Result data structures for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from experiments.config import ExperimentConfig


@dataclass(frozen=True)
class OptimizationTrace:
    """Per-step trace: u values, objective values, gradient norms, and theta history."""

    steps: Sequence[int]
    u_values: Sequence[float]
    objective_values: Sequence[float]
    u_grad_estimates: Sequence[float]
    u_true_gradients: Optional[Sequence[float]] = None
    theta_grad_norms: Optional[Sequence[float]] = None
    true_theta_grad_norms: Optional[Sequence[float]] = None
    step_sizes: Optional[Sequence[float]] = None
    theta_values: Optional[Sequence[np.ndarray]] = None
    optimizer_success: Optional[bool] = None
    optimizer_status: Optional[int] = None
    optimizer_message: Optional[str] = None
    constraint_violation: Optional[float] = None
    acceptance_multiplier: Optional[float] = None


@dataclass(frozen=True)
class EstimatorResult:
    """Final result for one estimator: theta, mean action, objective value, and wall time."""

    theta: np.ndarray
    u: float
    value: float
    time: float
    mean_acceptance: float | None = None
    constraint_violation: float | None = None
    acceptance_multiplier: float | None = None


@dataclass(frozen=True)
class ExperimentResult:
    """Full experiment result: config, samples, traces, and final values per estimator."""

    config: ExperimentConfig
    x_samples: np.ndarray  # Shape (n_samples, state_dim)
    initial_value: float
    results: Mapping[str, EstimatorResult]
    traces: Mapping[str, OptimizationTrace]
    u_star: Optional[float] = None
    value_at_u_star: Optional[float] = None
    initial_mean_acceptance: Optional[float] = None


__all__ = [
    "EstimatorResult",
    "ExperimentResult",
    "OptimizationTrace",
]
