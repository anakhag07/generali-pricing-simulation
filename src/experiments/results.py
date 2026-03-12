"""Result data structures for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np

from objective.base import StateVector
from experiments.config import ExperimentConfig


@dataclass(frozen=True)
class OptimizationTrace:
    steps: Sequence[int]
    u_values: Sequence[float]
    objective_values: Sequence[float]
    u_grad_estimates: Sequence[float]
    u_true_gradients: Optional[Sequence[float]] = None
    theta_grad_norms: Optional[Sequence[float]] = None
    true_theta_grad_norms: Optional[Sequence[float]] = None
    step_sizes: Optional[Sequence[float]] = None
    theta_values: Optional[Sequence[np.ndarray]] = None
    optimizer_status: Optional[int] = None
    optimizer_message: Optional[str] = None


@dataclass(frozen=True)
class EstimatorResult:
    theta: np.ndarray
    u: float
    value: float
    time: float


@dataclass(frozen=True)
class ExperimentResult:
    config: ExperimentConfig
    x_samples: Sequence[StateVector]
    initial_value: float
    results: Mapping[str, EstimatorResult]
    traces: Mapping[str, OptimizationTrace]
    u_star: Optional[float] = None
    value_at_u_star: Optional[float] = None


__all__ = [
    "EstimatorResult",
    "ExperimentResult",
    "OptimizationTrace",
]
