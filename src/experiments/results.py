"""Result data structures for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from experiments.config import ExperimentConfig


@dataclass(frozen=True)
class ConstantBaselineResult:
    """Evaluation of a fixed action $$u$$ on the experiment batch."""

    u: float
    value: float
    mean_acceptance: float | None = None


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
    mean_acceptance_values: Optional[Sequence[float]] = None
    projected_loss_values: Optional[Sequence[float]] = None
    projected_revenue_values: Optional[Sequence[float]] = None
    theta_values: Optional[Sequence[np.ndarray]] = None
    optimizer_success: Optional[bool] = None
    optimizer_optimality: Optional[float] = None
    optimizer_lagrangian_grad: Optional[Sequence[float]] = None
    optimizer_status: Optional[int] = None
    optimizer_message: Optional[str] = None
    constraint_violation: Optional[float] = None
    acceptance_multiplier: Optional[float] = None
    constraint_penalty: Optional[float] = None


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
    constraint_penalty: float | None = None


@dataclass(frozen=True)
class PolicyEvaluation:
    """Final policy metrics evaluated on one data split."""

    n_samples: int
    objective_value: float
    objective_sum: float
    mean_u: float
    u_q25: float
    u_q75: float
    mean_acceptance: float | None = None
    projected_loss: float | None = None
    projected_revenue: float | None = None


@dataclass(frozen=True)
class ExperimentResult:
    """Full experiment result: config, samples, traces, and final values per estimator."""

    config: ExperimentConfig
    x_samples: Any  # Training samples; real data may be a DataFrame
    initial_value: float
    results: Mapping[str, EstimatorResult]
    traces: Mapping[str, OptimizationTrace]
    u_star: Optional[float] = None
    value_at_u_star: Optional[float] = None
    initial_mean_acceptance: Optional[float] = None
    constant_u_baselines: Sequence[ConstantBaselineResult] = ()
    x_test: Any | None = None
    train_indices: Optional[np.ndarray] = None
    test_indices: Optional[np.ndarray] = None
    train_row_indices: Optional[np.ndarray] = None
    test_row_indices: Optional[np.ndarray] = None
    train_metrics: Mapping[str, PolicyEvaluation] = field(default_factory=dict)
    test_metrics: Mapping[str, PolicyEvaluation] = field(default_factory=dict)


__all__ = [
    "ConstantBaselineResult",
    "EstimatorResult",
    "ExperimentResult",
    "OptimizationTrace",
    "PolicyEvaluation",
]
