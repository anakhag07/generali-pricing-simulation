"""Base configuration for the one-dimensional zeroth-order proof checks."""

from __future__ import annotations

import numpy as np

from experiments.config import CorrectnessSpec, ExperimentConfig
from objective import ZerothOrderProofObjective


CONFIG = ExperimentConfig(
    state_dim=1,
    n_samples=1,
    objective=ZerothOrderProofObjective(),
    theta0=np.asarray([1.0], dtype=float),
    step_rule="constant",
    compute_backend="numpy",
    perturbation_space="theta",
    t_steps=1500,
    step_size=0.01,
    sigma=0.1,
    n_grad_samples=128,
    enabled_estimators=("finite_difference", "stein_difference"),
    correctness=CorrectnessSpec(gradient_source="exact"),
    x_fixed=np.zeros((1, 1), dtype=float),
    plot=False,
    verbose=False,
    wandb_enabled=False,
)


__all__ = ["CONFIG"]
