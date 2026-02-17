"""Fast configuration for tests and smoke runs."""

from __future__ import annotations

from experiments.config import ExperimentConfig

CONFIG = ExperimentConfig(
    t_steps=1,
    n_samples=2,
    lbfgs_maxiter=5,
    lbfgs_samples=4,
    plot=False,
)
