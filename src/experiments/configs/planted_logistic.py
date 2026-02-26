"""Planted logistic objective configuration."""

from __future__ import annotations

import numpy as np

from data.planted_logistic import PlantedLogisticObjective
from experiments.config import ExperimentConfig
from experiments.defaults import default_policy_spec

STATE_DIM = 3
BETA = np.asarray([0.5, -0.2, 0.3], dtype=float)

CONFIG = ExperimentConfig(
    seed=7,
    state_dim=STATE_DIM,
    objective_model=PlantedLogisticObjective(
        alpha=1.0,
        beta=BETA,
        bias=-0.2,
        u_star=1.1,
    ),
    policy_spec=default_policy_spec(STATE_DIM),
    n_samples=20,
    step_rule="constant",
    t_steps=20000,
    step_size=0.05,
    sigma=0.1,
    n_grad_samples=64,
    lbfgs_maxiter=20000,
    plot=True,

    enabled_estimators=("zeroth_order","first_order", "lbfgs"),
    log_steps=False,
)
