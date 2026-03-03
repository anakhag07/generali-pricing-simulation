"""Custom experiment configuration (edit this file for recent runs)."""

from __future__ import annotations

import numpy as np

from data.fixed_objective import FixedRegressionObjective
from experiments.config import ExperimentConfig
from optimization.policy import POLICY_SOFTMAX, PolicySpec

STATE_DIM = 2

BETA_1 = np.linspace(0.02, 0.5, num=STATE_DIM, dtype=float)
BETA_3 = np.linspace(0.005, 0.2, num=STATE_DIM, dtype=float)

POLICY_THETA = np.asarray([0.1] + [0.01] * STATE_DIM, dtype=float)

CONFIG = ExperimentConfig(
    seed=7,
    state_dim=STATE_DIM,
    objective_model=FixedRegressionObjective.from_parameters(
        beta_1=BETA_1,
        beta_2=-0.8,
        beta_3=BETA_3,
        beta_4=0.4,
    ),
    policy_spec=PolicySpec(theta=POLICY_THETA, kind=POLICY_SOFTMAX),
    n_samples=10,
    step_rule="armijo",
    t_steps=1000000,
    step_size=0.01,
    sigma=0.1,
    n_grad_samples=10,
    lbfgs_maxiter=100000,
    plot=True,
    plot_dir="plots",

    enabled_estimators=("zeroth_order", "first_order", "lbfgs"),
    # enabled_estimators=("first_order", "lbfgs"),

    log_steps=False,
)
