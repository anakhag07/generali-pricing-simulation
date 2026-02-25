"""Custom experiment configuration (edit this file for recent runs)."""

from __future__ import annotations

import numpy as np

from data.fixed_objective import FixedRegressionObjective
from experiments.config import ExperimentConfig
from optimization.policy import POLICY_SOFTMAX, PolicySpec

STATE_DIM = 3

if STATE_DIM == 3:
    BETA_1 = np.asarray([0.02, 0.2, 0.5], dtype=float)
    BETA_3 = np.asarray([0.005, 0.1, 0.2], dtype=float)
else:
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
    t_steps=500,
    step_size=0.01,
    sigma=0.1,
    n_samples=64,
    lbfgs_maxiter=200,
    plot=True,
    plot_dir="plots",
)
