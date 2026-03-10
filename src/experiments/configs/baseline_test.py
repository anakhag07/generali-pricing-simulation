"""Fast configuration for tests and smoke runs."""

from __future__ import annotations

import numpy as np

from experiments.config import ExperimentConfig
from experiments.defaults import default_policy_spec
from objective.fixed_objective import FixedRegressionObjective

STATE_DIM = 3
BETA_1 = np.asarray([0.02, 0.2, 0.5], dtype=float)
BETA_3 = np.asarray([0.005, 0.1, 0.2], dtype=float)

CONFIG = ExperimentConfig(
    state_dim=STATE_DIM,
    objective_model=FixedRegressionObjective.from_parameters(
        beta_1=BETA_1,
        beta_2=-0.8,
        beta_3=BETA_3,
        beta_4=0.4,
    ),
    policy_spec=default_policy_spec(STATE_DIM),
    n_samples=2,
    step_rule="constant",
    t_steps=1,
    n_grad_samples=2,
    lbfgs_maxiter=5,
    plot=False,
)
