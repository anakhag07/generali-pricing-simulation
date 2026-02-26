"""Baseline deterministic fixed-regression configuration."""

from __future__ import annotations

import numpy as np

from data.fixed_objective import FixedRegressionObjective
from experiments.config import ExperimentConfig
from experiments.defaults import default_policy_spec

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
    n_samples=10,
    step_rule="constant",
)
