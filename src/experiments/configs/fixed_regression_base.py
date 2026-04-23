"""Base fixed-regression experiment configuration."""

from __future__ import annotations

import numpy as np

from experiments.config import (
    build_experiment_config,
    canonical_runtime_block,
    canonical_training_block,
    make_fixed_regression_objective,
    make_softmax_policy,
)

STATE_DIM = 5

BETA_1 = np.linspace(0.02, 0.5, num=STATE_DIM, dtype=float)
BETA_2 = -1.2
BETA_3 = np.linspace(0.005, 0.2, num=STATE_DIM, dtype=float)
BETA_4 = 0.4

POLICY_THETA = np.zeros(STATE_DIM + 1, dtype=float)

TRAINING = canonical_training_block(
    n_samples=100,
    step_rule="l-bfgs-b",
    t_steps=50000,
    step_size=0.01,
    sigma=0.1,
    n_grad_samples=10,
    enabled_estimators=("spsa", "finite-difference", "stein-difference", "first_order"),
    perturbation_space="u",
    grad_norm_tol=1e-10,
    ftol=1e-10,
    constant_u_baselines=[-0.5, -0.3, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],

)

RUNTIME = canonical_runtime_block(
    plot=True,
    verbose=True,
    wandb_enabled=False,
)

CONFIG = build_experiment_config(
    seed=7,
    state_dim=STATE_DIM,
    objective=make_fixed_regression_objective(
        policy=make_softmax_policy(),
        beta_1=BETA_1,
        beta_2=BETA_2,
        beta_3=BETA_3,
        beta_4=BETA_4,
    ),
    theta0=POLICY_THETA,
    training=TRAINING,
    runtime=RUNTIME,
)
