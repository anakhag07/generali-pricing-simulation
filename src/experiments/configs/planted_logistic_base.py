"""Planted logistic objective configuration."""

from __future__ import annotations

import numpy as np

from experiments.config import (
    build_experiment_config,
    canonical_runtime_block,
    canonical_training_block,
    make_planted_logistic_objective,
    make_softmax_policy,
)

STATE_DIM = 3
POLICY = make_softmax_policy()
BETA = np.asarray([0.5, -0.2, 0.3], dtype=float)
POLICY_THETA = np.zeros(POLICY.theta_dim(STATE_DIM), dtype=float)

TRAINING = canonical_training_block(
    n_samples=20,
    step_rule="l-bfgs-b",
    t_steps=5000,
    step_size=0.05,
    sigma=0.1,
    n_grad_samples=8,
    enabled_estimators=("finite_difference", "stein_difference"),
    perturbation_space="u",
)

RUNTIME = canonical_runtime_block(
    plot=True,
    verbose=False,
    wandb_enabled=False,
)

CONFIG = build_experiment_config(
    seed=7,
    state_dim=STATE_DIM,
    objective=make_planted_logistic_objective(
        policy=POLICY,
        alpha=1.0,
        beta=BETA,
        bias=-0.2,
        u_star=0.1,
    ),
    theta0=POLICY_THETA,
    training=TRAINING,
    runtime=RUNTIME,
)
