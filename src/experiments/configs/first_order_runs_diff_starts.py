"""Comparison config for varying starts with a planted logistic objective."""

from __future__ import annotations

import numpy as np

from experiments.config import (
    CorrectnessSpec,
    build_experiment_config,
    canonical_runtime_block,
    canonical_training_block,
    make_fixed_regression_objective,
    make_planted_logistic_objective,
    make_softmax_policy,
)
from optimization.steps import STEP_RULE_LBFGSB

STATE_DIM = 5
N_SAMPLES = 100
STEP_RULE = STEP_RULE_LBFGSB
SEED = 7
T_STEPS = 1000
STEP_SIZE = 0.01
SIGMA = 0.05
N_GRAD_SAMPLES = 256
VERBOSE = True
PLOT = True
PLOT_DIR = "plots"
ENABLED_ESTIMATORS = ("first_order", "spsa", "stein-difference", "finite-difference")
WANDB_ENABLED = True

FIXED_BETA_1 = np.linspace(0.02, 0.5, num=STATE_DIM, dtype=float)
FIXED_BETA_2 = -1.2
FIXED_BETA_3 = np.linspace(0.005, 0.2, num=STATE_DIM, dtype=float)
FIXED_BETA_4 = 0.4

PLANTED_ALPHA = 1.0
PLANTED_BETA = np.asarray([0.5, -0.2, 0.3, 0.1, -0.4], dtype=float)
PLANTED_BIAS = -0.2
PLANTED_U_STAR = 1.1

CORRECTNESS_GRADIENT_SOURCE = "exact"

POLICY = make_softmax_policy()
THETA0 = np.asarray([0.1] + [0.01] * STATE_DIM, dtype=float)

OBJECTIVE = make_fixed_regression_objective(
    policy=POLICY,
    beta_1=FIXED_BETA_1,
    beta_2=FIXED_BETA_2,
    beta_3=FIXED_BETA_3,
    beta_4=FIXED_BETA_4,
)
# OBJECTIVE = make_planted_logistic_objective(
#     policy=POLICY,
#     alpha=PLANTED_ALPHA,
#     beta=PLANTED_BETA,
#     bias=PLANTED_BIAS,
#     u_star=PLANTED_U_STAR,
# )

TRAINING = canonical_training_block(
    n_samples=N_SAMPLES,
    step_rule=STEP_RULE,
    t_steps=T_STEPS,
    step_size=STEP_SIZE,
    sigma=SIGMA,
    n_grad_samples=N_GRAD_SAMPLES,
    enabled_estimators=ENABLED_ESTIMATORS,
)

RUNTIME = canonical_runtime_block(
    plot=PLOT,
    verbose=VERBOSE,
    wandb_enabled=WANDB_ENABLED,
    plot_dir=PLOT_DIR,
)

CORRECTNESS = CorrectnessSpec(
    gradient_source=CORRECTNESS_GRADIENT_SOURCE,
)

CONFIG = build_experiment_config(
    seed=SEED,
    state_dim=STATE_DIM,
    objective=OBJECTIVE,
    theta0=THETA0,
    training=TRAINING,
    runtime=RUNTIME,
    correctness=CORRECTNESS,
)
