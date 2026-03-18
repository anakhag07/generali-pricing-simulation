"""Template scaffold for creating new experiment presets.

Copy this file, replace ``None`` placeholders with concrete values, and then
register the new preset in ``experiments.configs.__init__``.
"""

from __future__ import annotations

from typing import Literal, Sequence, TypeAlias

from experiments.config import CorrectnessSpec
from objective.base import Objective, Policy

# --- Uncomment this block to build a runnable config ---
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

# REQUIRED: define policy and initial theta
POLICY = make_softmax_policy()
THETA0 = np.asarray([0.1] + [0.01] * STATE_DIM, dtype=float)

# REQUIRED: choose exactly ONE objective block
OBJECTIVE = make_fixed_regression_objective(
    policy=POLICY,
    beta_1=None,
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
    batch_size=BATCH_SIZE,
    grad_norm_tol=GRAD_NORM_TOL,
    ftol=FTOL,
)

RUNTIME = canonical_runtime_block(
    plot=PLOT,
    verbose=VERBOSE,
    wandb_enabled=WANDB_ENABLED,
    plot_dir=PLOT_DIR,
    wandb_project=WANDB_PROJECT,
    wandb_entity=WANDB_ENTITY,
    wandb_group=WANDB_GROUP,
    wandb_job_type=WANDB_JOB_TYPE,
    wandb_tags=WANDB_TAGS,
    wandb_mode=WANDB_MODE,
    wandb_log_plots=WANDB_LOG_PLOTS,
    wandb_estimator_allowlist=WANDB_ESTIMATOR_ALLOWLIST,
)

CORRECTNESS = CorrectnessSpec(
    gradient_source=CORRECTNESS_GRADIENT_SOURCE,
    numdiff_method=CORRECTNESS_NUMDIFF_METHOD,
    numdiff_step=CORRECTNESS_NUMDIFF_STEP,
    numdiff_aggregate=CORRECTNESS_NUMDIFF_AGGREGATE,
    numdiff_bounds=CORRECTNESS_NUMDIFF_BOUNDS,
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

# Optional placeholder if you keep this file and choose to fill it in place.
CONFIG: object | None = None
