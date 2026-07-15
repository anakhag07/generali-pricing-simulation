"""Template scaffold for creating new experiment presets.

Copy this file, fill in the ``None`` placeholders, then register your new
preset in ``experiments.configs.__init__``.
"""

from __future__ import annotations

# Fill these placeholders first.

# --- Core ExperimentConfig fields ---
STATE_DIM = None  # REQUIRED
N_SAMPLES = None  # REQUIRED
TRAIN_FRACTION = None  # OPTIONAL
TEST_FRACTION = None  # OPTIONAL
STEP_RULE = None  # REQUIRED
COMPUTE_BACKEND = None  # OPTIONAL: numpy | jax
OBJECTIVE = None  # REQUIRED
THETA0 = None  # REQUIRED
BATCH_SIZE = None  # OPTIONAL
SEED = None  # REQUIRED
T_STEPS = None  # REQUIRED
STEP_SIZE = None  # REQUIRED
GRAD_NORM_TOL = None  # OPTIONAL
FTOL = None  # OPTIONAL
INITIAL_CONSTR_PENALTY = None  # OPTIONAL
ACCEPTANCE_FLOOR = None  # OPTIONAL
ACCEPTANCE_PENALTY_WEIGHT = None  # OPTIONAL
ACCEPTANCE_PENALTY_TEMPERATURE = None  # OPTIONAL
LAGRANGIAN_LAMBDA = None  # OPTIONAL
PROXIMAL_WEIGHT = None  # OPTIONAL
U_REFERENCE = None  # OPTIONAL
U_REFERENCE_SOURCE = None  # OPTIONAL: array | constant | historical
U_REFERENCE_VALUE = None  # OPTIONAL
SUPPORT_WEIGHT = None  # OPTIONAL
SIGMA_PROVIDER = None  # OPTIONAL
CONSTANT_U_BASELINES = None  # OPTIONAL
SIGMA = None  # REQUIRED
N_GRAD_SAMPLES = None  # REQUIRED
VERBOSE = None  # OPTIONAL
PLOT = None  # OPTIONAL
ENABLED_ESTIMATORS = None  # REQUIRED
PERTURBATION_SPACE = None  # REQUIRED
WANDB_ENABLED = None  # OPTIONAL
WANDB_PROJECT = None  # OPTIONAL
WANDB_ENTITY = None  # OPTIONAL
WANDB_GROUP = None  # OPTIONAL
WANDB_JOB_TYPE = None  # OPTIONAL
WANDB_TAGS = None  # OPTIONAL
WANDB_MODE = None  # OPTIONAL
WANDB_LOG_PLOTS = None  # OPTIONAL
WANDB_ESTIMATOR_ALLOWLIST = None  # OPTIONAL
CORRECTNESS = None  # OPTIONAL

# --- Objective parameter placeholders ---
POLICY = None  # REQUIRED when building objective

FIXED_BETA_1 = None  # REQUIRED for fixed regression
FIXED_BETA_2 = None  # REQUIRED for fixed regression
FIXED_BETA_3 = None  # REQUIRED for fixed regression
FIXED_BETA_4 = None  # REQUIRED for fixed regression

PLANTED_ALPHA = None  # REQUIRED for planted logistic
PLANTED_BETA = None  # REQUIRED for planted logistic
PLANTED_BIAS = None  # REQUIRED for planted logistic
PLANTED_U_STAR = None  # REQUIRED for planted logistic

# --- CorrectnessSpec parameter placeholders ---
CORRECTNESS_GRADIENT_SOURCE = None  # OPTIONAL: exact | numdiff | none
CORRECTNESS_NUMDIFF_METHOD = None  # OPTIONAL: central | forward | backward
CORRECTNESS_NUMDIFF_STEP = None  # OPTIONAL
CORRECTNESS_NUMDIFF_AGGREGATE = None  # OPTIONAL: per-sample | batch
CORRECTNESS_NUMDIFF_BOUNDS = None  # OPTIONAL

# --- Uncomment this block to build a runnable config ---
# import numpy as np
# from experiments.config import (
#     CorrectnessSpec,
#     build_experiment_config,
#     canonical_runtime_block,
#     canonical_training_block,
#     make_fixed_regression_objective,
#     make_planted_logistic_objective,
#     make_softmax_policy,
# )
#
# POLICY = make_softmax_policy()
# THETA0 = np.zeros(POLICY.theta_dim(STATE_DIM), dtype=float)
#
# # Choose exactly one objective block.
# OBJECTIVE = make_fixed_regression_objective(
#     policy=POLICY,
#     beta_1=FIXED_BETA_1,
#     beta_2=FIXED_BETA_2,
#     beta_3=FIXED_BETA_3,
#     beta_4=FIXED_BETA_4,
# )
# # OBJECTIVE = make_planted_logistic_objective(
# #     policy=POLICY,
# #     alpha=PLANTED_ALPHA,
# #     beta=PLANTED_BETA,
# #     bias=PLANTED_BIAS,
# #     u_star=PLANTED_U_STAR,
# # )
#
# TRAINING = canonical_training_block(
#     n_samples=N_SAMPLES,
#     train_fraction=TRAIN_FRACTION,
#     test_fraction=TEST_FRACTION,
#     step_rule=STEP_RULE,
#     compute_backend=COMPUTE_BACKEND,
#     t_steps=T_STEPS,
#     step_size=STEP_SIZE,
#     sigma=SIGMA,
#     n_grad_samples=N_GRAD_SAMPLES,
#     enabled_estimators=ENABLED_ESTIMATORS,
#     constant_u_baselines=CONSTANT_U_BASELINES,
#     perturbation_space=PERTURBATION_SPACE,
#     batch_size=BATCH_SIZE,
#     grad_norm_tol=GRAD_NORM_TOL,
#     ftol=FTOL,
#     initial_constr_penalty=INITIAL_CONSTR_PENALTY,
#     acceptance_floor=ACCEPTANCE_FLOOR,
#     acceptance_penalty_weight=ACCEPTANCE_PENALTY_WEIGHT,
#     acceptance_penalty_temperature=ACCEPTANCE_PENALTY_TEMPERATURE,
#     lagrangian_lambda=LAGRANGIAN_LAMBDA,
#     proximal_weight=PROXIMAL_WEIGHT,
#     u_reference=U_REFERENCE,
#     u_reference_source=U_REFERENCE_SOURCE,
#     u_reference_value=U_REFERENCE_VALUE,
#     support_weight=SUPPORT_WEIGHT,
#     sigma_provider=SIGMA_PROVIDER,
# )
#
# RUNTIME = canonical_runtime_block(
#     plot=PLOT,
#     verbose=VERBOSE,
#     wandb_enabled=WANDB_ENABLED,
#     wandb_project=WANDB_PROJECT,
#     wandb_entity=WANDB_ENTITY,
#     wandb_group=WANDB_GROUP,
#     wandb_job_type=WANDB_JOB_TYPE,
#     wandb_tags=WANDB_TAGS,
#     wandb_mode=WANDB_MODE,
#     wandb_log_plots=WANDB_LOG_PLOTS,
#     wandb_estimator_allowlist=WANDB_ESTIMATOR_ALLOWLIST,
# )
#
# CORRECTNESS = CorrectnessSpec(
#     gradient_source=CORRECTNESS_GRADIENT_SOURCE,
#     numdiff_method=CORRECTNESS_NUMDIFF_METHOD,
#     numdiff_step=CORRECTNESS_NUMDIFF_STEP,
#     numdiff_aggregate=CORRECTNESS_NUMDIFF_AGGREGATE,
#     numdiff_bounds=CORRECTNESS_NUMDIFF_BOUNDS,
# )
#
# CONFIG = build_experiment_config(
#     seed=SEED,
#     state_dim=STATE_DIM,
#     objective=OBJECTIVE,
#     theta0=THETA0,
#     training=TRAINING,
#     runtime=RUNTIME,
#     correctness=CORRECTNESS,
# )

# Keep this template non-runnable by default.
CONFIG = None
