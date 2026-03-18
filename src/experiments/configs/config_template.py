"""Template scaffold for creating new experiment presets.

Copy this file, replace ``None`` placeholders with concrete values, and then
register the new preset in ``experiments.configs.__init__``.
"""

from __future__ import annotations

# --- Core ExperimentConfig fields ---
STATE_DIM = None
N_SAMPLES = None
STEP_RULE = None
OBJECTIVE = None
THETA0 = None
BATCH_SIZE = None
SEED = None
T_STEPS = None
STEP_SIZE = None
GRAD_NORM_TOL = None
FTOL = None
SIGMA = None
N_GRAD_SAMPLES = None
VERBOSE = None
PLOT = None
PLOT_DIR = None
ENABLED_ESTIMATORS = None
WANDB_ENABLED = None
WANDB_PROJECT = None
WANDB_ENTITY = None
WANDB_GROUP = None
WANDB_JOB_TYPE = None
WANDB_TAGS = None
WANDB_MODE = None
WANDB_LOG_PLOTS = None
WANDB_ESTIMATOR_ALLOWLIST = None
CORRECTNESS = None

# --- Objective parameter placeholders ---
POLICY = None

FIXED_BETA_1 = None
FIXED_BETA_2 = None
FIXED_BETA_3 = None
FIXED_BETA_4 = None

PLANTED_ALPHA = None
PLANTED_BETA = None
PLANTED_BIAS = None
PLANTED_U_STAR = None

# --- CorrectnessSpec parameter placeholders ---
CORRECTNESS_GRADIENT_SOURCE = None
CORRECTNESS_NUMDIFF_METHOD = None
CORRECTNESS_NUMDIFF_STEP = None
CORRECTNESS_NUMDIFF_AGGREGATE = None
CORRECTNESS_NUMDIFF_BOUNDS = None

# --- Structured template blocks ---
FIXED_REGRESSION_OBJECTIVE_TEMPLATE = {
    "policy": POLICY,
    "beta_1": FIXED_BETA_1,
    "beta_2": FIXED_BETA_2,
    "beta_3": FIXED_BETA_3,
    "beta_4": FIXED_BETA_4,
}

PLANTED_LOGISTIC_OBJECTIVE_TEMPLATE = {
    "policy": POLICY,
    "alpha": PLANTED_ALPHA,
    "beta": PLANTED_BETA,
    "bias": PLANTED_BIAS,
    "u_star": PLANTED_U_STAR,
}

CORRECTNESS_TEMPLATE = {
    "gradient_source": CORRECTNESS_GRADIENT_SOURCE,
    "numdiff_method": CORRECTNESS_NUMDIFF_METHOD,
    "numdiff_step": CORRECTNESS_NUMDIFF_STEP,
    "numdiff_aggregate": CORRECTNESS_NUMDIFF_AGGREGATE,
    "numdiff_bounds": CORRECTNESS_NUMDIFF_BOUNDS,
}

EXPERIMENT_CONFIG_TEMPLATE = {
    "state_dim": STATE_DIM,
    "n_samples": N_SAMPLES,
    "step_rule": STEP_RULE,
    "objective": OBJECTIVE,
    "theta0": THETA0,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "t_steps": T_STEPS,
    "step_size": STEP_SIZE,
    "grad_norm_tol": GRAD_NORM_TOL,
    "ftol": FTOL,
    "sigma": SIGMA,
    "n_grad_samples": N_GRAD_SAMPLES,
    "verbose": VERBOSE,
    "plot": PLOT,
    "plot_dir": PLOT_DIR,
    "enabled_estimators": ENABLED_ESTIMATORS,
    "wandb_enabled": WANDB_ENABLED,
    "wandb_project": WANDB_PROJECT,
    "wandb_entity": WANDB_ENTITY,
    "wandb_group": WANDB_GROUP,
    "wandb_job_type": WANDB_JOB_TYPE,
    "wandb_tags": WANDB_TAGS,
    "wandb_mode": WANDB_MODE,
    "wandb_log_plots": WANDB_LOG_PLOTS,
    "wandb_estimator_allowlist": WANDB_ESTIMATOR_ALLOWLIST,
    "correctness": CORRECTNESS,
}

# Optional placeholder if you keep this file and choose to fill it in place.
CONFIG = None
