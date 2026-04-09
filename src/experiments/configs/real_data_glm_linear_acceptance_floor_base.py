"""GLM-backed constrained diagnostic config with a mean-acceptance floor.

Uses the same trained GLM artifacts and raw state batch as ``real_data_glm_linear_base``
but adds a smooth config-driven penalty so mean acceptance stays above 90% of the
observed acceptance level in the exported notebook CSV.
"""

from __future__ import annotations

import numpy as np

from data.loader import (
    ACCEPTANCE_STATE_COLS,
    FEATURE_COLS_GLM,
    LOSS_FEATURE_COLS,
    extract_glm_u_coef,
    load_mean_observed_acceptance,
    load_model_artifacts,
    load_x_array,
)
from experiments.config import (
    CorrectnessSpec,
    build_experiment_config,
    canonical_runtime_block,
    canonical_training_block,
    make_model_based_objective,
)
from objective.policy import LinearPolicy

STATE_DIM = len(FEATURE_COLS_GLM)

_acceptance_model, _loss_model = load_model_artifacts("glm")
_u_coef = extract_glm_u_coef(_acceptance_model)
_policy = LinearPolicy()
_acceptance_floor = 0.9 * load_mean_observed_acceptance("glm")

THETA0 = np.array([1.1] + [0.0] * _acceptance_model.policy_feature_dim(), dtype=float)

TRAINING = canonical_training_block(
    n_samples=5000,
    step_rule="l-bfgs-b",
    t_steps=1000,
    step_size=0.01,
    sigma=0.01,
    n_grad_samples=10,
    enabled_estimators=("first_order", "finite_difference", "spsa", "stein_difference"),
    perturbation_space="u",
    grad_norm_tol=1e-6,
    acceptance_floor=_acceptance_floor,
    acceptance_penalty_weight=1e4,
    acceptance_penalty_temperature=0.05,
)

RUNTIME = canonical_runtime_block(
    plot=True,
    verbose=True,
    wandb_enabled=True,
)

CORRECTNESS = CorrectnessSpec(gradient_source="none")

CONFIG = build_experiment_config(
    seed=8,
    state_dim=STATE_DIM,
    x_fixed=load_x_array("glm", n_rows=5000),
    objective=make_model_based_objective(
        policy=_policy,
        acceptance_model=_acceptance_model,
        loss_model=_loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=9,
        u_coef=_u_coef,
    ),
    theta0=THETA0,
    training=TRAINING,
    runtime=RUNTIME,
    correctness=CORRECTNESS,
)
