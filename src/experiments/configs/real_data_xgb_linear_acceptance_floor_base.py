"""XGBoost-backed constrained diagnostic config with a mean-acceptance floor.

Uses the same trained XGBoost artifacts and raw state batch as
``real_data_xgb_base`` but swaps in ``LinearPolicy`` and adds a smooth
config-driven penalty so mean acceptance stays above the observed
acceptance level in the exported notebook CSV.
"""

from __future__ import annotations

import numpy as np

from data.loader import (
    ACCEPTANCE_STATE_COLS,
    FEATURE_COLS_XGB,
    LOSS_FEATURE_COLS,
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

STATE_DIM = len(FEATURE_COLS_XGB)

_acceptance_model, _loss_model = load_model_artifacts("xgb")
_policy = LinearPolicy()
_acceptance_floor = load_mean_observed_acceptance("xgb")

# Start from a constant centered action so the linear policy does not begin clipped.
# Under the shifted revenue term this is a 1.2 premium multiplier.
THETA0 = np.zeros(_policy.theta_dim(_acceptance_model.policy_feature_dim()), dtype=float)
THETA0[0] = 0.2

TRAINING = canonical_training_block(
    n_samples=5000,
    step_rule="l-bfgs-b",
    t_steps=1000,
    step_size=0.01,
    sigma=0.01,
    n_grad_samples=10,
    enabled_estimators=("finite_difference", "spsa", "stein_difference"),
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
    x_fixed=load_x_array("xgb", n_rows=5000),
    objective=make_model_based_objective(
        policy=_policy,
        acceptance_model=_acceptance_model,
        loss_model=_loss_model,
        acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
        loss_cols=tuple(LOSS_FEATURE_COLS),
        premium_col=9,
        u_coef=None,
        u_bounds=(-0.05, 0.5),
    ),
    theta0=THETA0,
    training=TRAINING,
    runtime=RUNTIME,
    correctness=CORRECTNESS,
)
