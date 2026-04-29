"""XGBoost-backed softmax-policy config with a trust-region acceptance floor.

Uses the same trained XGBoost artifacts and raw state batch as
``real_data_xgb_base`` but enforces the historical observed acceptance level
directly with SciPy's trust-region constrained solver.
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
from objective.policy import SoftmaxPolicy

STATE_DIM = len(FEATURE_COLS_XGB)

_acceptance_model, _loss_model = load_model_artifacts("xgb")
_policy = SoftmaxPolicy()
_acceptance_floor = load_mean_observed_acceptance("xgb")

THETA0 = np.zeros(_policy.theta_dim(_acceptance_model.policy_feature_dim()), dtype=float)

TRAINING = canonical_training_block(
    n_samples=5000,
    step_rule="trust-constr",
    t_steps=1000,
    step_size=0.01,
    sigma=0.05,
    n_grad_samples=50,
    enabled_estimators=("finite_difference", "spsa", "stein_difference"),
    perturbation_space="u",
    grad_norm_tol=1e-6,
    acceptance_floor=_acceptance_floor,
    initial_constr_penalty=1.0,
)

RUNTIME = canonical_runtime_block(
    plot=True,
    verbose=True,
    wandb_enabled=True,
    wandb_project="xgb-softmax-policy-trust-region-constrained",
)

CORRECTNESS = CorrectnessSpec(gradient_source="none")

CONFIG = build_experiment_config(
    seed=42,
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
    ),
    theta0=THETA0,
    training=TRAINING,
    runtime=RUNTIME,
    correctness=CORRECTNESS,
)
