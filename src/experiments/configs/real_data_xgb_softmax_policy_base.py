"""XGBoost-backed base config using a softmax policy on real insurance data.

Uses trained XGBoost artifacts (classifier + regressor) with the first 5,000
rows of the real dataset as the fixed state distribution. The objective owns
the acceptance-side preprocessing from the bundled pickle so raw CSV rows stay
at the optimization boundary.

XGBoost has no analytical gradient; d_acceptance/du is computed via central
finite differences inside ModelBasedObjective. first_order is disabled.
"""

from __future__ import annotations

import numpy as np

from data.loader import (
    ACCEPTANCE_STATE_COLS,
    FEATURE_COLS_XGB,
    LOSS_FEATURE_COLS,
    load_model_artifacts,
    load_x_array,
    sample_csv_row_indices,
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
SEED = 42

_acceptance_model, _loss_model = load_model_artifacts("xgb")
_policy = SoftmaxPolicy()

THETA0 = np.zeros(_policy.theta_dim(_acceptance_model.policy_feature_dim()), dtype=float)

TRAINING = canonical_training_block(
    n_samples=5000,
    step_rule="l-bfgs-b",
    t_steps=1000,
    step_size=0.01,
    sigma=0.05,
    n_grad_samples=50,
    enabled_estimators=("finite_difference", "spsa", "stein_difference"),
    perturbation_space="u",
    grad_norm_tol=1e-6,
)

RUNTIME = canonical_runtime_block(
    plot=True,
    verbose=True,
    wandb_enabled=True,
    wandb_project="xgb-softmax-policy-unconstrained",
)

CORRECTNESS = CorrectnessSpec(gradient_source="none")
_ROW_INDICES = sample_csv_row_indices("xgb", n_rows=TRAINING["n_samples"], seed=SEED)

CONFIG = build_experiment_config(
    seed=SEED,
    state_dim=STATE_DIM,
    x_fixed=load_x_array("xgb", row_indices=_ROW_INDICES),
    x_fixed_row_indices=_ROW_INDICES,
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
