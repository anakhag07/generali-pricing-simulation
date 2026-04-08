"""XGBoost-backed experiment config using real insurance data (pickle path).

Uses trained XGBoost artifacts (classifier + regressor) with the first 5,000
rows of the real dataset as the fixed state distribution.
The policy consumes the acceptance bundle's processed state features while the
objective still evaluates the black-box models on raw CSV rows.

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
)
from experiments.config import (
    CorrectnessSpec,
    build_experiment_config,
    canonical_runtime_block,
    canonical_training_block,
    make_model_based_objective,
)
from objective.policy import FeatureProcessedPolicy, SoftmaxPolicy

STATE_DIM = len(FEATURE_COLS_XGB)

_acceptance_model, _loss_model = load_model_artifacts("xgb")
_policy = FeatureProcessedPolicy(
    policy=SoftmaxPolicy(),
    raw_feature_cols=tuple(FEATURE_COLS_XGB),
    preprocess_feature_cols=_acceptance_model.x_feature_cols,
    preprocessor=_acceptance_model.preprocessor,
)

THETA0 = np.array([0.4] + [0.01] * _acceptance_model.policy_feature_dim(), dtype=float)

TRAINING = canonical_training_block(
    n_samples=5000,
    step_rule="l-bfgs-b",
    t_steps=1000,
    step_size=0.01,
    sigma=0.05,
    n_grad_samples=50,
    enabled_estimators=("finite_difference", "spsa", "gauss_stein", "stein_difference"),
    perturbation_space="u",
    grad_norm_tol=1e-6,
)

RUNTIME = canonical_runtime_block(
    plot=True,
    verbose=True,
    wandb_enabled=True,
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
        u_coef=None,  # XGBoost: use numerical FD for d_acceptance/du
    ),
    theta0=THETA0,
    training=TRAINING,
    runtime=RUNTIME,
    correctness=CORRECTNESS,
)
