"""GLM-backed experiment config using real insurance data (pickle path).

Uses trained GLM artifacts (logistic regression + linear regression) with the
first 5,000 rows of the real dataset as the fixed state distribution. The
objective owns the acceptance-side preprocessing from the bundled pickle so raw
CSV rows stay at the optimization boundary.

GLM enables an analytical first-order gradient (u_coef extracted from the
logistic regression pipeline). All 5 estimators are enabled for comparison.

Note: SoftmaxPolicy outputs centered u in (-0.5, 0.5), so this preset starts
from theta = 0 to initialize at the baseline premium multiplier (u = 0 means
revenue uses 1.0 * premium) with the largest policy slope.
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
from objective.policy import SoftmaxPolicy

STATE_DIM = len(FEATURE_COLS_GLM)

_acceptance_model, _loss_model = load_model_artifacts("glm")
_u_coef = extract_glm_u_coef(_acceptance_model)
_policy = SoftmaxPolicy()
_acceptance_floor = load_mean_observed_acceptance("glm")

THETA0 = np.zeros(_acceptance_model.policy_feature_dim() + 1, dtype=float)

TRAINING = canonical_training_block(
    n_samples=5000,
    step_rule="trust-constr",
    t_steps=1000,
    step_size=0.01,
    sigma=0.05,
    n_grad_samples=50,
    enabled_estimators=("first_order", "finite_difference", "spsa", "stein_difference"),
    perturbation_space="u",
    grad_norm_tol=1e-6,
    acceptance_floor=_acceptance_floor,
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
