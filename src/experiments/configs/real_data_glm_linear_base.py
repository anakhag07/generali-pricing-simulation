"""GLM-backed diagnostic config using a linear policy on real insurance data.

Uses the same trained GLM artifacts and fixed state batch as ``real_data_glm_base``
but swaps in ``LinearPolicy`` so the first-order path is not masked by the
softmax policy's upper-bound saturation at ``u = 1.5``.

The initial theta sets a constant action ``u = 1.1`` across the batch, which
starts near the GLM training U range before optimization moves the policy.
"""

from __future__ import annotations

import numpy as np

from data.loader import (
    ACCEPTANCE_STATE_COLS,
    LOSS_FEATURE_COLS,
    extract_glm_u_coef,
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

STATE_DIM = 12  # 9 base + premium + X_prev_renewal_perc + X_year

_acceptance_model, _loss_model = load_model_artifacts("glm")
_u_coef = extract_glm_u_coef(_acceptance_model)

THETA0 = np.array([1.1] + [0.0] * STATE_DIM, dtype=float)  # length 13; initial u == 1.1

TRAINING = canonical_training_block(
    n_samples=5000,
    step_rule="l-bfgs-b",
    t_steps=1000,
    step_size=0.01,
    sigma=0.05,
    n_grad_samples=50,
    enabled_estimators=("first_order", "finite_difference", "spsa", "gauss_stein", "stein_difference"),
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
    seed=8,
    state_dim=STATE_DIM,
    x_fixed=load_x_array("glm", n_rows=5000),
    objective=make_model_based_objective(
        policy=LinearPolicy(),
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
