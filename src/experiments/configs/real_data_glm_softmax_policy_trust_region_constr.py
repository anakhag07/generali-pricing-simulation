"""GLM-backed softmax-policy config with a trust-region acceptance floor.

Uses the same trained GLM artifacts and fixed state batch as
``real_data_glm_softmax_policy_base`` but enforces the historical observed
acceptance level directly with SciPy's trust-region constrained solver.
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

STATE_DIM = len(FEATURE_COLS_GLM)
SEED = 42

_acceptance_model, _loss_model = load_model_artifacts("glm")
_u_coef = extract_glm_u_coef(_acceptance_model)
_policy = SoftmaxPolicy()
_acceptance_floor = load_mean_observed_acceptance("glm")

THETA0 = np.zeros(_policy.theta_dim(_acceptance_model.policy_feature_dim()), dtype=float)

TRAINING = canonical_training_block(
    n_samples=5000,
    step_rule="trust-constr",
    t_steps=1000,
    step_size=0.01,
    sigma=0.05,
    n_grad_samples=50,
    enabled_estimators=(
        "first_order",
        "finite_difference",
        "spsa",
        "stein_difference",
    ),
    perturbation_space="u", 
    grad_norm_tol=1e-6,
    acceptance_floor=_acceptance_floor,
    initial_constr_penalty=1.0,
    constant_u_baselines=[-0.5, -0.3, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1],
)

RUNTIME = canonical_runtime_block(
    plot=True,
    verbose=True,
    wandb_enabled=True,
    wandb_project='glm-softmax-policy-trust-region-constrained'
)

CORRECTNESS = CorrectnessSpec(gradient_source="none")
_ROW_INDICES = sample_csv_row_indices("glm", n_rows=TRAINING["n_samples"], seed=SEED)

CONFIG = build_experiment_config(
    seed=SEED,
    state_dim=STATE_DIM,
    x_fixed=load_x_array("glm", row_indices=_ROW_INDICES),
    x_fixed_row_indices=_ROW_INDICES,
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
