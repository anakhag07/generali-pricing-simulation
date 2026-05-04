"""GLM-backed base config using a 2-layer MLP policy.

Mirrors `real_data_glm_softmax_policy_base.py` but swaps the softmax/quadratic
linear policy for an `MLPPolicy` with two hidden layers of width 16 and tanh
activations. Bounded action `u = 0.5 - sigmoid(z)` is preserved so the GLM
analytical first-order gradient (chain rule via `u_coef`) still applies.

Initialization is Glorot-uniform with the same SEED so theta is non-trivial
(zero init would collapse hidden units by symmetry).

`finite_difference` is omitted: at theta_dim ~= 497 the FD gradient call costs
~2 * theta_dim objective evals (~3 min/grad on this dataset, ~50 h per run).
Use first-order, SPSA, and stein-difference for tractable comparison.
"""

from __future__ import annotations

import numpy as np

from data.loader import (
    ACCEPTANCE_STATE_COLS,
    FEATURE_COLS_GLM,
    LOSS_FEATURE_COLS,
    extract_glm_u_coef,
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
from objective.policy import IdentityFeatureMap, MLPPolicy, mlp_init_theta

STATE_DIM = len(FEATURE_COLS_GLM)
SEED = 42
HIDDEN = 16

_acceptance_model, _loss_model = load_model_artifacts("glm")
_u_coef = extract_glm_u_coef(_acceptance_model)
_policy = MLPPolicy(feature_map=IdentityFeatureMap(), hidden=HIDDEN)

_d_in = _acceptance_model.policy_feature_dim()
THETA0 = mlp_init_theta(np.random.default_rng(SEED), d_in=_d_in, hidden=HIDDEN)

TRAINING = canonical_training_block(
    n_samples=5000,
    step_rule="l-bfgs-b",
    t_steps=1000,
    step_size=0.01,
    sigma=0.05,
    n_grad_samples=50,
    enabled_estimators=(
        "first_order",
        "spsa",
        "stein_difference",
    ),
    perturbation_space="u",
    grad_norm_tol=1e-6,
)

RUNTIME = canonical_runtime_block(
    plot=True,
    verbose=True,
    wandb_enabled=True,
    wandb_project="glm-mlp-policy-unconstrained",
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
