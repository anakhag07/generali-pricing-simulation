"""XGBoost CSV-backed experiment config using pre-computed predictions (u-space only).

Uses pre-computed prob_acceptance and Y_hat from the XGBoost CSV dataset.
Evaluates f(u) by filtering rows where |U - u| < tol and averaging predictions.
The SoftmaxPolicy maps state -> u; value is computed at the mean policy action.

first_order is disabled (CSV path has no analytical gradient).
"""

from __future__ import annotations

import numpy as np

from data.loader import load_csv_dataset
from experiments.config import (
    build_experiment_config,
    canonical_runtime_block,
    canonical_training_block,
    make_csv_objective,
    make_softmax_policy,
)

STATE_DIM = 1

_df = load_csv_dataset("xgb")
_policy = make_softmax_policy()

_x_fixed = np.zeros((1, STATE_DIM), dtype=float)

THETA0 = np.array([0.4, 0.0], dtype=float)

TRAINING = canonical_training_block(
    n_samples=1,
    step_rule="l-bfgs-b",
    t_steps=500,
    step_size=0.01,
    sigma=0.05,
    n_grad_samples=20,
    enabled_estimators=("finite_difference", "spsa", "gauss_stein", "stein_difference"),
    perturbation_space="u",
    grad_norm_tol=1e-6,
)

RUNTIME = canonical_runtime_block(
    plot=True,
    verbose=False,
    wandb_enabled=False,
)

CONFIG = build_experiment_config(
    seed=42,
    state_dim=STATE_DIM,
    x_fixed=_x_fixed,
    objective=make_csv_objective(df=_df, policy=_policy, tol=0.005),
    theta0=THETA0,
    training=TRAINING,
    runtime=RUNTIME,
)
