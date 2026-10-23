"""Build the existing ExperimentConfig around a prepared fitted-band objective."""

from __future__ import annotations

import numpy as np

from objective.banded import BandObjective
from experiments.config import ExperimentConfig, CorrectnessSpec
from .real_data_factory import make_policy


def build_band_config(*, fitted_band, band_artifact, policy_kind, endpoint, sense,
                      step_rule, enabled_estimators, perturbation_space,
                      feature_order="linear", softmax_action_bounds=None, **overrides):
    """Compose a declared band/policy target without altering optimizer or seed semantics."""
    if "constant" in enabled_estimators:
        raise ValueError("Use policy_kind=constant with a requested gradient estimator; automatic constant-policy copies are unsupported for fitted bands.")
    if overrides.get("compute_backend", "numpy") != "numpy":
        raise ValueError("Fitted-band adapters currently require compute_backend=numpy.")
    if "first_order" in enabled_estimators and fitted_band.action_derivatives is None:
        raise ValueError("This fitted band does not support first_order; choose a query estimator explicitly.")
    if fitted_band.metadata["method"] == "support_grid":
        grid = fitted_band.state["action_grid"]
        if policy_kind != "softmax" or softmax_action_bounds is None:
            raise ValueError("Bounded support grids require softmax with explicit action bounds.")
        if not grid[0] <= softmax_action_bounds[0] < softmax_action_bounds[1] <= grid[-1]:
            raise ValueError("Policy bounds must lie inside the support-grid domain.")
        if perturbation_space != "theta":
            raise ValueError("Support grids require theta-space perturbations to keep every query inside the action domain.")
    policy = make_policy(policy_kind, feature_order, softmax_action_bounds)
    objective = BandObjective(fitted_band, policy, endpoint=endpoint, sense=sense,
                              artifact=band_artifact, acceptance_floor=overrides.get("acceptance_floor"))
    if overrides.get("acceptance_penalty_weight") is not None:
        from objective import AcceptancePenaltyObjective
        objective = AcceptancePenaltyObjective(objective, overrides.get("acceptance_floor"),
            overrides["acceptance_penalty_weight"], overrides.get("acceptance_penalty_temperature", 0.01))
    elif overrides.get("lagrangian_lambda") is not None:
        from objective import AcceptanceLagrangianObjective
        objective = AcceptanceLagrangianObjective(objective, overrides.get("acceptance_floor"), overrides["lagrangian_lambda"])
    X = fitted_band.state["X"]
    X = X if X.shape[1] else np.zeros((len(X), 1))
    for key in ("state_dim", "n_samples", "x_fixed", "x_fixed_row_indices"):
        if key in overrides:
            raise ValueError(f"{key} is bound to the prepared band setting; change the setting instead.")
    return ExperimentConfig(state_dim=X.shape[1], n_samples=len(X), x_fixed=X,
        x_fixed_row_indices=fitted_band.state.get("row_indices"), objective=objective,
        step_rule=step_rule, enabled_estimators=tuple(enabled_estimators),
        perturbation_space=perturbation_space,
        correctness=overrides.pop("correctness", CorrectnessSpec(gradient_source="none")), **overrides)
