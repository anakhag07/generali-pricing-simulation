from __future__ import annotations

import numpy as np
import pytest

from experiments.config import ExperimentConfig
from experiments.defaults import default_policy_spec
from objective.fixed_objective import FixedRegressionObjective


def test_beta_2_must_be_negative() -> None:
    with pytest.raises(
        ValueError,
        match="beta_2 must be negative; acceptance probability should decrease as policy value increases.",
    ):
        FixedRegressionObjective.from_parameters(
            beta_1=[1.0],
            beta_2=0.0,
            beta_3=[1.0],
            beta_4=1.0,
        )

    with pytest.raises(
        ValueError,
        match="beta_2 must be negative; acceptance probability should decrease as policy value increases.",
    ):
        FixedRegressionObjective.from_parameters(
            beta_1=[1.0],
            beta_2=0.5,
            beta_3=[1.0],
            beta_4=1.0,
        )


def test_state_dim_requires_matching_objective() -> None:
    state_dim = 5
    beta_1 = np.linspace(0.1, 0.5, num=state_dim, dtype=float)
    beta_3 = np.linspace(0.1, 0.5, num=state_dim, dtype=float)
    objective_model = FixedRegressionObjective.from_parameters(
        beta_1=beta_1,
        beta_2=-0.5,
        beta_3=beta_3,
        beta_4=0.4,
    )
    config = ExperimentConfig(
        state_dim=state_dim,
        objective_model=objective_model,
        policy_spec=default_policy_spec(state_dim),
        n_samples=5,
        step_rule="constant",
    )
    assert config.state_dim == state_dim


def test_verbose_default_and_override() -> None:
    objective_model = FixedRegressionObjective.from_parameters(
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    config_default = ExperimentConfig(
        state_dim=1,
        objective_model=objective_model,
        policy_spec=default_policy_spec(1),
        n_samples=5,
        step_rule="constant",
    )
    assert config_default.verbose is False

    config_verbose = ExperimentConfig(
        state_dim=1,
        objective_model=objective_model,
        policy_spec=default_policy_spec(1),
        n_samples=5,
        step_rule="constant",
        verbose=True,
    )
    assert config_verbose.verbose is True


def test_enabled_estimators_validation() -> None:
    objective_model = FixedRegressionObjective.from_parameters(
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    with pytest.raises(ValueError, match="Unknown estimators"):
        ExperimentConfig(
            state_dim=1,
            objective_model=objective_model,
            policy_spec=default_policy_spec(1),
            n_samples=5,
            step_rule="constant",
            enabled_estimators=("not-a-method",),
        )

    with pytest.raises(ValueError, match="enabled_estimators must include at least one"):
        ExperimentConfig(
            state_dim=1,
            objective_model=objective_model,
            policy_spec=default_policy_spec(1),
            n_samples=5,
            step_rule="constant",
            enabled_estimators=(),
        )


def test_step_rule_validation() -> None:
    objective_model = FixedRegressionObjective.from_parameters(
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )

    with pytest.raises(ValueError, match="step_rule must be one of"):
        ExperimentConfig(
            state_dim=1,
            objective_model=objective_model,
            policy_spec=default_policy_spec(1),
            n_samples=5,
            step_rule="unknown",
        )

    with pytest.raises(ValueError, match="step_size must be positive"):
        ExperimentConfig(
            state_dim=1,
            objective_model=objective_model,
            policy_spec=default_policy_spec(1),
            n_samples=5,
            step_rule="constant",
            step_size=0.0,
        )


def test_grad_norm_tol_validation() -> None:
    objective_model = FixedRegressionObjective.from_parameters(
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )

    with pytest.raises(ValueError, match="grad_norm_tol must be positive"):
        ExperimentConfig(
            state_dim=1,
            objective_model=objective_model,
            policy_spec=default_policy_spec(1),
            n_samples=5,
            step_rule="constant",
            grad_norm_tol=0.0,
        )
