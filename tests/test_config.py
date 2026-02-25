from __future__ import annotations

import numpy as np
import pytest

from data.fixed_objective import FixedRegressionObjective
from experiments.config import ExperimentConfig
from experiments.defaults import default_policy_spec


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
    )
    assert config.state_dim == state_dim


def test_log_steps_default_and_override() -> None:
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
    )
    assert config_default.log_steps is True

    config_quiet = ExperimentConfig(
        state_dim=1,
        objective_model=objective_model,
        policy_spec=default_policy_spec(1),
        log_steps=False,
    )
    assert config_quiet.log_steps is False
