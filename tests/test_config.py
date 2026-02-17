from __future__ import annotations

import pytest

from data.fixed_objective import FixedRegressionObjective
from experiments.config import ExperimentConfig


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


def test_state_dim_defaults() -> None:
    config = ExperimentConfig(state_dim=5)
    assert config.objective_model is not None
    assert isinstance(config.objective_model, FixedRegressionObjective)
    assert config.objective_model.acceptance.beta_1.size >= 5
    assert config.objective_model.loss.beta_3.size >= 5
    assert config.policy_spec.theta.size >= 6
