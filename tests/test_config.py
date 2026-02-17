from __future__ import annotations

import pytest

from experiments.config import ExperimentConfig


def test_beta_2_must_be_negative() -> None:
    with pytest.raises(
        ValueError,
        match="beta_2 must be negative; acceptance probability should decrease as policy value increases.",
    ):
        ExperimentConfig(beta_2=0.0)

    with pytest.raises(
        ValueError,
        match="beta_2 must be negative; acceptance probability should decrease as policy value increases.",
    ):
        ExperimentConfig(beta_2=0.5)


def test_state_dim_defaults() -> None:
    config = ExperimentConfig(state_dim=5)
    assert config.beta_1.size >= 5
    assert config.beta_3.size >= 5
    assert config.policy_spec.theta.size >= 6
