"""Tests for verbose config field in ExperimentConfig."""

from __future__ import annotations

import numpy as np

from data.fixed_objective import FixedRegressionObjective
from experiments.config import ExperimentConfig
from optimization.policy import POLICY_LINEAR, PolicySpec


def _make_config(verbose: bool = False) -> ExperimentConfig:
    """Create a minimal ExperimentConfig with specified verbose setting."""
    return ExperimentConfig(
        state_dim=2,
        objective_model=FixedRegressionObjective.from_parameters(
            beta_1=np.array([0.1, 0.2]),
            beta_2=-0.5,
            beta_3=np.array([0.05, 0.1]),
            beta_4=0.3,
        ),
        policy_spec=PolicySpec(
            theta=np.array([0.1, 0.01, 0.01]),
            kind=POLICY_LINEAR,
        ),
        n_samples=5,
        step_rule="constant",
        verbose=verbose,
    )


def test_verbose_default_false() -> None:
    """Default verbose should be False."""
    config = ExperimentConfig(
        state_dim=2,
        objective_model=FixedRegressionObjective.from_parameters(
            beta_1=np.array([0.1, 0.2]),
            beta_2=-0.5,
            beta_3=np.array([0.05, 0.1]),
            beta_4=0.3,
        ),
        policy_spec=PolicySpec(
            theta=np.array([0.1, 0.01, 0.01]),
            kind=POLICY_LINEAR,
        ),
        n_samples=5,
        step_rule="constant",
    )
    assert config.verbose is False


def test_verbose_can_be_set_true() -> None:
    """verbose=True should be accepted."""
    config = _make_config(verbose=True)
    assert config.verbose is True


def test_verbose_in_to_dict() -> None:
    """verbose should be serialized in to_dict()."""
    config_false = _make_config(verbose=False)
    config_true = _make_config(verbose=True)

    assert config_false.to_dict()["verbose"] is False
    assert config_true.to_dict()["verbose"] is True
