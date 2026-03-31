"""Tests for verbose config field in ExperimentConfig."""

from __future__ import annotations

import numpy as np

from objective import FixedRegressionObjective, LinearPolicy
from experiments.config import ExperimentConfig
from experiments.defaults import default_theta0


def _make_config(verbose: bool = False) -> ExperimentConfig:
    """Create a minimal ExperimentConfig with specified verbose setting."""
    policy = LinearPolicy()
    return ExperimentConfig(
        state_dim=2,
        objective=FixedRegressionObjective.from_parameters(
            policy=policy,
            beta_1=np.array([0.1, 0.2]),
            beta_2=-0.5,
            beta_3=np.array([0.05, 0.1]),
            beta_4=0.3,
        ),
        theta0=default_theta0(2),
        n_samples=5,
        step_rule="constant",
        perturbation_space="theta",
        verbose=verbose,
    )


def test_verbose_default_false() -> None:
    """Default verbose should be False."""
    config = ExperimentConfig(
        state_dim=2,
        objective=FixedRegressionObjective.from_parameters(
            policy=LinearPolicy(),
            beta_1=np.array([0.1, 0.2]),
            beta_2=-0.5,
            beta_3=np.array([0.05, 0.1]),
            beta_4=0.3,
        ),
        theta0=default_theta0(2),
        n_samples=5,
        step_rule="constant",
        perturbation_space="theta",
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
