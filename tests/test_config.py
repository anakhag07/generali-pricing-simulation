from __future__ import annotations

import numpy as np
import pytest

from experiments.config import ExperimentConfig
from experiments.defaults import default_theta0, default_policy
from objective import FixedRegressionObjective, SoftmaxPolicy


def test_beta_2_must_be_negative() -> None:
    policy = SoftmaxPolicy()
    with pytest.raises(
        ValueError,
        match="beta_2 must be negative; acceptance probability should decrease as policy value increases.",
    ):
        FixedRegressionObjective.from_parameters(
            policy=policy,
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
            policy=policy,
            beta_1=[1.0],
            beta_2=0.5,
            beta_3=[1.0],
            beta_4=1.0,
        )


def test_state_dim_requires_matching_objective() -> None:
    state_dim = 5
    beta_1 = np.linspace(0.1, 0.5, num=state_dim, dtype=float)
    beta_3 = np.linspace(0.1, 0.5, num=state_dim, dtype=float)
    objective = FixedRegressionObjective.from_parameters(
        policy=default_policy(state_dim),
        beta_1=beta_1,
        beta_2=-0.5,
        beta_3=beta_3,
        beta_4=0.4,
    )
    config = ExperimentConfig(
        state_dim=state_dim,
        objective=objective,
        theta0=default_theta0(state_dim),
        n_samples=5,
        step_rule="constant",
    )
    assert config.state_dim == state_dim


def test_verbose_default_and_override() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    config_default = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=default_theta0(1),
        n_samples=5,
        step_rule="constant",
    )
    assert config_default.verbose is False

    config_verbose = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=default_theta0(1),
        n_samples=5,
        step_rule="constant",
        verbose=True,
    )
    assert config_verbose.verbose is True


def test_enabled_estimators_validation() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    theta0 = default_theta0(1)
    with pytest.raises(ValueError, match="Unknown estimators"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=theta0,
            n_samples=5,
            step_rule="constant",
            enabled_estimators=("not-a-method",),
        )

    with pytest.raises(ValueError, match="Unknown estimators"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=theta0,
            n_samples=5,
            step_rule="constant",
            enabled_estimators=("lbfgs",),
        )

    with pytest.raises(ValueError, match="enabled_estimators must include at least one"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=theta0,
            n_samples=5,
            step_rule="constant",
            enabled_estimators=(),
        )


def test_enabled_estimators_accepts_spsa() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=default_theta0(1),
        n_samples=5,
        step_rule="constant",
        enabled_estimators=("spsa",),
    )
    assert config.enabled_estimators == ("spsa",)


def test_enabled_estimators_accepts_stein_difference() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=default_theta0(1),
        n_samples=5,
        step_rule="constant",
        enabled_estimators=("stein_difference",),
    )
    assert config.enabled_estimators == ("stein_difference",)


def test_enabled_estimators_accepts_stein_difference_alias() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=default_theta0(1),
        n_samples=5,
        step_rule="constant",
        enabled_estimators=("stein-difference",),
    )
    assert config.enabled_estimators == ("stein_difference",)


def test_step_rule_validation() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    theta0 = default_theta0(1)

    with pytest.raises(ValueError, match="step_rule must be one of"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=theta0,
            n_samples=5,
            step_rule="unknown",
        )

    config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=theta0,
        n_samples=5,
        step_rule="l-bfgs-b",
    )
    assert config.step_rule == "l-bfgs-b"

    with pytest.raises(ValueError, match="step_size must be positive"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=theta0,
            n_samples=5,
            step_rule="constant",
            step_size=0.0,
        )


def test_grad_norm_tol_validation() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )

    with pytest.raises(ValueError, match="grad_norm_tol must be positive"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=default_theta0(1),
            n_samples=5,
            step_rule="constant",
            grad_norm_tol=0.0,
        )


def test_ftol_validation_and_serialization() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    theta0 = default_theta0(1)

    with pytest.raises(ValueError, match="ftol must be positive"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=theta0,
            n_samples=5,
            step_rule="constant",
            ftol=0.0,
        )

    config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=theta0,
        n_samples=5,
        step_rule="constant",
        ftol=1e-9,
    )
    payload = config.to_dict()
    assert payload["ftol"] == 1e-9


def test_batch_size_validation() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    theta0 = default_theta0(1)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=theta0,
            n_samples=5,
            batch_size=0,
            step_rule="constant",
        )

    with pytest.raises(ValueError, match="batch_size must be <= n_samples"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=theta0,
            n_samples=5,
            batch_size=6,
            step_rule="constant",
        )


def test_batch_size_serialization() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=default_theta0(1),
        n_samples=5,
        batch_size=2,
        step_rule="constant",
    )
    payload = config.to_dict()
    assert payload["batch_size"] == 2


def test_wandb_allowlist_validation() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    with pytest.raises(ValueError, match="Unknown wandb estimators"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=default_theta0(1),
            n_samples=5,
            step_rule="constant",
            wandb_estimator_allowlist=("bogus",),
        )


def test_wandb_config_serialization() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=default_theta0(1),
        n_samples=5,
        step_rule="constant",
        wandb_enabled=True,
        wandb_project="pricing-sim",
        wandb_tags=("smoke", "wandb"),
        wandb_estimator_allowlist=("spsa",),
    )
    payload = config.to_dict()
    assert payload["wandb"]["enabled"] is True
    assert payload["wandb"]["project"] == "pricing-sim"
    assert payload["wandb"]["tags"] == ["smoke", "wandb"]
    assert payload["wandb"]["estimator_allowlist"] == ["spsa"]
