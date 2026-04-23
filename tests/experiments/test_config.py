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
        perturbation_space="theta",
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
        perturbation_space="theta",
    )
    assert config_default.verbose is False

    config_verbose = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=default_theta0(1),
        n_samples=5,
        step_rule="constant",
        perturbation_space="theta",
        verbose=True,
    )
    assert config_verbose.verbose is True


def test_constant_u_baselines_are_serialized() -> None:
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
        perturbation_space="theta",
        constant_u_baselines=(-0.3, 0.0, 0.2),
    )

    assert config.constant_u_baselines == (-0.3, 0.0, 0.2)
    assert config.to_dict()["constant_u_baselines"] == [-0.3, 0.0, 0.2]


def test_constant_u_baselines_reject_duplicates() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )

    with pytest.raises(ValueError, match="must not contain duplicates"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=default_theta0(1),
            n_samples=5,
            step_rule="constant",
            perturbation_space="theta",
            constant_u_baselines=(0.0, 0.0),
        )


def test_acceptance_floor_requires_supporting_objective() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    with pytest.raises(ValueError, match="mean_acceptance"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=default_theta0(1),
            n_samples=5,
            step_rule="constant",
            perturbation_space="theta",
            acceptance_floor=0.2,
            acceptance_penalty_weight=10.0,
        )


def test_acceptance_floor_requires_positive_penalty_weight() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_policy_base")
    with pytest.raises(ValueError, match="acceptance_penalty_weight"):
        ExperimentConfig(
            **{
                **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                "acceptance_floor": 0.5,
                "acceptance_penalty_weight": 0.0,
            }
        )


def test_lagrangian_lambda_requires_acceptance_floor() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    with pytest.raises(ValueError, match="lagrangian_lambda requires acceptance_floor"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=default_theta0(1),
            n_samples=5,
            step_rule="constant",
            perturbation_space="theta",
            lagrangian_lambda=0.0,
        )


def test_lagrangian_lambda_requires_mean_acceptance_grad() -> None:
    class ObjectiveWithAcceptanceNoGrad:
        def value(self, theta, x_batch):
            del theta, x_batch
            return 0.0

        def grad(self, theta, x_batch):
            del theta, x_batch
            return np.zeros(2, dtype=float)

        def mean_acceptance(self, theta, x_batch):
            del theta, x_batch
            return 0.5

    objective = ObjectiveWithAcceptanceNoGrad()
    with pytest.raises(ValueError, match="lagrangian_lambda requires an objective with mean_acceptance_grad"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=default_theta0(1),
            n_samples=5,
            step_rule="constant",
            perturbation_space="theta",
            acceptance_floor=0.5,
            lagrangian_lambda=2.0,
        )


def test_lagrangian_lambda_rejects_penalty_weight() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_policy_base")
    with pytest.raises(ValueError, match="mutually exclusive"):
        ExperimentConfig(
            **{
                **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                "acceptance_floor": 0.5,
                "acceptance_penalty_weight": 10.0,
                "lagrangian_lambda": 2.0,
            }
        )


def test_lagrangian_lambda_rejected_for_trust_constr() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_policy_base")
    with pytest.raises(ValueError, match="only supported for unconstrained step rules"):
        ExperimentConfig(
            **{
                **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                "step_rule": "trust-constr",
                "acceptance_floor": 0.5,
                "lagrangian_lambda": 2.0,
            }
        )


def test_trust_constr_requires_acceptance_floor() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )

    with pytest.raises(ValueError, match="requires acceptance_floor"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=default_theta0(1),
            n_samples=5,
            step_rule="trust-constr",
            perturbation_space="theta",
        )


def test_trust_constr_requires_mean_acceptance_grad() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )

    with pytest.raises(ValueError, match="mean_acceptance"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=default_theta0(1),
            n_samples=5,
            step_rule="trust-constr",
            perturbation_space="theta",
            acceptance_floor=0.2,
        )


def test_trust_constr_rejects_penalty_weight() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_policy_base")
    with pytest.raises(ValueError, match="acceptance_penalty_weight"):
        ExperimentConfig(
            **{
                **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                "step_rule": "trust-constr",
                "acceptance_floor": 0.5,
                "acceptance_penalty_weight": 10.0,
            }
        )


def test_trust_constr_requires_full_batch() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_policy_base")
    with pytest.raises(ValueError, match="batch_size=None"):
        ExperimentConfig(
            **{
                **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                "step_rule": "trust-constr",
                "acceptance_floor": 0.5,
                "batch_size": 32,
            }
        )


def test_trust_constr_rejects_ftol() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_policy_base")
    with pytest.raises(ValueError, match="ftol"):
        ExperimentConfig(
            **{
                **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                "step_rule": "trust-constr",
                "acceptance_floor": 0.5,
                "ftol": 1e-8,
            }
        )


def test_trust_constr_accepts_initial_constr_penalty() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_policy_base")
    trust_config = ExperimentConfig(
        **{
            **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
            "step_rule": "trust-constr",
            "acceptance_floor": 0.5,
            "initial_constr_penalty": 2.5,
        }
    )

    assert trust_config.initial_constr_penalty == 2.5
    assert trust_config.to_dict()["initial_constr_penalty"] == 2.5


def test_initial_constr_penalty_must_be_positive() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_linear_policy_base")
    with pytest.raises(ValueError, match="initial_constr_penalty must be positive"):
        ExperimentConfig(
            **{
                **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                "step_rule": "trust-constr",
                "acceptance_floor": 0.5,
                "initial_constr_penalty": 0.0,
            }
        )


def test_initial_constr_penalty_rejected_outside_trust_constr() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )

    with pytest.raises(ValueError, match="only used by step_rule='trust-constr'"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=default_theta0(1),
            n_samples=5,
            step_rule="constant",
            perturbation_space="theta",
            initial_constr_penalty=1.0,
        )


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
        perturbation_space="theta",
            enabled_estimators=("not-a-method",),
        )

    with pytest.raises(ValueError, match="Unknown estimators"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=theta0,
            n_samples=5,
            step_rule="constant",
        perturbation_space="theta",
            enabled_estimators=("lbfgs",),
        )

    with pytest.raises(ValueError, match="enabled_estimators must include at least one"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=theta0,
            n_samples=5,
            step_rule="constant",
        perturbation_space="theta",
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
        perturbation_space="theta",
        enabled_estimators=("spsa",),
    )
    assert config.enabled_estimators == ("spsa",)


def test_enabled_estimators_accepts_finite_difference() -> None:
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
        perturbation_space="theta",
        enabled_estimators=("finite_difference",),
    )
    assert config.enabled_estimators == ("finite_difference",)


def test_enabled_estimators_accepts_finite_difference_alias() -> None:
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
        perturbation_space="theta",
        enabled_estimators=("finite-difference",),
    )
    assert config.enabled_estimators == ("finite_difference",)


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
        perturbation_space="theta",
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
        perturbation_space="theta",
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
        perturbation_space="theta",
        )

    config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=theta0,
        n_samples=5,
        step_rule="l-bfgs-b",
        perturbation_space="theta",
    )
    assert config.step_rule == "l-bfgs-b"

    from experiments.configs import get_config

    glm_cfg = get_config("real_data_glm_linear_policy_base")
    trust_config = ExperimentConfig(
        **{
            **{k: getattr(glm_cfg, k) for k in glm_cfg.__dataclass_fields__},
            "step_rule": "trust-constr",
            "acceptance_floor": 0.5,
        }
    )
    assert trust_config.step_rule == "trust-constr"

    with pytest.raises(ValueError, match="step_size must be positive"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=theta0,
            n_samples=5,
            step_rule="constant",
        perturbation_space="theta",
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
        perturbation_space="theta",
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
        perturbation_space="theta",
            ftol=0.0,
        )

    config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=theta0,
        n_samples=5,
        step_rule="constant",
        perturbation_space="theta",
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
        perturbation_space="theta",
        )

    with pytest.raises(ValueError, match="batch_size must be <= n_samples"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=theta0,
            n_samples=5,
            batch_size=6,
            step_rule="constant",
        perturbation_space="theta",
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
        perturbation_space="theta",
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
        perturbation_space="theta",
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
        perturbation_space="theta",
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
