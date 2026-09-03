from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from experiments.config import CorrectnessSpec, ExperimentConfig
from experiments.configs import get_config
from objective.modifications import (
    HomoskedasticGaussianNoise,
    NoiseModification,
    NoisyObjective,
    ProximalThetaRegularizer,
    RegularizationModification,
    RegularizedObjective,
)
from objective import FixedRegressionObjective, QuadraticFeatureMap, SoftmaxPolicy
from objective.policy import Policy, policy_theta_dim


class _SeparatePolicyColumnsObjective:
    """Minimal objective whose policy consumes columns appended to raw state."""

    policy = SoftmaxPolicy()
    acceptance_state_cols = ("raw_x",)
    policy_feature_cols = ("policy_x",)
    acceptance_model = None

    def value(self, theta, x_batch) -> float:
        return float(np.mean(self.policy_value(theta, x_batch)))

    def grad(self, theta, x_batch) -> np.ndarray:
        return np.mean(self.policy_grad(theta, x_batch), axis=0)

    def policy_value(self, theta, x_batch) -> np.ndarray:
        return self.policy.value(theta, x_batch.loc[:, ["policy_x"]].to_numpy(dtype=float))

    def policy_grad(self, theta, x_batch) -> np.ndarray:
        return self.policy.grad(theta, x_batch.loc[:, ["policy_x"]].to_numpy(dtype=float))


def test_policy_probe_preserves_appended_policy_feature_columns() -> None:
    x_fixed = pd.DataFrame({"raw_x": [1.0, 2.0], "policy_x": [0.25, -0.5]})

    config = ExperimentConfig(
        state_dim=1,
        objective=_SeparatePolicyColumnsObjective(),
        theta0=np.zeros(2),
        n_samples=2,
        step_rule="l-bfgs-b",
        perturbation_space="u",
        enabled_estimators=("first_order",),
        x_fixed=x_fixed,
    )

    assert list(config.x_fixed.columns) == ["raw_x", "policy_x"]


def _theta0(state_dim: int = 1, policy: Policy | None = None) -> np.ndarray:
    dim = policy_theta_dim(policy, state_dim) if policy is not None else state_dim + 1
    return np.zeros(dim, dtype=float)


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
        policy=SoftmaxPolicy(),
        beta_1=beta_1,
        beta_2=-0.5,
        beta_3=beta_3,
        beta_4=0.4,
    )
    config = ExperimentConfig(
        state_dim=state_dim,
        objective=objective,
        theta0=_theta0(state_dim),
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
        theta0=_theta0(1),
        n_samples=5,
        step_rule="constant",
        perturbation_space="theta",
    )
    assert config_default.verbose is False

    config_verbose = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=_theta0(1),
        n_samples=5,
        step_rule="constant",
        perturbation_space="theta",
        verbose=True,
    )
    assert config_verbose.verbose is True


def test_train_test_fraction_defaults_and_serialization() -> None:
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
        theta0=_theta0(1),
        n_samples=5,
        step_rule="constant",
        perturbation_space="theta",
        train_fraction=0.8,
        test_fraction=0.2,
    )

    assert config.train_fraction == pytest.approx(0.8)
    assert config.test_fraction == pytest.approx(0.2)
    payload = config.to_dict()
    assert payload["train_fraction"] == pytest.approx(0.8)
    assert payload["test_fraction"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("train_fraction", "test_fraction", "message"),
    [
        (-0.1, 1.1, "train_fraction must be in"),
        (1.1, -0.1, "train_fraction must be in"),
        (0.0, 1.0, "train_fraction must be positive"),
        (0.8, 0.1, "sum to 1.0"),
        (np.inf, 0.0, "must be finite"),
    ],
)
def test_train_test_fraction_validation(
    train_fraction: float,
    test_fraction: float,
    message: str,
) -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )

    with pytest.raises(ValueError, match=message):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=_theta0(1),
            n_samples=5,
            step_rule="constant",
            perturbation_space="theta",
            train_fraction=train_fraction,
            test_fraction=test_fraction,
        )


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
        theta0=_theta0(1),
        n_samples=5,
        step_rule="constant",
        perturbation_space="theta",
        constant_u_baselines=(-0.3, 0.0, 0.2),
    )

    assert config.constant_u_baselines == (-0.3, 0.0, 0.2)
    assert config.to_dict()["constant_u_baselines"] == [-0.3, 0.0, 0.2]


def test_constant_enabled_estimator_is_accepted() -> None:
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
        theta0=_theta0(1),
        n_samples=5,
        step_rule="constant",
        perturbation_space="theta",
        enabled_estimators=("constant",),
        wandb_estimator_allowlist=("constant",),
    )

    assert config.enabled_estimators == ("constant",)
    assert config.wandb_estimator_allowlist == ("constant",)


def test_config_accepts_quadratic_policy_theta_dim() -> None:
    state_dim = 2
    policy = SoftmaxPolicy(feature_map=QuadraticFeatureMap())
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1, 0.2],
        beta_2=-0.5,
        beta_3=[0.2, 0.3],
        beta_4=0.4,
    )
    config = ExperimentConfig(
        state_dim=state_dim,
        objective=objective,
        theta0=_theta0(state_dim, policy),
        n_samples=5,
        step_rule="constant",
        perturbation_space="theta",
    )

    assert config.theta0.shape == (policy.theta_dim(state_dim),)


def test_softmax_policy_action_bounds_are_serialized() -> None:
    policy = SoftmaxPolicy(action_low=-0.1, action_high=0.2)
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
        theta0=_theta0(1, policy),
        n_samples=5,
        step_rule="constant",
        perturbation_space="theta",
    )

    payload = config.to_dict()["objective"]["policy"]
    assert payload["action_low"] == -0.1
    assert payload["action_high"] == 0.2


def test_config_rejects_old_theta_dim_for_quadratic_policy() -> None:
    policy = SoftmaxPolicy(feature_map=QuadraticFeatureMap())
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1, 0.2],
        beta_2=-0.5,
        beta_3=[0.2, 0.3],
        beta_4=0.4,
    )

    with pytest.raises(ValueError, match="policy requires 6"):
        ExperimentConfig(
            state_dim=2,
            objective=objective,
            theta0=np.zeros(3, dtype=float),
            n_samples=5,
            step_rule="constant",
            perturbation_space="theta",
        )


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
            theta0=_theta0(1),
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
            theta0=_theta0(1),
            n_samples=5,
            step_rule="constant",
            perturbation_space="theta",
            acceptance_floor=0.2,
            acceptance_penalty_weight=10.0,
        )


def test_acceptance_floor_requires_positive_penalty_weight() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_base", overrides={"policy_kind": "linear", "n_samples": 25, "plot": False, "wandb_enabled": False})
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
            theta0=_theta0(1),
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
            theta0=_theta0(1),
            n_samples=5,
            step_rule="constant",
            perturbation_space="theta",
            acceptance_floor=0.5,
            lagrangian_lambda=2.0,
        )


def test_lagrangian_lambda_rejects_penalty_weight() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_base", overrides={"policy_kind": "linear", "n_samples": 25, "plot": False, "wandb_enabled": False})
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

    cfg = get_config("real_data_glm_base", overrides={"policy_kind": "linear", "n_samples": 25, "plot": False, "wandb_enabled": False})
    with pytest.raises(ValueError, match="only supported for unconstrained step rules"):
        ExperimentConfig(
            **{
                **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                "step_rule": "trust-constr",
                "acceptance_floor": 0.5,
                "lagrangian_lambda": 2.0,
            }
        )


def test_trust_constr_supports_no_acceptance_floor() -> None:
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
        theta0=_theta0(1),
        n_samples=5,
        step_rule="trust-constr",
        perturbation_space="theta",
    )

    assert config.acceptance_floor is None


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
            theta0=_theta0(1),
            n_samples=5,
            step_rule="trust-constr",
            perturbation_space="theta",
            acceptance_floor=0.2,
        )


def test_trust_constr_rejects_penalty_weight() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_base", overrides={"policy_kind": "linear", "n_samples": 25, "plot": False, "wandb_enabled": False})
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

    cfg = get_config("real_data_glm_base", overrides={"policy_kind": "linear", "n_samples": 25, "plot": False, "wandb_enabled": False})
    with pytest.raises(ValueError, match="batch_size=None"):
        ExperimentConfig(
            **{
                **{k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
                "step_rule": "trust-constr",
                "acceptance_floor": 0.5,
                "batch_size": 2,
            }
        )


def test_trust_constr_rejects_ftol() -> None:
    from experiments.configs import get_config

    cfg = get_config("real_data_glm_base", overrides={"policy_kind": "linear", "n_samples": 25, "plot": False, "wandb_enabled": False})
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

    cfg = get_config("real_data_glm_base", overrides={"policy_kind": "linear", "n_samples": 25, "plot": False, "wandb_enabled": False})
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

    cfg = get_config("real_data_glm_base", overrides={"policy_kind": "linear", "n_samples": 25, "plot": False, "wandb_enabled": False})
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
            theta0=_theta0(1),
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
    theta0 = _theta0(1)
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
        theta0=_theta0(1),
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
        theta0=_theta0(1),
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
        theta0=_theta0(1),
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
        theta0=_theta0(1),
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
        theta0=_theta0(1),
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
    theta0 = _theta0(1)

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

    glm_cfg = get_config("real_data_glm_base", overrides={"policy_kind": "linear", "n_samples": 25, "plot": False, "wandb_enabled": False})
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


def test_compute_backend_validation() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )

    with pytest.raises(ValueError, match="compute_backend"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=_theta0(1),
            n_samples=5,
            step_rule="constant",
            perturbation_space="theta",
            compute_backend="bogus",
        )

    with pytest.raises(ValueError, match="trust-constr"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=_theta0(1),
            n_samples=5,
            step_rule="constant",
            perturbation_space="theta",
            compute_backend="jax",
        )

    with pytest.raises(ValueError, match="batch_size=None"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=_theta0(1),
            n_samples=5,
            step_rule="trust-constr",
            perturbation_space="theta",
            compute_backend="jax",
            batch_size=2,
        )


def test_optax_step_rule_validation() -> None:
    policy = SoftmaxPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )

    adam_config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=_theta0(1),
        n_samples=5,
        step_rule="optax-adam",
        perturbation_space="theta",
    )
    assert adam_config.step_rule == "optax-adam"

    jax_config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=_theta0(1),
        n_samples=5,
        step_rule="optax-adam",
        perturbation_space="theta",
        compute_backend="jax",
    )
    assert jax_config.compute_backend == "jax"

    with pytest.raises(ValueError, match="ftol is not supported by optax step rules"):
        ExperimentConfig(
            state_dim=1,
            objective=objective,
            theta0=_theta0(1),
            n_samples=5,
            step_rule="optax-sgd",
            perturbation_space="theta",
            ftol=1e-6,
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
            theta0=_theta0(1),
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
    theta0 = _theta0(1)

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
    theta0 = _theta0(1)

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
        theta0=_theta0(1),
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
            theta0=_theta0(1),
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
        theta0=_theta0(1),
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


def test_objective_modifications_are_composed_serialized_and_not_reapplied() -> None:
    config = get_config(
        "synthetic_quadratic_base",
        overrides={
            "objective_modifications": (
                RegularizationModification(
                    regularizers=(ProximalThetaRegularizer(weight=0.2),)
                ),
                NoiseModification(noise=HomoskedasticGaussianNoise(std=0.1, seed=5)),
            ),
            "enabled_estimators": ("finite_difference",),
            "correctness": CorrectnessSpec(gradient_source="none"),
            "plot": False,
            "verbose": False,
        },
    )

    assert isinstance(config.objective, NoisyObjective)
    assert isinstance(config.objective.base_objective, RegularizedObjective)
    payload = config.to_dict()
    assert [item["type"] for item in payload["objective_modifications"]] == [
        "RegularizationModification",
        "NoiseModification",
    ]
    assert payload["objective"]["type"] == "NoisyObjective"
    assert payload["objective"]["base_objective"]["type"] == "RegularizedObjective"

    cloned = replace(config, seed=config.seed + 1)
    assert isinstance(cloned.objective, NoisyObjective)
    assert isinstance(cloned.objective.base_objective, RegularizedObjective)
    assert not isinstance(cloned.objective.base_objective.base_objective, RegularizedObjective)
