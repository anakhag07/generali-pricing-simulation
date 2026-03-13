"""Experiment configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

import numpy as np

from objective.base import ActionObjective, ThetaObjective
from objective.composed import PolicyObjective
from objective.fixed_objective import FixedRegressionObjective
from objective.planted_logistic import PlantedLogisticObjective
from objective.policy import ConstantPolicy, LinearPolicy, Policy, PolicySpec, SoftmaxPolicy
from optimization.steps import STEP_RULES


@dataclass(frozen=True)
class CorrectnessSpec:
    """Configuration for correctness proxies and true-gradient references."""

    gradient_source: Literal["exact", "numdiff", "none"] = "exact"
    numdiff_method: Literal["central", "forward", "backward"] = "central"
    numdiff_step: float = 1e-4
    numdiff_aggregate: Literal["per-sample", "batch"] = "batch"
    numdiff_bounds: Optional[tuple[float, float]] = None

    def __post_init__(self) -> None:
        if self.gradient_source not in {"exact", "numdiff", "none"}:
            raise ValueError("gradient_source must be 'exact', 'numdiff', or 'none'.")
        if self.numdiff_method not in {"central", "forward", "backward"}:
            raise ValueError("numdiff_method must be 'central', 'forward', or 'backward'.")
        if self.numdiff_step <= 0.0:
            raise ValueError("numdiff_step must be positive.")
        if self.numdiff_aggregate not in {"per-sample", "batch"}:
            raise ValueError("numdiff_aggregate must be 'per-sample' or 'batch'.")
        if self.numdiff_bounds is not None:
            lower, upper = self.numdiff_bounds
            lower = float(lower)
            upper = float(upper)
            if lower >= upper:
                raise ValueError("numdiff_bounds must be an increasing (lower, upper) tuple.")
            object.__setattr__(self, "numdiff_bounds", (lower, upper))


@dataclass(frozen=True)
class ExperimentConfig:
    state_dim: int
    n_samples: int
    step_rule: str
    objective: ThetaObjective | None = None
    theta0: np.ndarray | None = None
    objective_model: ActionObjective | None = None
    policy_spec: PolicySpec | Policy | None = None
    batch_size: int | None = None
    seed: int = 7
    t_steps: int = 100
    step_size: float = 0.01
    grad_norm_tol: Optional[float] = None
    ftol: Optional[float] = None
    sigma: float = 0.1
    n_grad_samples: int = 64
    verbose: bool = False
    plot: bool = True
    plot_dir: str = "plots"
    enabled_estimators: tuple[str, ...] = ("first_order", "gauss_stein")
    wandb_enabled: bool = False
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str = "experiment"
    wandb_tags: tuple[str, ...] = ()
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_log_plots: bool = True
    wandb_estimator_allowlist: tuple[str, ...] | None = None
    correctness: CorrectnessSpec = field(default_factory=CorrectnessSpec)

    def __post_init__(self) -> None:
        enabled_estimators = tuple(self.enabled_estimators)
        object.__setattr__(self, "enabled_estimators", enabled_estimators)
        if not enabled_estimators:
            raise ValueError("enabled_estimators must include at least one estimator.")
        if len(set(enabled_estimators)) != len(enabled_estimators):
            raise ValueError("enabled_estimators must not contain duplicates.")
        allowed_estimators = {"first_order", "gauss_stein", "spsa"}
        unknown = [name for name in enabled_estimators if name not in allowed_estimators]
        if unknown:
            allowed = ", ".join(sorted(allowed_estimators))
            unknown_list = ", ".join(unknown)
            raise ValueError(f"Unknown estimators: {unknown_list}. Allowed: {allowed}.")

        wandb_tags = tuple(self.wandb_tags)
        object.__setattr__(self, "wandb_tags", wandb_tags)
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be 'online', 'offline', or 'disabled'.")
        if self.wandb_enabled and self.wandb_mode == "disabled":
            raise ValueError("wandb_mode='disabled' is incompatible with wandb_enabled=True.")
        if self.wandb_estimator_allowlist is not None:
            wandb_allowlist = tuple(self.wandb_estimator_allowlist)
            object.__setattr__(self, "wandb_estimator_allowlist", wandb_allowlist)
            if len(set(wandb_allowlist)) != len(wandb_allowlist):
                raise ValueError("wandb_estimator_allowlist must not contain duplicates.")
            unknown_wandb = [name for name in wandb_allowlist if name not in allowed_estimators]
            if unknown_wandb:
                allowed = ", ".join(sorted(allowed_estimators))
                unknown_list = ", ".join(unknown_wandb)
                raise ValueError(
                    f"Unknown wandb estimators: {unknown_list}. Allowed: {allowed}."
                )

        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        objective = self.objective
        theta0_raw = self.theta0
        if objective is None or theta0_raw is None:
            if self.objective_model is None or self.policy_spec is None:
                raise ValueError(
                    "Provide objective and theta0, or objective_model and policy_spec."
                )
            if isinstance(self.policy_spec, PolicySpec):
                policy = self.policy_spec.as_policy()
                theta0_raw = self.policy_spec.theta
            else:
                policy = self.policy_spec
                if theta0_raw is None:
                    raise ValueError("theta0 must be provided when policy_spec is a Policy instance.")
            objective = PolicyObjective(action_objective=self.objective_model, policy=policy)
            object.__setattr__(self, "objective", objective)

        theta0 = np.asarray(theta0_raw, dtype=float)
        if theta0.ndim != 1 or theta0.size < 1:
            raise ValueError("theta0 must be a 1D array with at least one element.")
        object.__setattr__(self, "theta0", theta0)

        if self.batch_size is not None:
            if self.batch_size <= 0:
                raise ValueError("batch_size must be positive when provided.")
            if self.batch_size > self.n_samples:
                raise ValueError("batch_size must be <= n_samples when provided.")

        if self.step_rule not in STEP_RULES:
            allowed = ", ".join(sorted(STEP_RULES))
            raise ValueError(f"step_rule must be one of {allowed}.")
        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive.")
        if self.grad_norm_tol is not None and self.grad_norm_tol <= 0.0:
            raise ValueError("grad_norm_tol must be positive when provided.")
        if self.ftol is not None and self.ftol <= 0.0:
            raise ValueError("ftol must be positive when provided.")
        if self.n_grad_samples <= 0:
            raise ValueError("n_grad_samples must be positive.")

        value_fn = getattr(objective, "value", None)
        grad_fn = getattr(objective, "grad", None)
        if value_fn is None or not callable(value_fn):
            raise ValueError("objective must implement value(theta, x_batch).")
        if grad_fn is None or not callable(grad_fn):
            raise ValueError("objective must implement grad(theta, x_batch).")

        policy = getattr(objective, "policy", None)
        if policy is not None and hasattr(policy, "required_theta_dim"):
            required = int(policy.required_theta_dim(self.state_dim))
            if self.theta0.size < required:
                raise ValueError(
                    f"theta0 must have at least {required} elements for policy {type(policy).__name__}."
                )

    def to_dict(self) -> dict[str, Any]:
        objective = self.objective
        if objective is None:
            raise ValueError("objective must be initialized before serialization.")
        theta0 = self.theta0
        if theta0 is None:
            raise ValueError("theta0 must be initialized before serialization.")
        return {
            "state_dim": int(self.state_dim),
            "n_samples": int(self.n_samples),
            "batch_size": int(self.batch_size) if self.batch_size is not None else None,
            "step_rule": self.step_rule,
            "seed": int(self.seed),
            "t_steps": int(self.t_steps),
            "step_size": float(self.step_size),
            "grad_norm_tol": float(self.grad_norm_tol)
            if self.grad_norm_tol is not None
            else None,
            "ftol": float(self.ftol) if self.ftol is not None else None,
            "sigma": float(self.sigma),
            "n_grad_samples": int(self.n_grad_samples),
            "verbose": bool(self.verbose),
            "plot": bool(self.plot),
            "plot_dir": self.plot_dir,
            "enabled_estimators": list(self.enabled_estimators),
            "theta0": _as_list(theta0),
            "objective": _theta_objective_to_dict(objective),
            "wandb": {
                "enabled": bool(self.wandb_enabled),
                "project": self.wandb_project,
                "entity": self.wandb_entity,
                "group": self.wandb_group,
                "job_type": self.wandb_job_type,
                "tags": list(self.wandb_tags),
                "mode": self.wandb_mode,
                "log_plots": bool(self.wandb_log_plots),
                "estimator_allowlist": list(self.wandb_estimator_allowlist)
                if self.wandb_estimator_allowlist is not None
                else None,
            },
            "correctness": _correctness_to_dict(self.correctness),
        }


def _action_objective_to_dict(action_objective: object) -> dict[str, Any]:
    if isinstance(action_objective, FixedRegressionObjective):
        return {
            "type": "FixedRegressionObjective",
            "beta_1": _as_list(action_objective.acceptance.beta_1),
            "beta_2": float(action_objective.acceptance.beta_2),
            "beta_3": _as_list(action_objective.loss.beta_3),
            "beta_4": float(action_objective.revenue.beta_4),
        }
    if isinstance(action_objective, PlantedLogisticObjective):
        return {
            "type": "PlantedLogisticObjective",
            "alpha": float(action_objective.alpha),
            "beta": _as_list(action_objective.beta),
            "bias": float(action_objective.bias),
            "u_star": float(action_objective.u_star),
        }
    return {"type": type(action_objective).__name__}


def _policy_to_dict(policy: object) -> dict[str, Any]:
    if isinstance(policy, ConstantPolicy):
        return {"type": "ConstantPolicy"}
    if isinstance(policy, LinearPolicy):
        return {"type": "LinearPolicy"}
    if isinstance(policy, SoftmaxPolicy):
        return {"type": "SoftmaxPolicy"}
    return {"type": type(policy).__name__}


def _theta_objective_to_dict(objective: ThetaObjective) -> dict[str, Any]:
    if isinstance(objective, PolicyObjective):
        return {
            "type": "PolicyObjective",
            "action_objective": _action_objective_to_dict(objective.action_objective),
            "policy": _policy_to_dict(objective.policy),
        }
    return {"type": type(objective).__name__}


def _correctness_to_dict(correctness: CorrectnessSpec) -> dict[str, Any]:
    return {
        "gradient_source": correctness.gradient_source,
        "numdiff_method": correctness.numdiff_method,
        "numdiff_step": float(correctness.numdiff_step),
        "numdiff_aggregate": correctness.numdiff_aggregate,
        "numdiff_bounds": list(correctness.numdiff_bounds)
        if correctness.numdiff_bounds is not None
        else None,
    }


def _as_list(values: object) -> list[float]:
    arr = np.asarray(values, dtype=float)
    return [float(val) for val in arr.tolist()]


def make_fixed_regression_objective(
    *,
    beta_1: np.ndarray,
    beta_2: float,
    beta_3: np.ndarray,
    beta_4: float,
) -> FixedRegressionObjective:
    return FixedRegressionObjective.from_parameters(
        beta_1=np.asarray(beta_1, dtype=float),
        beta_2=float(beta_2),
        beta_3=np.asarray(beta_3, dtype=float),
        beta_4=float(beta_4),
    )


def make_planted_logistic_objective(
    *,
    alpha: float,
    beta: np.ndarray,
    bias: float,
    u_star: float,
) -> PlantedLogisticObjective:
    return PlantedLogisticObjective(
        alpha=float(alpha),
        beta=np.asarray(beta, dtype=float),
        bias=float(bias),
        u_star=float(u_star),
    )


def make_softmax_policy() -> SoftmaxPolicy:
    return SoftmaxPolicy()


def make_policy_objective(*, action_objective: ActionObjective, policy: Policy) -> PolicyObjective:
    return PolicyObjective(action_objective=action_objective, policy=policy)


def canonical_training_block(
    *,
    n_samples: int,
    step_rule: str,
    t_steps: int,
    step_size: float,
    sigma: float,
    n_grad_samples: int,
    enabled_estimators: tuple[str, ...],
    batch_size: int | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
) -> dict[str, Any]:
    return {
        "n_samples": int(n_samples),
        "step_rule": step_rule,
        "t_steps": int(t_steps),
        "step_size": float(step_size),
        "sigma": float(sigma),
        "n_grad_samples": int(n_grad_samples),
        "enabled_estimators": tuple(enabled_estimators),
        "batch_size": int(batch_size) if batch_size is not None else None,
        "grad_norm_tol": float(grad_norm_tol) if grad_norm_tol is not None else None,
        "ftol": float(ftol) if ftol is not None else None,
    }


def canonical_runtime_block(
    *,
    plot: bool,
    verbose: bool,
    wandb_enabled: bool,
    plot_dir: str = "plots",
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_job_type: str = "experiment",
    wandb_tags: tuple[str, ...] = (),
    wandb_mode: Literal["online", "offline", "disabled"] = "online",
    wandb_log_plots: bool = True,
    wandb_estimator_allowlist: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "plot": bool(plot),
        "plot_dir": plot_dir,
        "verbose": bool(verbose),
        "wandb_enabled": bool(wandb_enabled),
        "wandb_project": wandb_project,
        "wandb_entity": wandb_entity,
        "wandb_group": wandb_group,
        "wandb_job_type": wandb_job_type,
        "wandb_tags": tuple(wandb_tags),
        "wandb_mode": wandb_mode,
        "wandb_log_plots": bool(wandb_log_plots),
        "wandb_estimator_allowlist": tuple(wandb_estimator_allowlist)
        if wandb_estimator_allowlist is not None
        else None,
    }


def build_experiment_config(
    *,
    seed: int,
    state_dim: int,
    objective: ThetaObjective,
    theta0: np.ndarray,
    training: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
    correctness: CorrectnessSpec | None = None,
) -> ExperimentConfig:
    payload: dict[str, Any] = {
        "seed": int(seed),
        "state_dim": int(state_dim),
        "objective": objective,
        "theta0": np.asarray(theta0, dtype=float),
    }
    payload.update(dict(training))
    if runtime is not None:
        payload.update(dict(runtime))
    if correctness is not None:
        payload["correctness"] = correctness
    return ExperimentConfig(**payload)
