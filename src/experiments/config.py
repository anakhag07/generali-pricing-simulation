"""Experiment configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

import numpy as np

from objective.base import ObjectiveModel
from objective.fixed_objective import FixedRegressionObjective
from objective.policy import POLICY_LINEAR, POLICY_SOFTMAX, PolicySpec
from objective.planted_logistic import PlantedLogisticObjective
from optimization.steps import STEP_RULES


@dataclass(frozen=True)
class CorrectnessSpec:
    """Configuration for correctness proxies and true-gradient references."""

    gradient_source: Literal["exact", "numdiff", "none"] = "exact"
    numdiff_method: Literal["central", "forward", "backward"] = "central"
    numdiff_step: float = 1e-4
    numdiff_aggregate: Literal["per-sample", "batch"] = "per-sample"
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
    objective_model: ObjectiveModel
    policy_spec: PolicySpec
    n_samples: int
    step_rule: str
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

        if isinstance(self.objective_model, FixedRegressionObjective):
            if self.objective_model.acceptance.beta_1.size < self.state_dim:
                raise ValueError("beta_1 must have at least state_dim elements.")
            if self.objective_model.loss.beta_3.size < self.state_dim:
                raise ValueError("beta_3 must have at least state_dim elements.")

        if self.policy_spec.kind in (POLICY_LINEAR, POLICY_SOFTMAX):
            required = self.state_dim + 1
            if self.policy_spec.theta.size < required:
                raise ValueError(
                    "Policy theta must have at least state_dim + 1 elements for linear/softmax policies."
                )
        if self.correctness is None:
            object.__setattr__(
                self,
                "correctness",
                CorrectnessSpec(gradient_source="none"),
            )
        if self.correctness.gradient_source == "exact":
            grad_u = getattr(self.objective_model, "grad_u", None)
            if grad_u is None or not callable(grad_u):
                raise ValueError(
                    "objective_model must implement grad_u for gradient_source='exact'."
                )
        if self.correctness.gradient_source == "numdiff":
            if self.correctness.numdiff_aggregate != "per-sample":
                raise ValueError(
                    "numdiff_aggregate='batch' is not supported for theta-gradient correctness."
                )

    def to_dict(self) -> dict[str, Any]:
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
            "policy_spec": {
                "kind": self.policy_spec.kind,
                "theta": _as_list(self.policy_spec.theta),
            },
            "objective_model": _objective_to_dict(self.objective_model),
        }


def _objective_to_dict(objective_model: ObjectiveModel) -> dict[str, Any]:
    if isinstance(objective_model, FixedRegressionObjective):
        return {
            "type": "FixedRegressionObjective",
            "beta_1": _as_list(objective_model.acceptance.beta_1),
            "beta_2": float(objective_model.acceptance.beta_2),
            "beta_3": _as_list(objective_model.loss.beta_3),
            "beta_4": float(objective_model.revenue.beta_4),
        }
    if isinstance(objective_model, PlantedLogisticObjective):
        return {
            "type": "PlantedLogisticObjective",
            "alpha": float(objective_model.alpha),
            "beta": _as_list(objective_model.beta),
            "bias": float(objective_model.bias),
            "u_star": float(objective_model.u_star),
        }
    return {"type": type(objective_model).__name__}


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


def make_softmax_policy_spec(*, theta: np.ndarray) -> PolicySpec:
    return PolicySpec(theta=np.asarray(theta, dtype=float), kind=POLICY_SOFTMAX)


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
    objective_model: ObjectiveModel,
    policy_spec: PolicySpec,
    training: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
    correctness: CorrectnessSpec | None = None,
) -> ExperimentConfig:
    payload: dict[str, Any] = {
        "seed": int(seed),
        "state_dim": int(state_dim),
        "objective_model": objective_model,
        "policy_spec": policy_spec,
    }
    payload.update(dict(training))
    if runtime is not None:
        payload.update(dict(runtime))
    if correctness is not None:
        payload["correctness"] = correctness
    return ExperimentConfig(**payload)
