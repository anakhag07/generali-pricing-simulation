"""Experiment configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

import numpy as np

from objective.base import Objective, Policy
from objective.objectives import (
    FixedRegressionObjective,
    ModelBasedObjective,
    PlantedLogisticObjective,
)
from objective.policy import ConstantPolicy, LinearPolicy, SoftmaxPolicy
from optimization.steps import STEP_RULES


@dataclass(frozen=True)
class CorrectnessSpec:
    """Controls how "true" gradients are computed: exact, numdiff, or none."""

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
    """Frozen configuration for a single experiment run with validation."""

    state_dim: int
    n_samples: int
    step_rule: str
    objective: Objective
    theta0: np.ndarray
    perturbation_space: Literal["theta", "u"]
    batch_size: int | None = None
    seed: int = 7
    t_steps: int = 100
    step_size: float = 0.01
    grad_norm_tol: Optional[float] = None
    ftol: Optional[float] = None
    acceptance_floor: float | None = None
    acceptance_penalty_weight: float | None = None
    acceptance_penalty_temperature: float = 0.01
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
    x_fixed: np.ndarray | None = None  # real data rows; replaces sample_states when set

    def __post_init__(self) -> None:
        estimator_aliases = {
            "finite-difference": "finite_difference",
            "stein-difference": "stein_difference",
        }
        enabled_estimators = tuple(estimator_aliases.get(name, name) for name in self.enabled_estimators)
        object.__setattr__(self, "enabled_estimators", enabled_estimators)
        if not enabled_estimators:
            raise ValueError("enabled_estimators must include at least one estimator.")
        if len(set(enabled_estimators)) != len(enabled_estimators):
            raise ValueError("enabled_estimators must not contain duplicates.")
        allowed_estimators = {
            "first_order",
            "finite_difference",
            "gauss_stein",
            "spsa",
            "stein_difference",
        }
        unknown = [name for name in enabled_estimators if name not in allowed_estimators]
        if unknown:
            allowed = ", ".join(sorted(allowed_estimators))
            unknown_list = ", ".join(unknown)
            raise ValueError(f"Unknown estimators: {unknown_list}. Allowed: {allowed}.")

        if self.perturbation_space not in {"theta", "u"}:
            raise ValueError("perturbation_space must be 'theta' or 'u'.")

        wandb_tags = tuple(self.wandb_tags)
        object.__setattr__(self, "wandb_tags", wandb_tags)
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be 'online', 'offline', or 'disabled'.")
        if self.wandb_enabled and self.wandb_mode == "disabled":
            raise ValueError("wandb_mode='disabled' is incompatible with wandb_enabled=True.")
        if self.wandb_estimator_allowlist is not None:
            wandb_allowlist = tuple(estimator_aliases.get(name, name) for name in self.wandb_estimator_allowlist)
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

        if self.perturbation_space == "u":
            policy = getattr(self.objective, "policy", None)
            if policy is None or not callable(getattr(policy, "value", None)) or not callable(getattr(policy, "grad", None)):
                raise ValueError(
                    "perturbation_space='u' requires objective.policy with value() and grad()."
                )

        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        theta0_arr = np.asarray(self.theta0, dtype=float)
        if theta0_arr.ndim != 1 or theta0_arr.size < 1:
            raise ValueError("theta0 must be a 1D array with at least one element.")
        object.__setattr__(self, "theta0", theta0_arr)

        if self.x_fixed is not None:
            x_fixed_arr = np.asarray(self.x_fixed, dtype=float)
            if x_fixed_arr.ndim != 2:
                raise ValueError("x_fixed must be a 2D array of shape (n_rows, state_dim).")
            if x_fixed_arr.shape[1] != self.state_dim:
                raise ValueError(
                    f"x_fixed has {x_fixed_arr.shape[1]} columns but state_dim={self.state_dim}."
                )
            object.__setattr__(self, "x_fixed", x_fixed_arr)

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
        if self.acceptance_floor is not None:
            if not 0.0 < self.acceptance_floor < 1.0:
                raise ValueError("acceptance_floor must be in (0, 1) when provided.")
            if self.acceptance_penalty_weight is None or self.acceptance_penalty_weight <= 0.0:
                raise ValueError(
                    "acceptance_penalty_weight must be positive when acceptance_floor is provided."
                )
            if self.acceptance_penalty_temperature <= 0.0:
                raise ValueError(
                    "acceptance_penalty_temperature must be positive when acceptance_floor is provided."
                )
            mean_acceptance_fn = getattr(self.objective, "mean_acceptance", None)
            if not callable(mean_acceptance_fn):
                raise ValueError(
                    "acceptance_floor requires an objective with mean_acceptance(theta, x_batch)."
                )
        elif self.acceptance_penalty_weight is not None:
            raise ValueError(
                "acceptance_penalty_weight requires acceptance_floor to be provided."
            )
        if self.n_grad_samples <= 0:
            raise ValueError("n_grad_samples must be positive.")

        objective = self.objective
        value_fn = getattr(objective, "value", None)
        grad_fn = getattr(objective, "grad", None)
        if value_fn is None or not callable(value_fn):
            raise ValueError("objective must implement value(theta, x_batch).")
        if grad_fn is None or not callable(grad_fn):
            raise ValueError("objective must implement grad(theta, x_batch).")

        policy = getattr(objective, "policy", None)
        if policy is not None:
            policy_value = getattr(objective, "policy_value", None)
            policy_grad = getattr(objective, "policy_grad", None)
            if not callable(policy_value):
                policy_value = getattr(policy, "value", None)
            if not callable(policy_grad):
                policy_grad = getattr(policy, "grad", None)
            if not callable(policy_value) or not callable(policy_grad):
                raise ValueError("policy must implement value(theta, x_batch) and grad(theta, x_batch).")
            # Probe with a single-sample batch
            x_probe = np.zeros((1, self.state_dim), dtype=float)
            u_probe_arr = np.asarray(policy_value(theta0_arr, x_probe), dtype=float)
            if not bool(np.isfinite(u_probe_arr).all()):
                raise ValueError("policy.value(theta0, x_batch) must be finite.")
            grad_probe = np.asarray(policy_grad(theta0_arr, x_probe), dtype=float)
            if grad_probe.ndim != 2 or grad_probe.shape[1] != theta0_arr.size:
                raise ValueError("policy.grad(theta0, x_batch) must return (n_samples, theta_dim).")

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary for JSON output."""
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
            "acceptance_floor": float(self.acceptance_floor)
            if self.acceptance_floor is not None
            else None,
            "acceptance_penalty_weight": float(self.acceptance_penalty_weight)
            if self.acceptance_penalty_weight is not None
            else None,
            "acceptance_penalty_temperature": float(self.acceptance_penalty_temperature),
            "sigma": float(self.sigma),
            "n_grad_samples": int(self.n_grad_samples),
            "verbose": bool(self.verbose),
            "plot": bool(self.plot),
            "plot_dir": self.plot_dir,
            "enabled_estimators": list(self.enabled_estimators),
            "perturbation_space": self.perturbation_space,
            "theta0": _as_list(self.theta0),
            "objective": _objective_to_dict(self.objective),
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
            "x_fixed_shape": list(self.x_fixed.shape) if self.x_fixed is not None else None,
        }


def _objective_to_dict(objective: Objective) -> dict[str, Any]:
    """Serialize objective to dictionary."""
    if isinstance(objective, FixedRegressionObjective):
        return {
            "type": "FixedRegressionObjective",
            "policy": _policy_to_dict(objective.policy),
            "beta_1": _as_list(objective.beta_1),
            "beta_2": float(objective.beta_2),
            "beta_3": _as_list(objective.beta_3),
            "beta_4": float(objective.beta_4),
        }
    if isinstance(objective, PlantedLogisticObjective):
        return {
            "type": "PlantedLogisticObjective",
            "policy": _policy_to_dict(objective.policy),
            "alpha": float(objective.alpha),
            "beta": _as_list(objective.beta),
            "bias": float(objective.bias),
            "u_star": float(objective.u_star),
        }
    if isinstance(objective, ModelBasedObjective):
        return {
            "type": "ModelBasedObjective",
            "policy": _policy_to_dict(objective.policy),
            "acceptance_state_cols": list(objective.acceptance_state_cols),
            "loss_cols": list(objective.loss_cols),
            "premium_col": int(objective.premium_col),
            "u_coef": float(objective.u_coef) if objective.u_coef is not None else None,
            "u_bounds": list(objective.u_bounds) if objective.u_bounds is not None else None,
            "acceptance_floor": float(objective.acceptance_floor)
            if objective.acceptance_floor is not None
            else None,
            "acceptance_penalty_weight": float(objective.acceptance_penalty_weight)
            if objective.acceptance_penalty_weight is not None
            else None,
            "acceptance_penalty_temperature": float(objective.acceptance_penalty_temperature),
        }
    return {"type": type(objective).__name__}


def _policy_to_dict(policy: object) -> dict[str, Any]:
    """Serialize policy to dictionary."""
    if isinstance(policy, ConstantPolicy):
        return {"type": "ConstantPolicy"}
    if isinstance(policy, LinearPolicy):
        return {"type": "LinearPolicy"}
    if isinstance(policy, SoftmaxPolicy):
        return {"type": "SoftmaxPolicy"}
    return {"type": type(policy).__name__}


def _correctness_to_dict(correctness: CorrectnessSpec) -> dict[str, Any]:
    """Serialize correctness spec to dictionary."""
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
    """Convert array-like to list of floats."""
    arr = np.asarray(values, dtype=float)
    return [float(val) for val in arr.tolist()]


# --- Factory helpers for preset configs ---


def make_fixed_regression_objective(
    *,
    policy: Policy,
    beta_1: np.ndarray | Sequence[float],
    beta_2: float,
    beta_3: np.ndarray | Sequence[float],
    beta_4: float,
) -> FixedRegressionObjective:
    """Create a FixedRegressionObjective with the given parameters."""
    return FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=np.asarray(beta_1, dtype=float),
        beta_2=float(beta_2),
        beta_3=np.asarray(beta_3, dtype=float),
        beta_4=float(beta_4),
    )


def make_planted_logistic_objective(
    *,
    policy: Policy,
    alpha: float,
    beta: np.ndarray | Sequence[float],
    bias: float,
    u_star: float,
) -> PlantedLogisticObjective:
    """Create a PlantedLogisticObjective with the given parameters."""
    return PlantedLogisticObjective.from_parameters(
        policy=policy,
        alpha=float(alpha),
        beta=np.asarray(beta, dtype=float),
        bias=float(bias),
        u_star=float(u_star),
    )


def make_softmax_policy() -> SoftmaxPolicy:
    """Create a SoftmaxPolicy instance."""
    return SoftmaxPolicy()


def make_model_based_objective(
    *,
    policy: Policy,
    acceptance_model: object,
    loss_model: object,
    acceptance_state_cols: tuple[str, ...],
    loss_cols: tuple[str, ...],
    premium_col: int = 9,
    u_coef: float | None = None,
    u_bounds: tuple[float, float] | None = None,
    acceptance_floor: float | None = None,
    acceptance_penalty_weight: float | None = None,
    acceptance_penalty_temperature: float = 0.01,
) -> ModelBasedObjective:
    """Create a ModelBasedObjective wrapping trained sklearn/XGBoost models."""
    return ModelBasedObjective(
        policy=policy,
        acceptance_model=acceptance_model,
        loss_model=loss_model,
        acceptance_state_cols=acceptance_state_cols,
        loss_cols=loss_cols,
        premium_col=premium_col,
        u_coef=u_coef,
        u_bounds=u_bounds,
        acceptance_floor=acceptance_floor,
        acceptance_penalty_weight=acceptance_penalty_weight,
        acceptance_penalty_temperature=acceptance_penalty_temperature,
    )


def canonical_training_block(
    *,
    n_samples: int,
    step_rule: str,
    t_steps: int,
    step_size: float,
    sigma: float,
    n_grad_samples: int,
    enabled_estimators: tuple[str, ...],
    perturbation_space: Literal["theta", "u"],
    batch_size: int | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    acceptance_floor: float | None = None,
    acceptance_penalty_weight: float | None = None,
    acceptance_penalty_temperature: float = 0.01,
) -> dict[str, Any]:
    """Create a canonical training configuration block."""
    return {
        "n_samples": int(n_samples),
        "step_rule": step_rule,
        "t_steps": int(t_steps),
        "step_size": float(step_size),
        "sigma": float(sigma),
        "n_grad_samples": int(n_grad_samples),
        "enabled_estimators": tuple(enabled_estimators),
        "perturbation_space": perturbation_space,
        "batch_size": int(batch_size) if batch_size is not None else None,
        "grad_norm_tol": float(grad_norm_tol) if grad_norm_tol is not None else None,
        "ftol": float(ftol) if ftol is not None else None,
        "acceptance_floor": float(acceptance_floor) if acceptance_floor is not None else None,
        "acceptance_penalty_weight": float(acceptance_penalty_weight)
        if acceptance_penalty_weight is not None
        else None,
        "acceptance_penalty_temperature": float(acceptance_penalty_temperature),
    }


def canonical_runtime_block(
    *,
    plot: bool,
    verbose: bool,
    wandb_enabled: bool,
    plot_dir: str = "plots",
    wandb_project: str | None = None,
    wandb_entity: str | None = "generali-pricing",
    wandb_group: str | None = None,
    wandb_job_type: str = "experiment",
    wandb_tags: tuple[str, ...] = (),
    wandb_mode: Literal["online", "offline", "disabled"] = "online",
    wandb_log_plots: bool = True,
    wandb_estimator_allowlist: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Create a canonical runtime configuration block."""
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
    objective: Objective,
    theta0: np.ndarray,
    training: Mapping[str, Any],
    runtime: Mapping[str, Any] | None = None,
    correctness: CorrectnessSpec | None = None,
    x_fixed: np.ndarray | None = None,
) -> ExperimentConfig:
    """Build an ExperimentConfig from component blocks.

    If ``x_fixed`` is provided, the runner uses it directly as the state array
    instead of sampling from N(0, I).
    """
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
    if x_fixed is not None:
        payload["x_fixed"] = np.asarray(x_fixed, dtype=float)
    return ExperimentConfig(**payload)
