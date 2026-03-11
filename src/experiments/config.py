"""Experiment configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np

from model.policy import POLICY_LINEAR, POLICY_SOFTMAX, PolicySpec
from objective.base import ObjectiveModel
from objective.fixed_objective import FixedRegressionObjective
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
    sigma: float = 0.1
    n_grad_samples: int = 64
    lbfgs_maxiter: int = 200
    lbfgs_seed: Optional[int] = None
    verbose: bool = False
    plot: bool = True
    plot_dir: str = "plots"
    enabled_estimators: tuple[str, ...] = ("first_order", "zeroth_order", "lbfgs")
    correctness: CorrectnessSpec = field(default_factory=CorrectnessSpec)

    def __post_init__(self) -> None:
        enabled_estimators = tuple(self.enabled_estimators)
        object.__setattr__(self, "enabled_estimators", enabled_estimators)
        if not enabled_estimators:
            raise ValueError("enabled_estimators must include at least one estimator.")
        if len(set(enabled_estimators)) != len(enabled_estimators):
            raise ValueError("enabled_estimators must not contain duplicates.")
        allowed_estimators = {"first_order", "zeroth_order", "spsa", "lbfgs"}
        unknown = [name for name in enabled_estimators if name not in allowed_estimators]
        if unknown:
            allowed = ", ".join(sorted(allowed_estimators))
            unknown_list = ", ".join(unknown)
            raise ValueError(f"Unknown estimators: {unknown_list}. Allowed: {allowed}.")

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

        if self.n_grad_samples <= 0:
            raise ValueError("n_grad_samples must be positive.")

        if self.lbfgs_maxiter <= 0:
            raise ValueError("lbfgs_maxiter must be positive.")

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
        if self.lbfgs_seed is None:
            object.__setattr__(self, "lbfgs_seed", int(self.seed + 997))

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
            "sigma": float(self.sigma),
            "n_grad_samples": int(self.n_grad_samples),
            "lbfgs_maxiter": int(self.lbfgs_maxiter),
            "lbfgs_seed": int(self.lbfgs_seed) if self.lbfgs_seed is not None else None,
            "verbose": bool(self.verbose),
            "plot": bool(self.plot),
            "plot_dir": self.plot_dir,
            "enabled_estimators": list(self.enabled_estimators),
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
