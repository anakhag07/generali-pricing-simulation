"""Experiment configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from data.fixed_objective import FixedRegressionObjective
from data.planted_logistic import PlantedLogisticObjective
from data.models import ObjectiveModel
from optimization.policy import POLICY_LINEAR, POLICY_SOFTMAX, PolicySpec
from optimization.steps import STEP_RULES


@dataclass(frozen=True)
class ExperimentConfig:
    state_dim: int
    objective_model: ObjectiveModel
    policy_spec: PolicySpec
    n_samples: int
    step_rule: str
    seed: int = 7
    t_steps: int = 100
    step_size: float = 0.01
    sigma: float = 0.1
    n_grad_samples: int = 64
    lbfgs_maxiter: int = 200
    lbfgs_seed: Optional[int] = None
    log_steps: bool = True
    plot: bool = True
    plot_dir: str = "plots"
    enabled_estimators: tuple[str, ...] = ("first_order", "zeroth_order", "lbfgs")

    def __post_init__(self) -> None:
        enabled_estimators = tuple(self.enabled_estimators)
        object.__setattr__(self, "enabled_estimators", enabled_estimators)
        if not enabled_estimators:
            raise ValueError("enabled_estimators must include at least one estimator.")
        if len(set(enabled_estimators)) != len(enabled_estimators):
            raise ValueError("enabled_estimators must not contain duplicates.")
        allowed_estimators = {"first_order", "zeroth_order", "lbfgs"}
        unknown = [name for name in enabled_estimators if name not in allowed_estimators]
        if unknown:
            allowed = ", ".join(sorted(allowed_estimators))
            unknown_list = ", ".join(unknown)
            raise ValueError(f"Unknown estimators: {unknown_list}. Allowed: {allowed}.")

        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")

        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        if self.step_rule not in STEP_RULES:
            allowed = ", ".join(sorted(STEP_RULES))
            raise ValueError(f"step_rule must be one of {allowed}.")

        if self.step_size <= 0.0:
            raise ValueError("step_size must be positive.")

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_dim": int(self.state_dim),
            "n_samples": int(self.n_samples),
            "step_rule": self.step_rule,
            "seed": int(self.seed),
            "t_steps": int(self.t_steps),
            "step_size": float(self.step_size),
            "sigma": float(self.sigma),
            "n_grad_samples": int(self.n_grad_samples),
            "lbfgs_maxiter": int(self.lbfgs_maxiter),
            "lbfgs_seed": int(self.lbfgs_seed) if self.lbfgs_seed is not None else None,
            "log_steps": bool(self.log_steps),
            "plot": bool(self.plot),
            "plot_dir": self.plot_dir,
            "enabled_estimators": list(self.enabled_estimators),
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


def _as_list(values: object) -> list[float]:
    arr = np.asarray(values, dtype=float)
    return [float(val) for val in arr.tolist()]
