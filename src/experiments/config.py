"""Experiment configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from data.fixed_objective import FixedRegressionObjective
from data.models import ObjectiveModel
from optimization.policy import POLICY_LINEAR, POLICY_SOFTMAX, PolicySpec


@dataclass(frozen=True)
class ExperimentConfig:
    state_dim: int
    objective_model: ObjectiveModel
    policy_spec: PolicySpec
    n_samples: int
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
