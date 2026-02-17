"""Experiment configuration definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from optimization.policy import POLICY_LINEAR, POLICY_SOFTMAX, PolicySpec

OBJECTIVE_STOCHASTIC = "stochastic"
OBJECTIVE_FIXED_REGRESSION = "fixed_regression"
OBJECTIVE_KINDS = (OBJECTIVE_STOCHASTIC, OBJECTIVE_FIXED_REGRESSION)


def _default_beta_1(dim: int) -> np.ndarray:
    if dim == 3:
        return np.asarray([0.02, 0.2, 0.5], dtype=float)
    return np.linspace(0.02, 0.5, num=dim, dtype=float)


def _default_beta_3(dim: int) -> np.ndarray:
    if dim == 3:
        return np.asarray([0.005, 0.1, 0.2], dtype=float)
    return np.linspace(0.005, 0.2, num=dim, dtype=float)


def _default_policy_spec(dim: int) -> PolicySpec:
    theta = np.asarray([0.1] + [0.01] * dim, dtype=float)
    return PolicySpec(theta=theta, kind=POLICY_SOFTMAX)


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    state_dim: int = 3
    previous_policy_price: float = 1000.0
    t_steps: int = 300
    step_size: float = 0.01
    sigma: float = 0.1
    n_samples: int = 64
    lbfgs_maxiter: int = 200
    lbfgs_samples: int = 128
    lbfgs_seed: Optional[int] = None
    objective_kind: str = OBJECTIVE_FIXED_REGRESSION
    beta_1: Optional[np.ndarray] = None
    beta_2: float = -0.8
    beta_3: Optional[np.ndarray] = None
    beta_4: float = 0.4
    policy_spec: Optional[PolicySpec] = None
    plot: bool = True
    plot_dir: str = "plots"

    def __post_init__(self) -> None:
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive.")
        if self.objective_kind not in OBJECTIVE_KINDS:
            raise ValueError(f"objective_kind must be one of {OBJECTIVE_KINDS}.")

        beta_1 = (
            _default_beta_1(self.state_dim)
            if self.beta_1 is None
            else np.asarray(self.beta_1, dtype=float)
        )
        beta_3 = (
            _default_beta_3(self.state_dim)
            if self.beta_3 is None
            else np.asarray(self.beta_3, dtype=float)
        )
        beta_2 = float(self.beta_2)
        beta_4 = float(self.beta_4)

        if beta_1.size < self.state_dim:
            raise ValueError("beta_1 must have at least state_dim elements.")
        if beta_3.size < self.state_dim:
            raise ValueError("beta_3 must have at least state_dim elements.")
        if np.any(beta_1 <= 0.0):
            raise ValueError("beta_1 entries must be positive.")
        if np.any(beta_3 <= 0.0):
            raise ValueError("beta_3 entries must be positive.")
        if beta_2 >= 0.0:
            raise ValueError(
                "beta_2 must be negative; acceptance probability should decrease as policy value increases."
            )
        if beta_4 <= 0.0:
            raise ValueError("beta_4 must be positive.")
        if self.lbfgs_maxiter <= 0:
            raise ValueError("lbfgs_maxiter must be positive.")
        if self.lbfgs_samples <= 0:
            raise ValueError("lbfgs_samples must be positive.")

        policy_spec = (
            _default_policy_spec(self.state_dim)
            if self.policy_spec is None
            else self.policy_spec
        )
        if policy_spec.kind in (POLICY_LINEAR, POLICY_SOFTMAX):
            required = self.state_dim + 1
            if policy_spec.theta.size < required:
                raise ValueError(
                    "Policy theta must have at least state_dim + 1 elements for linear/softmax policies."
                )

        object.__setattr__(self, "beta_1", beta_1)
        object.__setattr__(self, "beta_2", beta_2)
        object.__setattr__(self, "beta_3", beta_3)
        object.__setattr__(self, "beta_4", beta_4)
        object.__setattr__(self, "policy_spec", policy_spec)
        if self.lbfgs_seed is None:
            object.__setattr__(self, "lbfgs_seed", int(self.seed + 997))
