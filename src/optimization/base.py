"""Class-based optimization entry point backed by SciPy minimize."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from objective.base import ActionObjective, StateVector, ThetaObjective
from objective.composed import PolicyObjective
from objective.policy import policy_from_kind
from optimization.helpers import (
    mean_action_on_indices,
    objective_value_on_indices,
    sample_indices,
    scipy_method,
    x_batch,
)
from optimization.steps import STEP_RULE_LBFGSB

if TYPE_CHECKING:
    from experiments.reporters import StepReporter
    from experiments.results import OptimizationTrace

TrueThetaGradFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
MinimizeFn = Callable[..., Any]


class Optimization:
    """Optimize theta for a theta-level objective with pluggable gradients."""

    def __init__(
        self,
        objective: ThetaObjective | ActionObjective,
        x_samples_or_policy: Sequence[StateVector] | str,
        gradient_or_x_samples: Any,
        gradient: Any | None = None,
        *,
        algorithm: str = STEP_RULE_LBFGSB,
        t_steps: int,
        n_grad_samples: int,
        sigma: float,
        batch_size: int | None = None,
        true_grad_theta_fn: TrueThetaGradFn | None = None,
        grad_norm_tol: float | None = None,
        ftol: float | None = None,
        step_reporter: StepReporter | None = None,
        method_label: str | None = None,
        rng: np.random.Generator | None = None,
        minimize_fn: MinimizeFn = minimize,
    ) -> None:
        if isinstance(x_samples_or_policy, str):
            if gradient is None:
                raise ValueError("gradient must be provided when using legacy policy_kind signature.")
            policy_kind = x_samples_or_policy
            x_samples = gradient_or_x_samples
            self.objective = PolicyObjective(
                action_objective=objective,
                policy=policy_from_kind(policy_kind),
            )
            self.gradient = gradient
        else:
            self.objective = objective
            x_samples = x_samples_or_policy
            self.gradient = gradient_or_x_samples
        self.algorithm = algorithm
        self.t_steps = int(t_steps)
        self.n_grad_samples = int(n_grad_samples)
        self.sigma = float(sigma)
        self.batch_size = batch_size
        self.true_grad_theta_fn = true_grad_theta_fn
        self.grad_norm_tol = grad_norm_tol
        self.ftol = ftol
        self.step_reporter = step_reporter
        self.method_label = method_label if method_label is not None else getattr(self.gradient, "name", "opt")
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._minimize_fn = minimize_fn

        x_list = list(x_samples)
        if not x_list:
            raise ValueError("x_samples must contain at least one StateVector.")
        self.x_array = np.stack([np.asarray(x, dtype=float) for x in x_list], axis=0).astype(float)
        self.n_total = self.x_array.shape[0]
        self.batch_size_eff = self.n_total if batch_size is None else int(batch_size)
        if self.batch_size_eff <= 0 or self.batch_size_eff > self.n_total:
            raise ValueError("batch_size must be in [1, len(x_samples)].")
        if self.grad_norm_tol is not None and self.grad_norm_tol <= 0.0:
            raise ValueError("grad_norm_tol must be positive when provided.")
        if self.ftol is not None and self.ftol <= 0.0:
            raise ValueError("ftol must be positive when provided.")
        if self.t_steps <= 0:
            raise ValueError("t_steps must be positive.")
        self._full_indices = np.arange(self.n_total, dtype=int)

    def solve(self, theta_start: np.ndarray) -> tuple[np.ndarray, "OptimizationTrace"]:
        theta0 = np.asarray(theta_start, dtype=float)
        self.gradient.setup(self, theta0)

        steps: list[int] = []
        u_values: list[float] = []
        values: list[float] = []
        u_grad_estimates: list[float] = []
        theta_grad_norms: list[float] = []
        true_theta_grad_norms: list[float] = []
        theta_values: list[np.ndarray] = []

        def value_fn(theta_vec: np.ndarray) -> float:
            theta_arr = np.asarray(theta_vec, dtype=float)
            indices = sample_indices(self.rng, self.batch_size_eff, self.n_total, self._full_indices)
            return objective_value_on_indices(self.objective, self.x_array, self.n_total, theta_arr, indices)

        def grad_fn(theta_vec: np.ndarray) -> np.ndarray:
            theta_arr = np.asarray(theta_vec, dtype=float)
            indices = sample_indices(self.rng, self.batch_size_eff, self.n_total, self._full_indices)
            return np.asarray(self.gradient.theta_grad(self, theta_arr, indices), dtype=float)

        def record(theta_vec: np.ndarray) -> None:
            theta_arr = np.asarray(theta_vec, dtype=float)
            indices = sample_indices(self.rng, self.batch_size_eff, self.n_total, self._full_indices)
            value = objective_value_on_indices(self.objective, self.x_array, self.n_total, theta_arr, indices)
            grad_theta = np.asarray(self.gradient.theta_grad(self, theta_arr, indices), dtype=float)
            theta_grad_norm = float(np.linalg.norm(grad_theta))

            true_theta_grad_norm = None
            if self.true_grad_theta_fn is not None:
                grad_true = np.asarray(
                    self.true_grad_theta_fn(theta_arr, x_batch(self.x_array, indices, self.n_total)),
                    dtype=float,
                )
                true_theta_grad_norm = float(np.linalg.norm(grad_true))

            step = len(steps)
            mean_u = mean_action_on_indices(self.objective, self.x_array, self.n_total, theta_arr, indices)

            steps.append(step)
            u_values.append(mean_u)
            values.append(value)
            u_grad_estimates.append(float("nan"))
            theta_grad_norms.append(theta_grad_norm)
            theta_values.append(theta_arr.copy())
            if true_theta_grad_norm is not None:
                true_theta_grad_norms.append(true_theta_grad_norm)
            if self.step_reporter is not None:
                self.step_reporter.log_step(self.method_label, step, mean_u, value, theta_grad_norm)

        record(theta0)

        options: dict[str, float | int] = {"maxiter": int(self.t_steps)}
        if self.grad_norm_tol is not None:
            options["gtol"] = float(self.grad_norm_tol)
        if self.ftol is not None:
            options["ftol"] = float(self.ftol)

        result = self._minimize_fn(
            value_fn,
            x0=theta0,
            jac=grad_fn,
            method=scipy_method(self.algorithm),
            options=options,
            callback=record,
        )

        theta_final = np.asarray(result.x, dtype=float)
        if not np.allclose(theta_final, theta_values[-1]):
            record(theta_final)

        from experiments.results import OptimizationTrace

        trace = OptimizationTrace(
            steps=steps,
            u_values=u_values,
            objective_values=values,
            u_grad_estimates=u_grad_estimates,
            u_true_gradients=None,
            theta_grad_norms=theta_grad_norms,
            true_theta_grad_norms=true_theta_grad_norms if true_theta_grad_norms else None,
            theta_values=theta_values,
            optimizer_status=int(result.status),
            optimizer_message=str(result.message),
        )
        return theta_final, trace


__all__ = ["Optimization", "TrueThetaGradFn"]
