"""Class-based optimization entry point backed by SciPy minimize."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

import numpy as np
from scipy.optimize import minimize

from model.policy import POLICY_CONSTANT, POLICY_LINEAR, POLICY_SOFTMAX, phi_batch, policy_u_batch
from objective.base import ObjectiveModel, StateVector
from optimization.steps import STEP_RULE_LBFGSB

if TYPE_CHECKING:
    from experiments.reporters import StepReporter
    from experiments.results import OptimizationTrace

TrueGradFn = Callable[[StateVector, float], float]
MinimizeFn = Callable[..., Any]


def _build_objective_batch_fns(
    objective_model: ObjectiveModel,
    x_list: Sequence[StateVector],
    x_array: np.ndarray,
) -> tuple[Callable[..., np.ndarray], Callable[..., np.ndarray]]:
    batch_builder = getattr(objective_model, "prepare_batch", None)
    if callable(batch_builder):
        batch = cast(Any, batch_builder)(x_array)
        return cast(Any, batch).value, cast(Any, batch).grad_u

    value_batch = getattr(objective_model, "value_batch", None)
    grad_batch = getattr(objective_model, "grad_u_batch", None)

    if callable(value_batch):

        def value_fn(u_vals: np.ndarray) -> np.ndarray:
            return np.asarray(value_batch(x_array, u_vals), dtype=float)

    else:

        def value_fn(u_vals: np.ndarray) -> np.ndarray:
            return np.asarray(
                [objective_model.value(x, u) for x, u in zip(x_list, u_vals)],
                dtype=float,
            )

    if callable(grad_batch):

        def grad_fn(u_vals: np.ndarray) -> np.ndarray:
            return np.asarray(grad_batch(x_array, u_vals), dtype=float)

    else:

        def grad_fn(u_vals: np.ndarray) -> np.ndarray:
            return np.asarray(
                [objective_model.grad_u(x, u) for x, u in zip(x_list, u_vals)],
                dtype=float,
            )

    return value_fn, grad_fn


class Optimization:
    """Optimize policy parameters with a pluggable gradient estimator."""

    def __init__(
        self,
        objective_model: ObjectiveModel,
        policy_kind: str,
        x_samples: Sequence[StateVector],
        gradient: Any,
        *,
        algorithm: str = STEP_RULE_LBFGSB,
        t_steps: int,
        n_grad_samples: int,
        sigma: float,
        batch_size: int | None = None,
        true_grad_u_fn: TrueGradFn | None = None,
        grad_norm_tol: float | None = None,
        ftol: float | None = None,
        step_reporter: StepReporter | None = None,
        method_label: str | None = None,
        rng: np.random.Generator | None = None,
        minimize_fn: MinimizeFn = minimize,
    ) -> None:
        self.objective_model = objective_model
        self.policy_kind = policy_kind
        self.gradient = gradient
        self.algorithm = algorithm
        self.t_steps = int(t_steps)
        self.n_grad_samples = int(n_grad_samples)
        self.sigma = float(sigma)
        self.batch_size = batch_size
        self.true_grad_u_fn = true_grad_u_fn
        self.grad_norm_tol = grad_norm_tol
        self.ftol = ftol
        self.step_reporter = step_reporter
        self.method_label = method_label if method_label is not None else getattr(gradient, "name", "opt")
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self._minimize_fn = minimize_fn

        self.x_list = list(x_samples)
        if not self.x_list:
            raise ValueError("x_samples must contain at least one StateVector.")
        self.x_array = np.stack([x.as_array() for x in self.x_list], axis=0).astype(float)
        self.phi_array_all = (
            phi_batch(self.x_array)
            if self.policy_kind in (POLICY_LINEAR, POLICY_SOFTMAX)
            else None
        )
        self.value_batch_fn_all, self.grad_batch_fn_all = _build_objective_batch_fns(
            self.objective_model,
            self.x_list,
            self.x_array,
        )
        self.n_total = self.x_array.shape[0]
        if self.batch_size is None:
            self.batch_size_eff = self.n_total
        else:
            self.batch_size_eff = int(self.batch_size)
        if self.batch_size_eff <= 0 or self.batch_size_eff > self.n_total:
            raise ValueError("batch_size must be in [1, len(x_samples)].")
        if self.grad_norm_tol is not None and self.grad_norm_tol <= 0.0:
            raise ValueError("grad_norm_tol must be positive when provided.")
        if self.ftol is not None and self.ftol <= 0.0:
            raise ValueError("ftol must be positive when provided.")
        if self.t_steps <= 0:
            raise ValueError("t_steps must be positive.")

        self._full_indices = np.arange(self.n_total, dtype=int)

    def _scipy_method(self) -> str:
        if self.algorithm.lower() == STEP_RULE_LBFGSB:
            return "L-BFGS-B"
        raise ValueError(f"Unsupported algorithm '{self.algorithm}'.")

    def sample_indices(self) -> np.ndarray:
        if self.batch_size_eff >= self.n_total:
            return self._full_indices
        return self.rng.choice(self.n_total, size=self.batch_size_eff, replace=False)

    def batch_context(
        self,
        indices: np.ndarray,
    ) -> tuple[
        Sequence[StateVector],
        np.ndarray,
        np.ndarray | None,
        Callable[..., np.ndarray],
        Callable[..., np.ndarray],
    ]:
        if indices.size == self.n_total:
            return (
                self.x_list,
                self.x_array,
                self.phi_array_all,
                self.value_batch_fn_all,
                self.grad_batch_fn_all,
            )
        x_batch = self.x_array[indices]
        x_batch_list = [self.x_list[int(i)] for i in indices]
        phi_batch_values = self.phi_array_all[indices] if self.phi_array_all is not None else None
        value_batch_fn, grad_batch_fn = _build_objective_batch_fns(
            self.objective_model,
            x_batch_list,
            x_batch,
        )
        return x_batch_list, x_batch, phi_batch_values, value_batch_fn, grad_batch_fn

    def theta_grad_from_u_grad(
        self,
        theta: np.ndarray,
        phi_array: np.ndarray | None,
        u_vals: np.ndarray,
        grad_u_vals: np.ndarray,
    ) -> np.ndarray:
        n_samples = float(u_vals.size)
        if self.policy_kind == POLICY_CONSTANT:
            grad = np.zeros_like(theta)
            grad[0] = float(np.mean(grad_u_vals))
            return grad
        if self.policy_kind == POLICY_LINEAR:
            if phi_array is None:
                raise ValueError("phi_array is required for linear policies.")
            return (phi_array.T @ grad_u_vals) / n_samples
        if self.policy_kind == POLICY_SOFTMAX:
            if phi_array is None:
                raise ValueError("phi_array is required for softmax policies.")
            sigma_vals = u_vals - 0.5
            du_dz = sigma_vals * (1.0 - sigma_vals)
            weights = grad_u_vals * du_dz
            return (phi_array.T @ weights) / n_samples
        raise ValueError(
            f"Policy kind must be one of {(POLICY_CONSTANT, POLICY_LINEAR, POLICY_SOFTMAX)}."
        )

    def objective_on_indices(self, theta_arr: np.ndarray, indices: np.ndarray) -> float:
        _, x_batch, phi_batch_values, value_batch_fn, _ = self.batch_context(indices)
        u_vals = self.policy_u_batch(theta_arr, x_batch, phi_batch_values)
        return float(np.mean(value_batch_fn(u_vals)))

    def policy_u_batch(
        self,
        theta_arr: np.ndarray,
        x_batch: np.ndarray,
        phi_batch_values: np.ndarray | None,
    ) -> np.ndarray:
        return policy_u_batch(
            theta_arr,
            x_batch,
            kind=self.policy_kind,
            phi_array=phi_batch_values,
        )

    def solve(self, theta_start: np.ndarray) -> tuple[np.ndarray, "OptimizationTrace"]:
        theta0 = np.asarray(theta_start, dtype=float)
        self.gradient.setup(self, theta0)

        steps: list[int] = []
        u_values: list[float] = []
        values: list[float] = []
        u_grad_estimates: list[float] = []
        u_true_grads: list[float] = []
        theta_grad_norms: list[float] = []
        true_theta_grad_norms: list[float] = []
        theta_values: list[np.ndarray] = []

        def value_fn(theta_vec: np.ndarray) -> float:
            theta_arr = np.asarray(theta_vec, dtype=float)
            indices = self.sample_indices()
            return self.objective_on_indices(theta_arr, indices)

        def grad_fn(theta_vec: np.ndarray) -> np.ndarray:
            theta_arr = np.asarray(theta_vec, dtype=float)
            indices = self.sample_indices()
            grad_theta, _ = self.gradient.theta_and_u_grad(self, theta_arr, indices)
            return grad_theta

        def record(theta_vec: np.ndarray) -> None:
            theta_arr = np.asarray(theta_vec, dtype=float)
            indices = self.sample_indices()
            x_batch_list, x_batch, phi_batch_values, value_batch_fn, _ = self.batch_context(indices)
            u_vals = self.policy_u_batch(theta_arr, x_batch, phi_batch_values)
            value = float(np.mean(value_batch_fn(u_vals)))
            grad_theta, grad_u_vals = self.gradient.theta_and_u_grad(self, theta_arr, indices)
            theta_grad_norm = float(np.linalg.norm(grad_theta))

            true_theta_grad_norm = None
            mean_true_grad_u = None
            if self.true_grad_u_fn is not None:
                true_grad_u_vals = np.asarray(
                    [self.true_grad_u_fn(x, u) for x, u in zip(x_batch_list, u_vals)],
                    dtype=float,
                )
                mean_true_grad_u = float(np.mean(true_grad_u_vals))
                grad_theta_true = self.theta_grad_from_u_grad(
                    theta_arr,
                    phi_batch_values,
                    u_vals,
                    true_grad_u_vals,
                )
                true_theta_grad_norm = float(np.linalg.norm(grad_theta_true))

            step = len(steps)
            mean_u = float(np.mean(u_vals))
            mean_grad = float(np.mean(grad_u_vals)) if grad_u_vals is not None else float("nan")

            steps.append(step)
            u_values.append(mean_u)
            values.append(value)
            u_grad_estimates.append(mean_grad)
            theta_grad_norms.append(theta_grad_norm)
            theta_values.append(theta_arr.copy())
            if mean_true_grad_u is not None:
                u_true_grads.append(mean_true_grad_u)
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
            method=self._scipy_method(),
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
            u_true_gradients=u_true_grads if u_true_grads else None,
            theta_grad_norms=theta_grad_norms,
            true_theta_grad_norms=true_theta_grad_norms if true_theta_grad_norms else None,
            theta_values=theta_values,
            optimizer_status=int(result.status),
            optimizer_message=str(result.message),
        )
        return theta_final, trace


__all__ = ["Optimization", "TrueGradFn"]
