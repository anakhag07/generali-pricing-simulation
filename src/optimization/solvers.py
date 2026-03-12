"""SciPy-based optimization solvers for theta updates."""

from __future__ import annotations

from typing import Any, Callable, Literal, Sequence, cast

import numpy as np
from scipy.optimize import minimize

from objective.base import ObjectiveModel, StateVector
from experiments.reporters import StepReporter
from experiments.results import OptimizationTrace
from model.policy import POLICY_CONSTANT, POLICY_LINEAR, POLICY_SOFTMAX, phi_batch, policy_u_batch


TrueGradFn = Callable[[StateVector, float], float]


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


def _theta_grad_from_u_grad(
    theta: np.ndarray,
    policy_kind: str,
    phi_array: np.ndarray | None,
    u_vals: np.ndarray,
    grad_u_vals: np.ndarray,
) -> np.ndarray:
    n_samples = float(u_vals.size)
    if policy_kind == POLICY_CONSTANT:
        grad = np.zeros_like(theta)
        grad[0] = float(np.mean(grad_u_vals))
        return grad
    if policy_kind == POLICY_LINEAR:
        if phi_array is None:
            raise ValueError("phi_array is required for linear policies.")
        return (phi_array.T @ grad_u_vals) / n_samples
    if policy_kind == POLICY_SOFTMAX:
        if phi_array is None:
            raise ValueError("phi_array is required for softmax policies.")
        sigma_vals = u_vals - 0.5
        du_dz = sigma_vals * (1.0 - sigma_vals)
        weights = grad_u_vals * du_dz
        return (phi_array.T @ weights) / n_samples
    raise ValueError(
        f"Policy kind must be one of {(POLICY_CONSTANT, POLICY_LINEAR, POLICY_SOFTMAX)}."
    )


def _run_minimize_solver(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    t_steps: int,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None,
    grad_kind: Literal["first_order", "gauss_stein", "spsa"],
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
    method_label: str = "first-order",
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    x_list = list(x_samples)
    if not x_list:
        raise ValueError("x_samples must contain at least one StateVector.")
    if grad_norm_tol is not None and grad_norm_tol <= 0.0:
        raise ValueError("grad_norm_tol must be positive when provided.")
    if ftol is not None and ftol <= 0.0:
        raise ValueError("ftol must be positive when provided.")

    theta0 = np.asarray(theta_start, dtype=float)
    x_array = np.stack([x.as_array() for x in x_list], axis=0).astype(float)
    phi_array_all = (
        phi_batch(x_array) if policy_kind in (POLICY_LINEAR, POLICY_SOFTMAX) else None
    )
    value_batch_fn_all, grad_batch_fn_all = _build_objective_batch_fns(objective_model, x_list, x_array)
    n_total = x_array.shape[0]
    if batch_size is None:
        batch_size_eff = n_total
    else:
        batch_size_eff = int(batch_size)
    if batch_size_eff <= 0 or batch_size_eff > n_total:
        raise ValueError("batch_size must be in [1, len(x_samples)].")
    source_rng = rng if rng is not None else np.random.default_rng(0)
    full_indices = np.arange(n_total, dtype=int)

    def _sample_indices() -> np.ndarray:
        if batch_size_eff >= n_total:
            return full_indices
        return source_rng.choice(n_total, size=batch_size_eff, replace=False)

    def _batch_context(
        indices: np.ndarray,
    ) -> tuple[
        Sequence[StateVector],
        np.ndarray,
        np.ndarray | None,
        Callable[..., np.ndarray],
        Callable[..., np.ndarray],
    ]:
        if indices.size == n_total:
            return x_list, x_array, phi_array_all, value_batch_fn_all, grad_batch_fn_all
        x_batch = x_array[indices]
        x_batch_list = [x_list[int(i)] for i in indices]
        phi_batch_values = phi_array_all[indices] if phi_array_all is not None else None
        value_batch_fn, grad_batch_fn = _build_objective_batch_fns(
            objective_model,
            x_batch_list,
            x_batch,
        )
        return x_batch_list, x_batch, phi_batch_values, value_batch_fn, grad_batch_fn

    eps_base = None
    delta_base = None
    if grad_kind == "gauss_stein":
        if n_grad_samples <= 0:
            raise ValueError("n_grad_samples must be positive.")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        eps_base = source_rng.normal(0.0, 1.0, size=(n_grad_samples, x_array.shape[0])).astype(float)
    if grad_kind == "spsa":
        if n_grad_samples <= 0:
            raise ValueError("n_grad_samples must be positive.")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive.")
        delta_base = source_rng.choice(
            np.asarray([-1.0, 1.0], dtype=float),
            size=(n_grad_samples, theta0.size),
        )

    def u_grad_values(
        u_vals: np.ndarray,
        value_batch_fn: Callable[..., np.ndarray],
        grad_batch_fn: Callable[..., np.ndarray],
        indices: np.ndarray,
    ) -> np.ndarray:
        if grad_kind == "first_order":
            return grad_batch_fn(u_vals)
        if eps_base is None:
            raise ValueError("eps_base is required for gauss-stein gradients.")
        eps_values = eps_base[:, indices]
        accum = np.zeros_like(u_vals, dtype=float)
        for eps in eps_values:
            values = value_batch_fn(u_vals + sigma * eps)
            accum += values * eps
        return accum / float(eps_values.shape[0]) / max(sigma, 1e-8)

    def _objective_on_indices(theta_arr: np.ndarray, indices: np.ndarray) -> float:
        _, x_batch, phi_batch_values, value_batch_fn, _ = _batch_context(indices)
        u_vals = policy_u_batch(
            theta_arr,
            x_batch,
            kind=policy_kind,
            phi_array=phi_batch_values,
        )
        return float(np.mean(value_batch_fn(u_vals)))

    def value_fn(theta_vec: np.ndarray) -> float:
        theta_arr = np.asarray(theta_vec, dtype=float)
        indices = _sample_indices()
        return _objective_on_indices(theta_arr, indices)

    def grad_fn(theta_vec: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta_vec, dtype=float)
        indices = _sample_indices()
        if grad_kind == "spsa":
            if delta_base is None:
                raise ValueError("delta_base is required for SPSA gradients.")
            grad_theta = np.zeros_like(theta_arr, dtype=float)
            for delta in delta_base:
                value_plus = _objective_on_indices(theta_arr + sigma * delta, indices)
                value_minus = _objective_on_indices(theta_arr - sigma * delta, indices)
                grad_theta += ((value_plus - value_minus) / (2.0 * sigma)) * delta
            return grad_theta / float(delta_base.shape[0])
        _, x_batch, phi_batch_values, value_batch_fn, grad_batch_fn = _batch_context(indices)
        u_vals = policy_u_batch(theta_arr, x_batch, kind=policy_kind, phi_array=phi_batch_values)
        grad_u_vals = u_grad_values(u_vals, value_batch_fn, grad_batch_fn, indices)
        return _theta_grad_from_u_grad(
            theta_arr,
            policy_kind,
            phi_batch_values,
            u_vals,
            grad_u_vals,
        )

    steps: list[int] = []
    u_values: list[float] = []
    values: list[float] = []
    u_grad_estimates: list[float] = []
    u_true_grads: list[float] = []
    theta_grad_norms: list[float] = []
    true_theta_grad_norms: list[float] = []
    theta_values: list[np.ndarray] = []

    def record(theta_vec: np.ndarray) -> None:
        theta_arr = np.asarray(theta_vec, dtype=float)
        indices = _sample_indices()
        x_batch_list, x_batch, phi_batch_values, value_batch_fn, grad_batch_fn = _batch_context(indices)
        u_vals = policy_u_batch(theta_arr, x_batch, kind=policy_kind, phi_array=phi_batch_values)
        value = float(np.mean(value_batch_fn(u_vals)))
        if grad_kind == "spsa":
            grad_u_vals = None
            grad_theta = grad_fn(theta_arr)
        else:
            grad_u_vals = u_grad_values(u_vals, value_batch_fn, grad_batch_fn, indices)
            grad_theta = _theta_grad_from_u_grad(
                theta_arr,
                policy_kind,
                phi_batch_values,
                u_vals,
                grad_u_vals,
            )
        theta_grad_norm = float(np.linalg.norm(grad_theta))

        true_theta_grad_norm = None
        mean_true_grad_u = None
        if true_grad_u_fn is not None:
            true_grad_u_vals = np.asarray(
                [true_grad_u_fn(x, u) for x, u in zip(x_batch_list, u_vals)],
                dtype=float,
            )
            mean_true_grad_u = float(np.mean(true_grad_u_vals))
            grad_theta_true = _theta_grad_from_u_grad(
                theta_arr,
                policy_kind,
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
        if step_reporter is not None:
            step_reporter.log_step(method_label, step, mean_u, value, theta_grad_norm)

    record(theta0)

    options: dict[str, float | int] = {"maxiter": int(t_steps)}
    if grad_norm_tol is not None:
        options["gtol"] = float(grad_norm_tol)
    if ftol is not None:
        options["ftol"] = float(ftol)

    result = minimize(
        value_fn,
        x0=theta0,
        jac=grad_fn,
        method="L-BFGS-B",
        options=options,
        callback=record,
    )

    theta_final = np.asarray(result.x, dtype=float)
    if not np.allclose(theta_final, theta_values[-1]):
        record(theta_final)

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


def run_first_order_minimize(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    t_steps: int,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None = None,
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    return _run_minimize_solver(
        theta_start=theta_start,
        policy_kind=policy_kind,
        x_samples=x_samples,
        objective_model=objective_model,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        grad_kind="first_order",
        true_grad_u_fn=true_grad_u_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
        method_label="first-order",
    )


def run_gauss_stein_minimize(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    rng: np.random.Generator,
    t_steps: int,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None = None,
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    return _run_minimize_solver(
        theta_start=theta_start,
        policy_kind=policy_kind,
        x_samples=x_samples,
        objective_model=objective_model,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        grad_kind="gauss_stein",
        true_grad_u_fn=true_grad_u_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
        method_label="gauss-stein",
        rng=rng,
    )


def run_spsa_minimize(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    rng: np.random.Generator,
    t_steps: int,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None = None,
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    ftol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    return _run_minimize_solver(
        theta_start=theta_start,
        policy_kind=policy_kind,
        x_samples=x_samples,
        objective_model=objective_model,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        grad_kind="spsa",
        true_grad_u_fn=true_grad_u_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
        method_label="spsa",
        rng=rng,
    )


__all__ = ["run_first_order_minimize", "run_gauss_stein_minimize", "run_spsa_minimize"]
