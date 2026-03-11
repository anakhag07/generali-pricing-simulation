"""Helper routines for running optimization experiments."""

from __future__ import annotations

from typing import Any, Callable, Sequence, cast

import numpy as np
from scipy.optimize import minimize

from objective.base import ObjectiveModel, StateVector
from experiments.config import CorrectnessSpec
from experiments.reporters import StepReporter
from experiments.results import OptimizationTrace
from optimization.solvers import (
    run_first_order_minimize,
    run_spsa_minimize,
    run_zeroth_order_minimize,
)
from model.policy import POLICY_CONSTANT, POLICY_LINEAR, POLICY_SOFTMAX, phi_batch, policy_u_batch


ObjectiveFn = Callable[[float], float]
TrueGradFn = Callable[[StateVector, float], float]


def _clamp_u(value: float, bounds: tuple[float, float] | None) -> float:
    """Clamp u to bounds when provided."""
    if bounds is None:
        return float(value)
    lower, upper = bounds
    return float(min(max(value, lower), upper))


def _numdiff_grad(
    value_fn: ObjectiveFn,
    u: float,
    method: str,
    step: float,
    bounds: tuple[float, float] | None,
) -> float:
    """Approximate d/du using finite differences around u."""
    u_base = float(u)
    if method == "central":
        u_plus = _clamp_u(u_base + step, bounds)
        u_minus = _clamp_u(u_base - step, bounds)
        denom = u_plus - u_minus
        if denom == 0.0:
            return 0.0
        return float((value_fn(u_plus) - value_fn(u_minus)) / denom)
    if method == "forward":
        u_plus = _clamp_u(u_base + step, bounds)
        denom = u_plus - u_base
        if denom == 0.0:
            return 0.0
        return float((value_fn(u_plus) - value_fn(u_base)) / denom)
    if method == "backward":
        u_minus = _clamp_u(u_base - step, bounds)
        denom = u_base - u_minus
        if denom == 0.0:
            return 0.0
        return float((value_fn(u_base) - value_fn(u_minus)) / denom)
    raise ValueError(f"Unknown numdiff method '{method}'.")


def resolve_true_grad_u_fn(
    objective_model: ObjectiveModel,
    correctness: CorrectnessSpec,
) -> TrueGradFn | None:
    """Return a per-sample u-gradient proxy based on correctness settings."""
    if correctness.gradient_source == "none":
        return None
    if correctness.gradient_source == "exact":
        return lambda x, u: objective_model.grad_u(x, u)
    if correctness.gradient_source == "numdiff":
        if correctness.numdiff_aggregate != "per-sample":
            raise ValueError(
                "numdiff_aggregate='batch' is not supported for theta-gradient correctness."
            )

        def numdiff_grad(x: StateVector, u: float) -> float:
            def value_fn(u_val: float) -> float:
                return objective_model.value(x, u_val)

            return _numdiff_grad(
                value_fn,
                u,
                correctness.numdiff_method,
                correctness.numdiff_step,
                correctness.numdiff_bounds,
            )

        return numdiff_grad
    raise ValueError(f"Unknown gradient_source '{correctness.gradient_source}'.")


def _build_objective_batch_fns(
    objective_model: ObjectiveModel,
    x_list: Sequence[StateVector],
    x_array: np.ndarray,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return vectorized value/grad evaluators over x_array."""
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


def run_first_order(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    rng: np.random.Generator,
    t_steps: int,
    step_rule: str,
    step_size: float,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None = None,
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Optimize theta using SciPy minimize with exact u-gradients."""
    return run_first_order_minimize(
        theta_start=theta_start,
        policy_kind=policy_kind,
        x_samples=x_samples,
        objective_model=objective_model,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_u_fn=true_grad_u_fn,
        grad_norm_tol=grad_norm_tol,
        step_reporter=step_reporter,
    )


def run_zeroth_order(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    rng: np.random.Generator,
    t_steps: int,
    step_rule: str,
    step_size: float,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None = None,
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Optimize theta using SciPy minimize with Stein u-gradient estimates."""
    return run_zeroth_order_minimize(
        theta_start=theta_start,
        policy_kind=policy_kind,
        x_samples=x_samples,
        objective_model=objective_model,
        rng=rng,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_u_fn=true_grad_u_fn,
        grad_norm_tol=grad_norm_tol,
        step_reporter=step_reporter,
    )


def run_spsa(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    rng: np.random.Generator,
    t_steps: int,
    step_rule: str,
    step_size: float,
    n_grad_samples: int,
    sigma: float,
    batch_size: int | None = None,
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, OptimizationTrace]:
    """Optimize theta using SciPy minimize with SPSA theta-gradient estimates."""
    return run_spsa_minimize(
        theta_start=theta_start,
        policy_kind=policy_kind,
        x_samples=x_samples,
        objective_model=objective_model,
        rng=rng,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_u_fn=true_grad_u_fn,
        grad_norm_tol=grad_norm_tol,
        step_reporter=step_reporter,
    )


def run_lbfgs_theta(
    theta_start: np.ndarray,
    policy_kind: str,
    x_samples: Sequence[StateVector],
    objective_model: ObjectiveModel,
    maxiter: int,
    true_grad_u_fn: TrueGradFn | None = None,
    grad_norm_tol: float | None = None,
    step_reporter: StepReporter | None = None,
) -> tuple[np.ndarray, float, OptimizationTrace]:
    """Run L-BFGS-B on theta and record trace diagnostics."""
    x_list = list(x_samples)
    if not x_list:
        raise ValueError("x_samples must contain at least one StateVector.")
    if grad_norm_tol is not None and grad_norm_tol <= 0.0:
        raise ValueError("grad_norm_tol must be positive when provided.")
    theta0 = np.asarray(theta_start, dtype=float)

    x_array = np.stack([x.as_array() for x in x_list], axis=0).astype(float)
    phi_array = (
        phi_batch(x_array)
        if policy_kind in (POLICY_LINEAR, POLICY_SOFTMAX)
        else None
    )
    value_batch_fn, grad_batch_fn = _build_objective_batch_fns(
        objective_model, x_list, x_array
    )

    def value_fn(theta_vec: np.ndarray) -> float:
        theta_arr = np.asarray(theta_vec, dtype=float)
        u_vals = policy_u_batch(theta_arr, x_array, kind=policy_kind, phi_array=phi_array)
        return float(np.mean(value_batch_fn(u_vals)))

    def grad_fn(theta_vec: np.ndarray) -> np.ndarray:
        theta_arr = np.asarray(theta_vec, dtype=float)
        u_vals = policy_u_batch(theta_arr, x_array, kind=policy_kind, phi_array=phi_array)
        grad_u_vals = grad_batch_fn(u_vals)
        n_samples = float(len(x_list))
        if policy_kind == POLICY_CONSTANT:
            grad = np.zeros_like(theta_arr)
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
        raise ValueError(f"Policy kind must be one of {(POLICY_CONSTANT, POLICY_LINEAR, POLICY_SOFTMAX)}.")

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
        u_vals = policy_u_batch(theta_arr, x_array, kind=policy_kind, phi_array=phi_array)
        mean_u = float(np.mean(u_vals))
        mean_value = float(np.mean(value_batch_fn(u_vals)))
        grad_u_vals = grad_batch_fn(u_vals)
        mean_grad_u = float(np.mean(grad_u_vals))
        true_grad_u_vals = (
            [true_grad_u_fn(x, u) for x, u in zip(x_list, u_vals)]
            if true_grad_u_fn is not None
            else []
        )
        mean_true_grad_u = (
            float(np.mean(true_grad_u_vals)) if true_grad_u_vals else None
        )
        grad_theta = grad_fn(theta_arr)
        theta_grad_norm = float(np.linalg.norm(grad_theta))
        if true_grad_u_vals:
            grad_theta_true = np.zeros_like(theta_arr)
            u_vals_arr = np.asarray(u_vals, dtype=float)
            n_samples = float(len(x_list))
            if policy_kind == POLICY_CONSTANT:
                grad_theta_true[0] = float(np.mean(true_grad_u_vals))
            elif policy_kind == POLICY_LINEAR:
                if phi_array is None:
                    raise ValueError("phi_array is required for linear policies.")
                grad_theta_true = (phi_array.T @ np.asarray(true_grad_u_vals)) / n_samples
            elif policy_kind == POLICY_SOFTMAX:
                if phi_array is None:
                    raise ValueError("phi_array is required for softmax policies.")
                sigma_vals = u_vals_arr - 0.5
                du_dz = sigma_vals * (1.0 - sigma_vals)
                weights = np.asarray(true_grad_u_vals) * du_dz
                grad_theta_true = (phi_array.T @ weights) / n_samples
            else:
                raise ValueError(
                    f"Policy kind must be one of {(POLICY_CONSTANT, POLICY_LINEAR, POLICY_SOFTMAX)}."
                )
            true_theta_grad_norm = float(np.linalg.norm(grad_theta_true))
        else:
            true_theta_grad_norm = None
        current_step = len(steps)
        steps.append(current_step)
        u_values.append(mean_u)
        values.append(mean_value)
        u_grad_estimates.append(mean_grad_u)
        if mean_true_grad_u is not None:
            u_true_grads.append(mean_true_grad_u)
        theta_grad_norms.append(theta_grad_norm)
        if true_theta_grad_norm is not None:
            true_theta_grad_norms.append(true_theta_grad_norm)
        theta_values.append(theta_arr.copy())
        # Log step metrics (L-BFGS doesn't have user-controlled step_size)
        if step_reporter is not None:
            step_reporter.log_step("lbfgs", current_step, mean_u, mean_value, theta_grad_norm)

    record(theta0)

    def callback(theta_vec: np.ndarray) -> None:
        record(theta_vec)

    options: dict[str, float | int] = {"maxiter": int(maxiter)}
    if grad_norm_tol is not None:
        options["gtol"] = float(grad_norm_tol)

    result = minimize(
        value_fn,
        x0=theta0,
        jac=grad_fn,
        method="L-BFGS-B",
        options=options,
        callback=callback,
    )
    theta_lbfgs = np.asarray(result.x, dtype=float)
    value_lbfgs = value_fn(theta_lbfgs)
    if not np.allclose(theta_lbfgs, theta_values[-1]):
        record(theta_lbfgs)

    trace = OptimizationTrace(
        steps=steps,
        u_values=u_values,
        objective_values=values,
        u_grad_estimates=u_grad_estimates,
        u_true_gradients=u_true_grads if u_true_grads else None,
        theta_grad_norms=theta_grad_norms,
        true_theta_grad_norms=true_theta_grad_norms if true_theta_grad_norms else None,
        theta_values=theta_values,
    )
    return theta_lbfgs, value_lbfgs, trace
