"""Helper routines for running optimization experiments."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from objective.base import ActionObjective, StateVector, ThetaObjective
from objective.composed import PolicyObjective
from objective.policy import policy_from_kind
from experiments.config import CorrectnessSpec
from experiments.reporters import StepReporter
from experiments.results import OptimizationTrace
from optimization.solvers import (
    run_first_order_minimize,
    run_gauss_stein_minimize,
    run_spsa_minimize,
)


TrueThetaGradFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _clamp_theta(theta: np.ndarray, bounds: tuple[float, float] | None) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    if bounds is None:
        return theta_arr
    lower, upper = bounds
    return np.clip(theta_arr, lower, upper)


def _numdiff_theta_grad(
    objective: ThetaObjective,
    theta: np.ndarray,
    x_batch: np.ndarray,
    method: str,
    step: float,
    bounds: tuple[float, float] | None,
) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    grad = np.zeros_like(theta_arr)
    if method not in {"central", "forward", "backward"}:
        raise ValueError(f"Unknown numdiff method '{method}'.")

    for idx in range(theta_arr.size):
        basis = np.zeros_like(theta_arr)
        basis[idx] = 1.0
        if method == "central":
            theta_plus = _clamp_theta(theta_arr + step * basis, bounds)
            theta_minus = _clamp_theta(theta_arr - step * basis, bounds)
            denom = theta_plus[idx] - theta_minus[idx]
            if denom == 0.0:
                grad[idx] = 0.0
            else:
                grad[idx] = (objective.value(theta_plus, x_batch) - objective.value(theta_minus, x_batch)) / denom
            continue
        if method == "forward":
            theta_plus = _clamp_theta(theta_arr + step * basis, bounds)
            denom = theta_plus[idx] - theta_arr[idx]
            if denom == 0.0:
                grad[idx] = 0.0
            else:
                grad[idx] = (objective.value(theta_plus, x_batch) - objective.value(theta_arr, x_batch)) / denom
            continue
        theta_minus = _clamp_theta(theta_arr - step * basis, bounds)
        denom = theta_arr[idx] - theta_minus[idx]
        if denom == 0.0:
            grad[idx] = 0.0
        else:
            grad[idx] = (objective.value(theta_arr, x_batch) - objective.value(theta_minus, x_batch)) / denom
    return grad


def resolve_true_grad_theta_fn(
    objective: ThetaObjective,
    correctness: CorrectnessSpec,
) -> TrueThetaGradFn | None:
    """Return a theta-gradient proxy based on correctness settings."""
    if correctness.gradient_source == "none":
        return None
    if correctness.gradient_source == "exact":
        return lambda theta, x_batch: objective.grad(theta, x_batch)
    if correctness.gradient_source == "numdiff":
        return lambda theta, x_batch: _numdiff_theta_grad(
            objective,
            theta,
            x_batch,
            correctness.numdiff_method,
            correctness.numdiff_step,
            correctness.numdiff_bounds,
        )
    raise ValueError(f"Unknown gradient_source '{correctness.gradient_source}'.")


def _resolve_optimization_inputs(
    args: tuple,
    kwargs: dict,
) -> tuple[
    Sequence[StateVector],
    ThetaObjective,
    np.random.Generator,
    int,
    str,
    float,
    int,
    float,
    int | None,
    TrueThetaGradFn | None,
    float | None,
    float | None,
    StepReporter | None,
]:
    if not args:
        raise ValueError("Missing optimization arguments.")

    if isinstance(args[0], str):
        if len(args) < 4:
            raise ValueError("Legacy signature requires policy_kind, x_samples, objective_model, and rng.")
        policy_kind = args[0]
        x_samples = args[1]
        action_objective = args[2]
        objective = PolicyObjective(action_objective=action_objective, policy=policy_from_kind(policy_kind))
        rng = args[3]
        remaining = args[4:]
    else:
        if len(args) < 3:
            raise ValueError("Signature requires x_samples, objective, and rng.")
        x_samples = args[0]
        objective = args[1]
        rng = args[2]
        remaining = args[3:]

    names = ["t_steps", "step_rule", "step_size", "n_grad_samples", "sigma", "batch_size"]
    values: dict[str, object] = {}
    for name, value in zip(names, remaining):
        values[name] = value

    for name in names:
        if name in kwargs:
            values[name] = kwargs.pop(name)

    true_grad_theta_fn = kwargs.pop("true_grad_theta_fn", None)
    kwargs.pop("true_grad_u_fn", None)
    grad_norm_tol = kwargs.pop("grad_norm_tol", None)
    ftol = kwargs.pop("ftol", None)
    step_reporter = kwargs.pop("step_reporter", None)

    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unexpected optimization kwargs: {unexpected}")

    required = ["t_steps", "step_rule", "step_size", "n_grad_samples", "sigma"]
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(f"Missing required optimization args: {', '.join(missing)}")

    return (
        x_samples,
        objective,
        rng,
        int(values["t_steps"]),
        str(values["step_rule"]),
        float(values["step_size"]),
        int(values["n_grad_samples"]),
        float(values["sigma"]),
        int(values["batch_size"]) if values.get("batch_size") is not None else None,
        true_grad_theta_fn,
        float(grad_norm_tol) if grad_norm_tol is not None else None,
        float(ftol) if ftol is not None else None,
        step_reporter,
    )


def run_first_order(theta_start: np.ndarray, *args: object, **kwargs: object) -> tuple[np.ndarray, OptimizationTrace]:
    (
        x_samples,
        objective,
        rng,
        t_steps,
        step_rule,
        step_size,
        n_grad_samples,
        sigma,
        batch_size,
        true_grad_theta_fn,
        grad_norm_tol,
        ftol,
        step_reporter,
    ) = _resolve_optimization_inputs(tuple(args), dict(kwargs))
    del rng, step_rule, step_size
    return run_first_order_minimize(
        theta_start=theta_start,
        x_samples=x_samples,
        objective=objective,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
    )


def run_gauss_stein(theta_start: np.ndarray, *args: object, **kwargs: object) -> tuple[np.ndarray, OptimizationTrace]:
    (
        x_samples,
        objective,
        rng,
        t_steps,
        step_rule,
        step_size,
        n_grad_samples,
        sigma,
        batch_size,
        true_grad_theta_fn,
        grad_norm_tol,
        ftol,
        step_reporter,
    ) = _resolve_optimization_inputs(tuple(args), dict(kwargs))
    del step_rule, step_size
    return run_gauss_stein_minimize(
        theta_start=theta_start,
        x_samples=x_samples,
        objective=objective,
        rng=rng,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
    )


def run_spsa(theta_start: np.ndarray, *args: object, **kwargs: object) -> tuple[np.ndarray, OptimizationTrace]:
    (
        x_samples,
        objective,
        rng,
        t_steps,
        step_rule,
        step_size,
        n_grad_samples,
        sigma,
        batch_size,
        true_grad_theta_fn,
        grad_norm_tol,
        ftol,
        step_reporter,
    ) = _resolve_optimization_inputs(tuple(args), dict(kwargs))
    del step_rule, step_size
    return run_spsa_minimize(
        theta_start=theta_start,
        x_samples=x_samples,
        objective=objective,
        rng=rng,
        t_steps=t_steps,
        n_grad_samples=n_grad_samples,
        sigma=sigma,
        batch_size=batch_size,
        true_grad_theta_fn=true_grad_theta_fn,
        grad_norm_tol=grad_norm_tol,
        ftol=ftol,
        step_reporter=step_reporter,
    )
