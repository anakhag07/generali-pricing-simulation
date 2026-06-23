"""Step-size rules for optimization routines."""

from __future__ import annotations

from typing import Callable

import numpy as np

STEP_RULE_CONSTANT = "constant"
STEP_RULE_ARMIJO = "armijo"
STEP_RULE_LBFGSB = "l-bfgs-b"
STEP_RULE_TRUST_CONSTR = "trust-constr"
STEP_RULE_OPTAX_ADAM = "optax-adam"
STEP_RULES = (
    STEP_RULE_CONSTANT,
    STEP_RULE_ARMIJO,
    STEP_RULE_LBFGSB,
    STEP_RULE_TRUST_CONSTR,
    STEP_RULE_OPTAX_ADAM,
)

ObjectiveThetaFn = Callable[[np.ndarray], float]


def constant_step_size(step_size: float) -> float:
    return float(step_size)


def armijo_backtracking_step_size(
    theta: np.ndarray,
    grad: np.ndarray,
    objective_fn: ObjectiveThetaFn,
    initial_step: float,
    c: float = 1e-4,
    shrink: float = 0.5,
    max_backtracks: int = 20,
    min_step: float = 1e-6,
) -> float:
    """Armijo backtracking: find $$\\alpha = \\alpha_0 \\rho^i$$ s.t. $$J(\\theta+\\alpha d) \\le J(\\theta)+c\\alpha\\nabla J^\\top d$$."""
    if initial_step <= 0.0:
        raise ValueError("initial_step must be positive.")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be non-negative.")
    if min_step <= 0.0:
        raise ValueError("min_step must be positive.")
    if not (0.0 < shrink < 1.0):
        raise ValueError("shrink must be in (0, 1).")
    if c <= 0.0:
        raise ValueError("c must be positive.")

    theta_arr = np.asarray(theta, dtype=float)
    grad_arr = np.asarray(grad, dtype=float)
    if theta_arr.shape != grad_arr.shape:
        raise ValueError("theta and grad must have the same shape.")

    direction = -grad_arr
    grad_dir = float(np.dot(grad_arr, direction))
    base_value = float(objective_fn(theta_arr))

    if grad_dir == 0.0:
        return float(max(min_step, initial_step))

    for i in range(max_backtracks + 1):
        step = initial_step * (shrink**i)
        if step < min_step:
            step = min_step
        candidate = theta_arr + step * direction
        candidate_value = float(objective_fn(candidate))
        if candidate_value <= base_value + c * step * grad_dir:
            return float(step)
        if step <= min_step:
            break
    return float(min_step)
