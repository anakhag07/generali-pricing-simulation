"""Optax-driven manual update loop for the class-based optimizer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax

from optimization.helpers import sample_indices
from optimization.steps import (
    OPTAX_STEP_RULES,
    STEP_RULE_OPTAX_ADAM,
    STEP_RULE_OPTAX_SGD,
)

jax.config.update("jax_enable_x64", True)

if TYPE_CHECKING:
    from optimization.base import Optimization

RecordFn = Callable[..., None]


def optax_step_rule_optimizer(algorithm: str, step_size: float) -> optax.GradientTransformation:
    """Return the optax gradient transformation for a supported optax step rule."""
    if algorithm == STEP_RULE_OPTAX_ADAM:
        return optax.adam(learning_rate=float(step_size))
    if algorithm == STEP_RULE_OPTAX_SGD:
        return optax.sgd(learning_rate=float(step_size))
    allowed = ", ".join(OPTAX_STEP_RULES)
    raise ValueError(f"Unsupported optax step rule '{algorithm}'. Allowed: {allowed}.")


def run_optax_minimize_loop(
    optimizer: "Optimization",
    theta0: np.ndarray,
    record: RecordFn,
) -> tuple[np.ndarray, bool, int, str]:
    """Run optax updates with theta grads from the configured GradientMethod.

    Mirrors the manual constant/armijo loop: each step samples a mini-batch
    with ``batch_rng``, estimates the theta gradient, applies the optax
    update, and records trace metrics. The update itself is deterministic
    given the gradient stream, so no additional seed stream is introduced.
    """
    transformation = optax_step_rule_optimizer(optimizer.algorithm, optimizer.step_size)
    theta = jnp.asarray(np.asarray(theta0, dtype=float), dtype=jnp.float64)
    opt_state = transformation.init(theta)

    success = False
    status = 1
    message = "STOP: reached maximum iterations"
    for _ in range(optimizer.t_steps):
        indices = sample_indices(
            optimizer.batch_rng,
            optimizer.batch_size_eff,
            optimizer.n_total,
            optimizer._full_indices,
        )
        theta_np = np.asarray(theta, dtype=float)
        grad_theta = np.asarray(
            optimizer.gradient.theta_grad(optimizer, theta_np, indices), dtype=float
        )
        grad_norm = float(np.linalg.norm(grad_theta))
        if optimizer.grad_norm_tol is not None and grad_norm <= optimizer.grad_norm_tol:
            success = True
            status = 0
            message = "STOP: gradient norm below tolerance"
            break
        updates, opt_state = transformation.update(
            jnp.asarray(grad_theta, dtype=jnp.float64), opt_state, theta
        )
        theta = optax.apply_updates(theta, updates)
        record(np.asarray(theta, dtype=float), step_size=optimizer.step_size)

    return np.asarray(theta, dtype=float), success, status, message


__all__ = ["optax_step_rule_optimizer", "run_optax_minimize_loop"]
