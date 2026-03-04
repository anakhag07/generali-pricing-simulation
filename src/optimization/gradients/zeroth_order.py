"""Zeroth-order Stein estimator using function evaluations."""

from __future__ import annotations

from typing import Callable

import numpy as np

from optimization.common import gaussian_noise


ObjectiveFn = Callable[[float], float]
ObjectiveBatchFn = Callable[[np.ndarray], np.ndarray]


def stein_zeroth_order_grad(
    u: float,
    objective_fn: ObjectiveFn,
    rng: np.random.Generator,
    n_samples: int = 64,
    sigma: float = 0.1,
) -> float:
    """Estimate d/du using E[f(u + sigma * eps) * eps] / sigma."""

    estimates = []
    for _ in range(n_samples):
        eps = float(gaussian_noise(rng))
        u_perturbed = u + sigma * eps
        value = float(objective_fn(u_perturbed))
        estimates.append(value * eps)

    if not estimates:
        return 0.0
    return float(np.mean(estimates) / max(sigma, 1e-8))


def stein_zeroth_order_grad_batch(
    u_values: np.ndarray,
    objective_fn: ObjectiveBatchFn,
    rng: np.random.Generator,
    n_samples: int = 64,
    sigma: float = 0.1,
) -> np.ndarray:
    """Estimate d/du for each u using batched Stein samples."""
    u_arr = np.asarray(u_values, dtype=float)
    if u_arr.size == 0:
        return np.asarray([], dtype=float)
    if n_samples <= 0:
        return np.zeros_like(u_arr, dtype=float)

    eps = gaussian_noise(rng, shape=(n_samples, u_arr.size))
    accum = np.zeros_like(u_arr, dtype=float)
    for i in range(n_samples):
        u_perturbed = u_arr + sigma * eps[i]
        values = np.asarray(objective_fn(u_perturbed), dtype=float)
        accum = accum + values * eps[i]
    return accum / float(n_samples) / max(sigma, 1e-8)
