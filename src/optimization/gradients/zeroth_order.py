"""Zeroth-order Stein estimator using function evaluations."""

from __future__ import annotations

from typing import Callable

import numpy as np

from optimization.common import gaussian_noise


ObjectiveFn = Callable[[float], float]


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
        # u_perturbed = clip_u(u + sigma * eps)
        u_perturbed = u + sigma * eps
        value = float(objective_fn(u_perturbed))
        estimates.append(value * eps)

    if not estimates:
        return 0.0
    return float(np.mean(estimates) / max(sigma, 1e-8))
