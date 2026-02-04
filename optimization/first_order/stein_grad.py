"""First-order Stein gradient estimators using oracle gradients."""

from __future__ import annotations

from typing import Callable

import numpy as np

from optimization.common import clip_u, gaussian_noise
from optimization.objective import ObjectiveResult


OracleGradFn = Callable[[float], ObjectiveResult]


def stein_first_order_grad(
    u: float,
    oracle_grad_fn: OracleGradFn,
    rng: np.random.Generator,
    n_samples: int = 32,
    sigma: float = 0.1,
) -> float:
    """Estimate d/du of smoothed objective using oracle gradients.

    Uses E[grad f(u + sigma * eps)] where eps ~ N(0, 1).
    """

    grads = []
    for _ in range(n_samples):
        eps = float(gaussian_noise(rng))
        u_perturbed = clip_u(u + sigma * eps)
        result = oracle_grad_fn(u_perturbed)
        grads.append(result.grad_u)

    return float(np.mean(grads)) if grads else 0.0
