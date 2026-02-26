from __future__ import annotations

import numpy as np

from optimization.steps import armijo_backtracking_step_size


def test_armijo_backtracking_reduces_objective() -> None:
    def objective(theta: np.ndarray) -> float:
        return float(theta[0] ** 2)

    theta = np.asarray([1.0], dtype=float)
    grad = np.asarray([2.0], dtype=float)
    step = armijo_backtracking_step_size(
        theta,
        grad,
        objective_fn=objective,
        initial_step=10.0,
    )

    new_theta = theta - step * grad
    assert step <= 10.0
    assert step >= 1e-6
    assert objective(new_theta) <= objective(theta) + 1e-12
