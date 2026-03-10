from __future__ import annotations

import numpy as np

from objective.base import StateVector
from objective.planted_logistic import PlantedLogisticObjective


def test_planted_logistic_minimum_at_u_star() -> None:
    objective = PlantedLogisticObjective(
        alpha=1.2,
        beta=np.asarray([0.3, -0.1], dtype=float),
        bias=0.05,
        u_star=1.1,
    )
    x = StateVector(values=np.asarray([0.2, -0.4], dtype=float))
    u_star = objective.optimal_u()
    grad_at_star = objective.grad_u(x, u_star)
    assert abs(grad_at_star) < 1e-8

    value_star = objective.value(x, u_star)
    value_left = objective.value(x, u_star - 0.1)
    value_right = objective.value(x, u_star + 0.1)
    assert value_star <= value_left + 1e-10
    assert value_star <= value_right + 1e-10
