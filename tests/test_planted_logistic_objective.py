from __future__ import annotations

import numpy as np

from objective import PlantedLogisticObjective, SoftmaxPolicy


def test_planted_logistic_minimum_at_u_star() -> None:
    """Test that the gradient w.r.t. u is zero at u* and u* minimizes the objective."""
    objective = PlantedLogisticObjective.from_parameters(
        policy=SoftmaxPolicy(),
        alpha=1.2,
        beta=np.asarray([0.3, -0.1], dtype=float),
        bias=0.05,
        u_star=1.1,
    )
    x_batch = np.array([[0.2, -0.4]], dtype=float)
    u_star = objective.optimal_u()
    
    # Gradient at u* should be zero
    u_star_arr = np.array([u_star], dtype=float)
    grad_at_star = objective._grad_u_batch(x_batch, u_star_arr)[0]
    assert abs(grad_at_star) < 1e-8

    # Value at u* should be minimum
    value_star = objective._value_batch(x_batch, u_star_arr)[0]
    value_left = objective._value_batch(x_batch, np.array([u_star - 0.1]))[0]
    value_right = objective._value_batch(x_batch, np.array([u_star + 0.1]))[0]
    assert value_star <= value_left + 1e-10
    assert value_star <= value_right + 1e-10


def test_planted_logistic_optimal_u() -> None:
    """Test optimal_u method returns the planted u_star."""
    objective = PlantedLogisticObjective.from_parameters(
        policy=SoftmaxPolicy(),
        alpha=1.0,
        beta=[0.1, 0.2],
        bias=0.0,
        u_star=1.5,
    )
    assert objective.optimal_u() == 1.5
