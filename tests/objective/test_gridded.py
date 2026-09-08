import numpy as np
import pytest

from objective.gridded import (
    SplineSupportLowerBoundObjective,
    interpolate_customer_curves,
)
from objective.policy import SoftmaxPolicy


def test_interpolate_customer_curves_returns_values_and_local_slopes() -> None:
    grid = np.array([-0.2, 0.0, 0.2])
    curves = np.array([[1.0, 2.0, 4.0], [3.0, 1.0, 0.0]])

    values, slopes = interpolate_customer_curves(curves, grid, [-0.1, 0.1])

    np.testing.assert_allclose(values, [1.5, 0.5])
    np.testing.assert_allclose(slopes, [5.0, -5.0])


def test_interpolate_customer_curves_clips_actions_to_grid() -> None:
    values, slopes = interpolate_customer_curves(
        np.array([[1.0, 2.0], [4.0, 8.0]]),
        [0.0, 1.0],
        [-2.0, 3.0],
    )

    np.testing.assert_allclose(values, [1.0, 8.0])
    np.testing.assert_allclose(slopes, [1.0, 4.0])


@pytest.mark.parametrize(
    ("grid", "message"),
    [([0.0], "at least two"), ([0.0, 0.0], "strictly increasing")],
)
def test_interpolate_customer_curves_rejects_invalid_grid(grid, message) -> None:
    with pytest.raises(ValueError, match=message):
        interpolate_customer_curves(np.ones((1, len(grid))), grid, [0.0])


def test_spline_support_objective_gradient_matches_finite_difference() -> None:
    x_array = np.array([[-1.0], [1.0]])
    action_grid = np.array([-0.2, 0.0, 0.2])
    objective = SplineSupportLowerBoundObjective(
        policy=SoftmaxPolicy(action_low=-0.2, action_high=0.2),
        cost_grid=np.array([[0.4, 0.2, 0.3], [0.5, 0.4, 0.2]]),
        acceptance_grid=np.array([[0.9, 0.8, 0.7], [0.8, 0.7, 0.6]]),
        action_grid=action_grid,
        support_width=np.array([0.08, 0.04, 0.06]),
        acceptance_floor=0.65,
    )
    theta = np.array([0.1, -0.2])
    step = 1e-6

    analytical = objective.grad(theta, x_array)
    finite_difference = np.empty_like(theta)
    for index in range(theta.size):
        direction = np.zeros_like(theta)
        direction[index] = step
        finite_difference[index] = (
            objective.value(theta + direction, x_array)
            - objective.value(theta - direction, x_array)
        ) / (2.0 * step)

    np.testing.assert_allclose(analytical, finite_difference, atol=1e-8)

