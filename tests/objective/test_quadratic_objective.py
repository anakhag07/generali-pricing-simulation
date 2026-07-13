from __future__ import annotations

import numpy as np
import pytest

from objective import QuadraticObjective


def test_quadratic_value_gradient_and_optimum() -> None:
    objective = QuadraticObjective(dimension=3)
    theta = np.asarray([1.0, -2.0, 0.5])
    x_batch = np.asarray([[3.0], [-4.0]])

    assert objective.theta_dim() == 3
    assert objective.value(theta, x_batch) == pytest.approx(2.625)
    np.testing.assert_array_equal(objective.grad(theta, x_batch), theta)
    np.testing.assert_array_equal(objective.optimal_theta(), np.zeros(3))
    assert objective.value(objective.optimal_theta(), x_batch) == 0.0


def test_quadratic_gradient_matches_central_difference() -> None:
    objective = QuadraticObjective(dimension=4)
    theta = np.asarray([0.8, -0.3, 1.2, -2.0])
    x_batch = np.zeros((1, 2))
    epsilon = 1e-6
    estimate = np.empty_like(theta)

    for index in range(theta.size):
        direction = np.zeros_like(theta)
        direction[index] = epsilon
        estimate[index] = (
            objective.value(theta + direction, x_batch)
            - objective.value(theta - direction, x_batch)
        ) / (2.0 * epsilon)

    np.testing.assert_allclose(estimate, objective.grad(theta, x_batch), rtol=1e-9, atol=1e-9)


def test_quadratic_is_independent_of_state_values() -> None:
    objective = QuadraticObjective(dimension=2)
    theta = np.asarray([0.25, -0.75])

    value_a = objective.value(theta, np.zeros((1, 3)))
    value_b = objective.value(theta, np.full((10, 5), 100.0))

    assert value_a == value_b


@pytest.mark.parametrize("dimension", [0, -1])
def test_quadratic_rejects_nonpositive_dimension(dimension: int) -> None:
    with pytest.raises(ValueError, match="dimension must be positive"):
        QuadraticObjective(dimension=dimension)


@pytest.mark.parametrize("dimension", [True, 2.5])
def test_quadratic_rejects_noninteger_dimension(dimension: object) -> None:
    with pytest.raises(TypeError, match="dimension must be an integer"):
        QuadraticObjective(dimension=dimension)  # type: ignore[arg-type]


def test_quadratic_validates_theta_and_batch_shapes() -> None:
    objective = QuadraticObjective(dimension=2)

    with pytest.raises(ValueError, match="theta must be a 1D array with dimension 2"):
        objective.value(np.ones(3), np.zeros((1, 1)))
    with pytest.raises(ValueError, match="x_batch must be a 2D array"):
        objective.grad(np.ones(2), np.zeros(1))
