from __future__ import annotations

import numpy as np
import pytest

from objective.base import ObjectiveResult, StateVector
from model.policy import POLICY_LINEAR, PolicySpec
from reporting.visualization import select_theta_axes_max_variance, theta_objective_contour_grid


class QuadraticObjective:
    def value(self, x: StateVector, u: float) -> float:
        return u**2

    def grad_u(self, x: StateVector, u: float) -> float:
        return 2.0 * u

    def evaluate(self, x: StateVector, u: float) -> ObjectiveResult:
        return ObjectiveResult(value=self.value(x, u), grad_u=self.grad_u(x, u))


def test_theta_objective_contour_grid_shapes() -> None:
    x = StateVector(values=[1.0, -1.0])
    theta_base = np.asarray([0.1, 0.2, 0.3], dtype=float)
    policy_spec = PolicySpec(theta=theta_base, kind=POLICY_LINEAR)

    grid_x, grid_y, objective_grid = theta_objective_contour_grid(
        [x],
        QuadraticObjective(),
        policy_spec,
        theta_base,
        axis_indices=(0, 1),
        theta_refs=[theta_base, theta_base + 0.05],
        grid_size=20,
    )

    assert grid_x.shape == (20, 20)
    assert grid_y.shape == (20, 20)
    assert objective_grid.shape == (20, 20)


def test_theta_objective_contour_grid_rejects_invalid_axes() -> None:
    x = StateVector(values=[1.0, 0.5])
    theta_base = np.asarray([0.1, 0.2, 0.3], dtype=float)
    policy_spec = PolicySpec(theta=theta_base, kind=POLICY_LINEAR)

    with pytest.raises(ValueError, match="distinct"):
        theta_objective_contour_grid(
            [x],
            QuadraticObjective(),
            policy_spec,
            theta_base,
            axis_indices=(1, 1),
        )

    with pytest.raises(ValueError, match="valid indices"):
        theta_objective_contour_grid(
            [x],
            QuadraticObjective(),
            policy_spec,
            theta_base,
            axis_indices=(0, 5),
        )


def test_select_theta_axes_max_variance_orders_by_variance() -> None:
    theta_points = [
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.array([1.0, 2.0, 0.5], dtype=float),
        np.array([2.0, 4.0, 1.0], dtype=float),
        np.array([3.0, 6.0, 1.5], dtype=float),
    ]
    axis_indices = select_theta_axes_max_variance(theta_points)
    assert axis_indices == (1, 0)
