from __future__ import annotations

import numpy as np

from scripts.run_coverage_aware_policy_optimizer import (
    interpolate_rows,
    normalize_coverage_widths,
)


def test_normalize_coverage_widths_uses_each_customers_best_support() -> None:
    support = np.asarray(
        [
            [1.0, 2.0, 4.0],
            [8.0, 4.0, 2.0],
        ]
    )

    widths = normalize_coverage_widths(support, scale=10.0)

    np.testing.assert_allclose(
        widths,
        np.asarray(
            [
                [7.5, 5.0, 0.0],
                [0.0, 5.0, 7.5],
            ]
        ),
    )


def test_interpolate_rows_returns_customer_specific_values_and_slopes() -> None:
    grid = np.asarray([0.0, 0.1, 0.2])
    values = np.asarray(
        [
            [0.0, 1.0, 3.0],
            [4.0, 2.0, 1.0],
        ]
    )

    interpolated, slope = interpolate_rows(values, grid, np.asarray([0.15, 0.05]))

    np.testing.assert_allclose(interpolated, np.asarray([2.0, 3.0]))
    np.testing.assert_allclose(slope, np.asarray([20.0, -20.0]))


def test_interpolate_rows_clips_to_action_grid_endpoints() -> None:
    grid = np.asarray([0.0, 0.1, 0.2])
    values = np.asarray([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])

    interpolated, _ = interpolate_rows(values, grid, np.asarray([-0.1, 0.3]))

    np.testing.assert_allclose(interpolated, np.asarray([2.0, 2.0]))
