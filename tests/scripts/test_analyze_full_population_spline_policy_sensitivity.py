"""Tests for full-population spline-policy sensitivity analysis."""

from __future__ import annotations

import numpy as np

from scripts.analyze_full_population_spline_policy_sensitivity import (
    action_region_table,
    mean_abs_sensitivity_scores,
)


class _LinearDerivativeCache:
    def derivative(self, rows, grid):
        row_scale = np.asarray(rows, dtype=float)[:, None] + 1.0
        return -row_scale * (1.0 + np.asarray(grid, dtype=float)[None, :])


def test_mean_abs_sensitivity_scores_are_chunk_invariant() -> None:
    rows = np.arange(7)
    grid = np.asarray([-0.1, 0.0, 0.2])
    expected = mean_abs_sensitivity_scores(
        _LinearDerivativeCache(), rows, grid, chunk_rows=7
    )
    chunked = mean_abs_sensitivity_scores(
        _LinearDerivativeCache(), rows, grid, chunk_rows=2
    )

    np.testing.assert_allclose(chunked, expected)


def test_action_region_table_reports_sensitivity_composition() -> None:
    actions = np.asarray([-0.1, -0.05, 0.19, 0.2, 0.0, 0.195])
    ranks = np.asarray([0, 0, 1, 2, 1, 2])
    table = action_region_table(actions, ranks)
    upper = table[table["action_region"] == "upper_endpoint_bin"].set_index(
        "sensitivity_bucket"
    )

    assert upper.loc["low", "count"] == 0
    assert upper.loc["medium", "count"] == 1
    assert upper.loc["high", "count"] == 2
    assert np.isclose(upper["share_within_action_region"].sum(), 1.0)
