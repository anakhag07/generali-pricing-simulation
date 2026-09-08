"""Tests for the full-population spline-policy histograms."""

from __future__ import annotations

import numpy as np

from scripts.plot_full_population_spline_policy_histograms import (
    bounded,
    fixed_bins,
    histogram_records,
)


def test_fixed_bins_use_one_percentage_point_width() -> None:
    bounded_edges = fixed_bins(0.0, 0.16)
    wide_edges = fixed_bins(-0.1, 0.2)

    assert len(bounded_edges) == 17
    assert len(wide_edges) == 31
    np.testing.assert_allclose(np.diff(bounded_edges), 0.01)
    np.testing.assert_allclose(np.diff(wide_edges), 0.01)


def test_bounded_filters_then_clips_endpoint_roundoff() -> None:
    values = np.asarray([-0.01, -1e-14, 0.08, 0.16 + 1e-14, 0.17])
    selected = bounded(values, 0.0, 0.16)

    np.testing.assert_allclose(selected, [0.0, 0.08, 0.16])


def test_histogram_density_integrates_to_one() -> None:
    bins = fixed_bins(0.0, 0.16)
    records = histogram_records(
        plot="optimized",
        series="Optimized",
        values=np.asarray([0.0, 0.08, 0.159, 0.16]),
        bins=bins,
    )

    mass = sum(
        record["density"] * (record["bin_right"] - record["bin_left"])
        for record in records
    )
    assert np.isclose(mass, 1.0)
