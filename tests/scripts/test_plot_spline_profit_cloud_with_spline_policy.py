"""Tests for the spline profit-cloud and spline-policy overlay."""

from __future__ import annotations

import numpy as np

from scripts.plot_spline_profit_cloud_with_spline_policy import fixed_bins, histogram_frame


def test_fixed_bins_are_one_percentage_point_wide() -> None:
    bins = fixed_bins()

    assert len(bins) == 31
    assert bins[0] == -0.1
    assert np.isclose(bins[-1], 0.2)
    np.testing.assert_allclose(np.diff(bins), 0.01)


def test_histogram_density_integrates_to_one() -> None:
    frame = histogram_frame(np.asarray([-0.1, -0.02, 0.05, 0.19, 0.2]))

    mass = np.sum(frame["density"] * (frame["bin_right"] - frame["bin_left"]))
    assert int(frame["count"].sum()) == 5
    assert np.isclose(mass, 1.0)
