"""Focused tests for the synthetic-GP lower-bound policy analysis."""

from __future__ import annotations

import numpy as np

from scripts import run_spline_gp_lower_bound_policy as script


def test_synthetic_gp_uncertainty_is_low_inside_and_high_outside() -> None:
    posterior_std, design = script._synthetic_gp_posterior_std(script.ACTION_GRID)

    np.testing.assert_allclose(
        design,
        np.linspace(script.GP_SUPPORT_LOW, script.GP_SUPPORT_HIGH, 33),
    )
    inside = (script.ACTION_GRID >= 0.0) & (script.ACTION_GRID <= 0.16)
    assert float(np.max(posterior_std[inside])) < 0.5
    assert float(posterior_std[0]) > 0.99 * script.GP_AMPLITUDE
    assert float(posterior_std[-1]) > 0.99 * script.GP_AMPLITUDE
    offsets = np.linspace(0.0, 0.04, 41)
    left = np.interp(-offsets, script.ACTION_GRID, posterior_std)
    right = np.interp(0.16 + offsets, script.ACTION_GRID, posterior_std)
    np.testing.assert_allclose(left, right, atol=2e-10)


def test_gp_lower_bound_frame_uses_saved_profit_as_posterior_mean() -> None:
    mean_profit = 125.0 + 100.0 * script.ACTION_GRID

    frame, _ = script._gp_lower_bound_frame(mean_profit)

    np.testing.assert_allclose(frame["posterior_mean_profit"], mean_profit)
    np.testing.assert_allclose(
        frame["gp_lower_confidence_bound_profit"],
        mean_profit - frame["posterior_std_profit"],
    )
    np.testing.assert_allclose(
        frame["gp_upper_confidence_bound_profit"],
        mean_profit + frame["posterior_std_profit"],
    )


def test_gp_plot_is_a_vector_pdf(tmp_path) -> None:
    frame, _ = script._gp_lower_bound_frame(125.0 + 100.0 * script.ACTION_GRID)
    output_path = tmp_path / "gp_lower_bound.pdf"

    script._plot_gp_lower_bound(frame, output_path)

    assert output_path.read_bytes().startswith(b"%PDF")
