"""Tests for the monotone-spline local-support cloud script."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from reporting.profit_dispersion import row_index_sha256
from scripts import plot_monotone_spline_support_cloud as script


def _write_inputs(tmp_path):
    row_indices = np.arange(20_000)
    diagnostics = tmp_path / "coverage_diagnostics.npz"
    np.savez_compressed(
        diagnostics,
        row_indices=row_indices,
        observed_u=np.linspace(-0.2, 0.3, row_indices.size),
    )

    curve_u = np.linspace(-0.1, 0.2, 301)
    curves = pd.DataFrame(
        {
            "model_family": "spline",
            "acceptance_model": "monotone_spline_xgb",
            "loss_model": "xgb",
            "u": curve_u,
            "center_statistic": "mean",
            "center": 100.0 + 50.0 * curve_u,
        }
    )
    curve_csv = tmp_path / "curves.csv"
    curves.to_csv(curve_csv, index=False)
    curve_manifest = tmp_path / "curve_manifest.json"
    curve_manifest.write_text(
        json.dumps(
            {
                "sample": {
                    "seed": 20260831,
                    "row_indices_sha256": row_index_sha256(row_indices),
                },
                "model_pairs": {
                    "spline": {"acceptance": "monotone_spline_xgb", "loss": "xgb"}
                },
                "objective": {
                    "class": "ModelBasedObjective",
                    "minimization_formula": "acceptance * (loss - revenue)",
                    "plot_transform": "profit_i(u) = -objective_i(u)",
                },
            }
        ),
        encoding="utf-8",
    )
    return curve_csv, curve_manifest, diagnostics


def test_load_cloud_data_reuses_exact_sample_and_support_width(tmp_path) -> None:
    curve_csv, curve_manifest, diagnostics = _write_inputs(tmp_path)
    median_support = np.linspace(1.0, 10.0, script.SUPPORT_GRID.size)
    original = script._compute_median_local_support
    script._compute_median_local_support = lambda *args, **kwargs: median_support

    try:
        cloud, provenance = script._load_cloud_data(
            curve_csv,
            curve_manifest,
            diagnostics,
            n_neighbors=500,
            n_jobs=1,
        )
    finally:
        script._compute_median_local_support = original

    assert len(cloud) == 301
    np.testing.assert_allclose(cloud["u"], np.linspace(-0.1, 0.2, 301))
    np.testing.assert_allclose(
        cloud["support_cloud_lower_profit"],
        cloud["smoothed_mean_profit"] - cloud["smoothed_support_half_width"],
    )
    np.testing.assert_allclose(
        cloud["support_cloud_upper_profit"],
        cloud["smoothed_mean_profit"] + cloud["smoothed_support_half_width"],
    )
    assert np.all(cloud["smoothed_support_half_width"] > 0.0)
    assert provenance["n_customers"] == 20_000


def test_absolute_support_width_does_not_subtract_baseline_risk() -> None:
    relative_support, absolute_risk, half_width = (
        script._absolute_support_half_width(np.asarray([1.0, 4.0, 9.0]))
    )

    np.testing.assert_allclose(relative_support, [1.0 / 9.0, 4.0 / 9.0, 1.0])
    np.testing.assert_allclose(absolute_risk, [3.0, 1.5, 1.0])
    np.testing.assert_allclose(half_width, [10.0, 5.0, 10.0 / 3.0])
    assert np.all(half_width > 0.0)


def test_requested_plot_labels_are_exact() -> None:
    assert script.PLOT_TITLE == "Mean Predicted Profit Per Customer vs. Price Change"
    assert script.X_AXIS_LABEL == "Price Change"
    assert script.Y_AXIS_LABEL == "Mean Predicted Profit Per Customer"


def test_load_cloud_data_rejects_sample_mismatch(tmp_path) -> None:
    curve_csv, curve_manifest, diagnostics = _write_inputs(tmp_path)
    manifest = json.loads(curve_manifest.read_text(encoding="utf-8"))
    manifest["sample"]["row_indices_sha256"] = "wrong"
    curve_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="different customer samples"):
        script._load_cloud_data(
            curve_csv,
            curve_manifest,
            diagnostics,
            n_neighbors=500,
            n_jobs=1,
        )


def test_run_analysis_writes_pdf_csv_and_manifest(tmp_path) -> None:
    curve_csv, curve_manifest, diagnostics = _write_inputs(tmp_path)
    output_dir = tmp_path / "output"
    args = script._build_parser().parse_args(
        [
            "--curve-csv",
            str(curve_csv),
            "--curve-manifest",
            str(curve_manifest),
            "--diagnostics",
            str(diagnostics),
            "--output-dir",
            str(output_dir),
        ]
    )

    median_support = np.linspace(1.0, 10.0, script.SUPPORT_GRID.size)
    original = script._compute_median_local_support
    script._compute_median_local_support = lambda *args, **kwargs: median_support
    try:
        outputs = script.run_analysis(args)
    finally:
        script._compute_median_local_support = original

    assert len(outputs) == 3
    assert outputs[0].read_bytes().startswith(b"%PDF-")
    output_frame = pd.read_csv(outputs[1])
    assert len(output_frame) == 301
    assert np.all(output_frame["smoothed_support_half_width"] > 0.0)
    manifest = json.loads(outputs[2].read_text(encoding="utf-8"))
    assert manifest["model"]["objective"] == "ModelBasedObjective"
    assert manifest["model"]["acceptance"] == "monotone_spline_xgb"
    assert manifest["model"]["loss"] == "xgb"
    assert manifest["support"]["grid"] == "u=-0.100,...,0.200"
    assert manifest["support"]["baseline_subtracted"] is False
    assert manifest["optimizer"].startswith("not used")
