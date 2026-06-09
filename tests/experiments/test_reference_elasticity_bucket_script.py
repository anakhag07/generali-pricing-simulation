"""Tests for the GLM reference-elasticity bucket experiment script."""

from __future__ import annotations

import numpy as np


def test_reference_elasticity_script_defaults_to_first_order_refs() -> None:
    import scripts.run_glm_reference_elasticity_bucket_experiment as script

    assert script.REFERENCE_U_VALUES == (-0.1, 0.1, 0.2, 0.3)
    assert script.RUN_OVERRIDES["enabled_estimators"] == ("first_order",)
    assert script.RUN_OVERRIDES["plot"] is True
    assert "n_samples" not in script.RUN_OVERRIDES


def test_u_label_is_filesystem_safe() -> None:
    import scripts.run_glm_reference_elasticity_bucket_experiment as script

    assert script._u_label(-0.1) == "u_ref_minus0p1"
    assert script._u_label(0.1) == "u_ref_plus0p1"
    assert script._u_label(0.3) == "u_ref_plus0p3"


def test_reference_bucket_rows_are_sorted_by_bucket_rank() -> None:
    import scripts.run_glm_reference_elasticity_bucket_experiment as script

    rows = [
        {"u_ref": 0.1, "bucket_rank": 2, "bucket": "high"},
        {"u_ref": -0.1, "bucket_rank": 0, "bucket": "low"},
        {"u_ref": 0.1, "bucket_rank": 0, "bucket": "low"},
        {"u_ref": 0.1, "bucket_rank": 1, "bucket": "medium"},
    ]

    selected = script._rows_for_u_ref(rows, 0.1)

    assert [row["bucket"] for row in selected] == ["low", "medium", "high"]


def test_reference_bucket_plots_include_expected_outputs(tmp_path) -> None:
    import scripts.run_glm_reference_elasticity_bucket_experiment as script
    from experiments.sensitivity_buckets import SensitivityBucket

    rows = [
        {
            "u_ref": 0.1,
            "bucket": "low",
            "bucket_rank": 0,
            "u": 0.2,
            "mean_acceptance": 0.9,
            "value": -3.0,
            "elasticity_abs_mean": 0.1,
        },
        {
            "u_ref": 0.1,
            "bucket": "medium",
            "bucket_rank": 1,
            "u": 0.1,
            "mean_acceptance": 0.88,
            "value": -2.0,
            "elasticity_abs_mean": 0.2,
        },
        {
            "u_ref": 0.1,
            "bucket": "high",
            "bucket_rank": 2,
            "u": -0.1,
            "mean_acceptance": 0.86,
            "value": -1.0,
            "elasticity_abs_mean": 0.3,
        },
    ]
    buckets = tuple(
        SensitivityBucket(
            name=name,
            row_indices=np.arange(3, dtype=int),
            scores=np.asarray(scores, dtype=float),
        )
        for name, scores in [
            ("low", [0.05, 0.10, 0.15]),
            ("medium", [0.15, 0.20, 0.25]),
            ("high", [0.25, 0.30, 0.35]),
        ]
    )

    script._write_plots(rows, buckets, tmp_path, u_ref=0.1)

    assert (tmp_path / "bucket_vs_u_acceptance.png").exists()
    assert (tmp_path / "bucket_objective_acceptance.png").exists()
    assert (tmp_path / "bucket_u_acceptance.png").exists()
    assert (tmp_path / "elasticity_score_histograms.png").exists()
