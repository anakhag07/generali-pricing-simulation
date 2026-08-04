"""Contract tests for the matched 199-policy real-data hierarchy manifest."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments.configs import get_config
from experiments.manifest import load_experiment_manifest


MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "manifests"
    / "real_data_monotone_model_hierarchy_199.json"
)


def test_manifest_declares_four_requested_model_pairs_on_one_cohort() -> None:
    manifest = load_experiment_manifest(MANIFEST)

    assert manifest.base_preset == "real_data_glm_glm_20260728_base"
    assert len(manifest.variants) == 4
    assert manifest.seeds.run_seeds == (20260728,)
    assert manifest.seeds.vary == ("optimizer",)
    assert {
        (
            variant.overrides["acceptance_model_type"],
            variant.overrides["loss_model_type"],
        )
        for variant in manifest.variants
    } == {
        ("glm_20260527", "glm_20260527"),
        ("xgb_monotone_spline_20260728", "glm_20260527"),
        ("xgb_20260728", "glm_20260527"),
        ("xgb_20260728", "xgb_20260728"),
    }
    assert all(
        variant.overrides["row_cohort_model_type"] == "xgb_monotone_spline_20260728"
        for variant in manifest.variants
    )


def test_manifest_variants_build_on_identical_199_rows() -> None:
    manifest = load_experiment_manifest(MANIFEST)
    configs = [
        get_config(manifest.base_preset, overrides=variant.overrides)
        for variant in manifest.variants
    ]

    expected = configs[0].x_fixed_row_indices
    assert expected.shape == (199,)
    for config in configs:
        np.testing.assert_array_equal(config.x_fixed_row_indices, expected)
        assert config.enabled_estimators == ("finite_difference",)
        assert config.objective.u_bounds == (0.0, 0.16)
        assert config.objective.policy.action_low == 0.0
        assert config.objective.policy.action_high == 0.16
