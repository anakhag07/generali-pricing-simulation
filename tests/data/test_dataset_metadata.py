"""Tests for canonical dataset metadata."""

import pandas as pd


def test_dataset_metadata_matches_canonical_csv_columns():
    from data.dataset_metadata import DATASET_PATH, REQUIRED_DATASET_COLUMNS

    header = pd.read_csv(DATASET_PATH, sep=";", nrows=0)
    missing = [col for col in REQUIRED_DATASET_COLUMNS if col not in header.columns]

    assert missing == []


def test_model_artifact_metadata_paths_are_configured():
    from data.dataset_metadata import MODEL_ARTIFACTS

    assert set(MODEL_ARTIFACTS) == {"linear", "xgb", "monotone_spline_xgb"}
    for model_type, spec in MODEL_ARTIFACTS.items():
        expected_suffix = ".npz" if model_type == "monotone_spline_xgb" else ".pkl"
        assert spec["acceptance"]["path"].suffix == expected_suffix
        assert spec["loss"]["path"].suffix == ".pkl"
        assert spec["acceptance"]["contains_feature_processor"] is (
            model_type != "monotone_spline_xgb"
        )
        assert spec["loss"]["contains_feature_processor"] is True


def test_052726_objective_x_columns_exclude_lookahead_and_targets():
    from data.dataset_metadata import ACTION_COLS, LOOKAHEAD_X_COLS, TARGET_COLS, USED_X_COLS

    assert "X_policy_premium" in USED_X_COLS
    assert "X_upcoming_premium" in LOOKAHEAD_X_COLS
    assert "X_upcoming_premium" not in USED_X_COLS
    assert set(ACTION_COLS).isdisjoint(USED_X_COLS)
    assert set(TARGET_COLS).isdisjoint(USED_X_COLS)
