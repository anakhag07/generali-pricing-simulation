"""Tests for canonical dataset metadata."""

import pandas as pd


def test_dataset_metadata_matches_canonical_csv_columns():
    from data.dataset_metadata import DATASET_PATH, REQUIRED_DATASET_COLUMNS

    header = pd.read_csv(DATASET_PATH, sep=";", nrows=0)
    missing = [col for col in REQUIRED_DATASET_COLUMNS if col not in header.columns]

    assert missing == []


def test_model_artifact_metadata_paths_are_configured():
    from data.dataset_metadata import MODEL_ARTIFACTS

    assert set(MODEL_ARTIFACTS) == {"glm", "xgb"}
    for spec in MODEL_ARTIFACTS.values():
        assert spec["acceptance"]["path"].suffix == ".pkl"
        assert spec["loss"]["path"].suffix == ".pkl"
        assert spec["acceptance"]["contains_feature_processor"] is True
        assert spec["loss"]["contains_feature_processor"] is True
