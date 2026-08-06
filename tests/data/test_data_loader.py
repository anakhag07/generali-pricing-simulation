"""Tests for src/data/loader.py."""

import numpy as np
import pytest


def test_load_x_frame_glm_shape_and_columns():
    from data.loader import FEATURE_COLS_GLM, load_x_frame
    from data.dataset_metadata import LOOKAHEAD_X_COLS

    x = load_x_frame("linear", n_rows=50, seed=123)

    assert x.shape == (50, len(FEATURE_COLS_GLM))
    assert list(x.columns) == FEATURE_COLS_GLM
    assert not set(LOOKAHEAD_X_COLS).intersection(x.columns)


def test_load_x_frame_xgb_shape():
    from data.loader import FEATURE_COLS_XGB, load_x_frame

    x = load_x_frame("xgb", n_rows=50, seed=123)

    assert x.shape == (50, len(FEATURE_COLS_XGB))


def test_load_monotone_spline_frame_includes_policy_id() -> None:
    from data.loader import FEATURE_COLS_XGB, load_x_frame

    x = load_x_frame("monotone_spline_xgb", n_rows=25, seed=123)

    assert x.shape == (25, len(FEATURE_COLS_XGB) + 1)
    assert list(x.columns) == ["id", *FEATURE_COLS_XGB]
    assert x["id"].nunique() == 25


def test_dataset_column_roles_report_used_and_unused_x():
    from data.loader import dataset_column_roles

    roles = dataset_column_roles()

    assert "X_policy_premium" in roles["used_x_cols"]
    assert "X_upcoming_premium" in roles["lookahead_x_cols"]
    assert "X_upcoming_premium" in roles["unused_x_cols"]
    assert "U" in roles["objective_excluded_cols"]
    assert "Y_G_Loss" in roles["objective_excluded_cols"]
    assert "is_churn" in roles["objective_excluded_cols"]


def test_sample_csv_row_indices_is_deterministic():
    from data.loader import sample_csv_row_indices

    idx_1 = sample_csv_row_indices("linear", n_rows=25, seed=123)
    idx_2 = sample_csv_row_indices("linear", n_rows=25, seed=123)

    assert np.array_equal(idx_1, idx_2)
    assert idx_1.shape == (25,)
    assert np.unique(idx_1).shape == (25,)


def test_sample_csv_row_indices_changes_with_seed():
    from data.loader import sample_csv_row_indices

    idx_1 = sample_csv_row_indices("linear", n_rows=25, seed=123)
    idx_2 = sample_csv_row_indices("linear", n_rows=25, seed=456)

    assert not np.array_equal(idx_1, idx_2)


def test_load_x_array_row_indices_are_ordered_and_reusable():
    from data.loader import load_x_frame, sample_csv_row_indices

    row_indices = sample_csv_row_indices("linear", n_rows=20, seed=123)
    x_1 = load_x_frame("linear", row_indices=row_indices)
    x_2 = load_x_frame("linear", row_indices=row_indices)

    assert x_1.equals(x_2)


def test_load_observed_u_array_matches_requested_rows():
    from data.loader import _load_observed_u_array

    glm_u = _load_observed_u_array("linear", n_rows=25, seed=123)
    xgb_u = _load_observed_u_array("xgb", n_rows=25, seed=123)

    assert glm_u.shape == (25,)
    assert xgb_u.shape == (25,)
    assert np.all(np.isfinite(glm_u))
    assert np.all(np.isfinite(xgb_u))


def test_load_observed_u_array_uses_row_indices():
    from data.loader import load_observed_u_array, sample_csv_row_indices

    row_indices = sample_csv_row_indices("linear", n_rows=25, seed=123)
    u_1 = load_observed_u_array("linear", row_indices=row_indices)
    u_2 = load_observed_u_array("linear", row_indices=row_indices)

    assert np.array_equal(u_1, u_2)
    assert u_1.shape == (25,)


def test_load_observed_loss_array_uses_row_indices():
    from data.loader import load_observed_loss_array, sample_csv_row_indices

    row_indices = sample_csv_row_indices("linear", n_rows=25, seed=123)
    loss_1 = load_observed_loss_array("linear", row_indices=row_indices)
    loss_2 = load_observed_loss_array("linear", row_indices=row_indices)

    assert np.array_equal(loss_1, loss_2)
    assert loss_1.shape == (25,)
    assert np.all(np.isfinite(loss_1))


def test_load_model_artifacts_types():
    import sklearn.linear_model
    import xgboost

    from data.loader import ModelArtifactBundle, load_model_artifacts, unwrap_model_artifact

    glm_acc, glm_loss = load_model_artifacts("linear")
    assert isinstance(glm_acc, ModelArtifactBundle)
    assert isinstance(glm_loss, ModelArtifactBundle)
    assert glm_acc.preprocessor is not None
    assert glm_loss.preprocessor is not None
    assert glm_acc.source_format == "cv_selected_fold"
    assert glm_acc.probability_target == "acceptance"
    assert isinstance(unwrap_model_artifact(glm_acc), sklearn.linear_model.LogisticRegression)
    assert hasattr(unwrap_model_artifact(glm_loss), "predict")

    xgb_acc, xgb_loss = load_model_artifacts("xgb")
    assert isinstance(xgb_acc, ModelArtifactBundle)
    assert isinstance(xgb_loss, ModelArtifactBundle)
    assert isinstance(unwrap_model_artifact(xgb_acc), xgboost.XGBClassifier)
    assert isinstance(unwrap_model_artifact(xgb_loss), xgboost.XGBRegressor)

    spline_acc, spline_loss = load_model_artifacts("monotone_spline_xgb")
    assert spline_acc.model_type == "monotone_spline_xgb"
    assert spline_acc.probability_target == "acceptance"
    assert spline_acc.preprocessor is not None
    assert isinstance(unwrap_model_artifact(spline_loss), xgboost.XGBRegressor)


def test_selected_best_fold_artifacts_normalize_nested_preprocessor() -> None:
    from data.loader import _normalize_artifact

    marker = object()
    model = object()
    artifact = _normalize_artifact(
        {
            "model": model,
            "preprocessor": {
                "preprocessor": marker,
                "x_feature_cols": ["x1", "x2"],
                "u_cols": ["U"],
            },
            "best_fold": 3,
            "model_features": ["x1", "x2", "U"],
            "target": "acceptance",
        }
    )

    assert artifact.model is model
    assert artifact.preprocessor is marker
    assert artifact.x_feature_cols == ("x1", "x2")
    assert artifact.u_cols == ("U",)
    assert artifact.source_format == "selected_best_fold"
    assert artifact.probability_target == "acceptance"


def test_new_versioned_artifacts_load_and_predict() -> None:
    import xgboost

    from data.loader import (
        load_model_artifact_pair,
        load_x_frame,
        unwrap_model_artifact,
    )

    acceptance, loss = load_model_artifact_pair("xgb", "xgb")
    x = load_x_frame("xgb", n_rows=5, seed=123)
    acceptance_frame = x.copy()
    acceptance_frame["U"] = 0.08

    acceptance_prediction = acceptance.model.predict_proba(
        acceptance.model_frame(acceptance_frame)
    )
    loss_prediction = loss.model.predict(loss.model_frame(x))

    assert acceptance.source_format == "cv_selected_fold"
    assert loss.source_format == "cv_selected_fold"
    assert isinstance(unwrap_model_artifact(acceptance), xgboost.XGBClassifier)
    assert isinstance(unwrap_model_artifact(loss), xgboost.XGBRegressor)
    assert np.isfinite(acceptance_prediction).all()
    assert np.isfinite(loss_prediction).all()


def test_monotone_spline_uses_all_complete_rows_and_raw_fallback() -> None:
    from data.loader import (
        eligible_csv_row_indices,
        load_acceptance_artifact,
        load_x_frame,
    )

    acceptance = load_acceptance_artifact("monotone_spline_xgb")
    row_indices = eligible_csv_row_indices("monotone_spline_xgb")
    x = load_x_frame("monotone_spline_xgb", row_indices=row_indices[:250])

    assert acceptance.covered_row_indices().shape == (200,)
    assert row_indices.size > 200
    assert len(x) == 250
    assert not set(x["id"].astype(str)).issubset(set(acceptance.covered_policy_ids()))


def test_model_pair_resolution_matches_runtime_hierarchy() -> None:
    from data.loader import resolve_model_artifact_ids

    assert resolve_model_artifact_ids(model_type="linear") == ("linear", "linear")
    assert resolve_model_artifact_ids(model_type="xgb") == ("xgb", "xgb")
    assert resolve_model_artifact_ids(model_type="monotone_spline_xgb") == (
        "monotone_spline_xgb",
        "xgb",
    )


def test_extract_glm_u_coef_is_finite():
    from data.loader import extract_glm_u_coef, load_model_artifacts

    glm_acc, _ = load_model_artifacts("linear")
    coef = extract_glm_u_coef(glm_acc)
    assert np.isfinite(coef)
    assert coef != 0.0


def test_extract_glm_acceptance_coefficients_matches_u_coef():
    from data.loader import extract_glm_acceptance_coefficients, extract_glm_u_coef, load_model_artifacts

    glm_acc, _ = load_model_artifacts("linear")
    coeffs = extract_glm_acceptance_coefficients(glm_acc)

    assert len(coeffs["x_feature_names"]) == len(coeffs["x_coef"])
    assert coeffs["x_feature_names"]
    assert all(name != "U" for name in coeffs["x_feature_names"])
    assert np.isfinite(coeffs["intercept"])
    assert coeffs["probability_target"] == "acceptance"
    assert coeffs["u_coef"] == pytest.approx(extract_glm_u_coef(glm_acc))


def test_extract_linear_loss_coefficients_has_expected_features():
    from data.loader import extract_linear_loss_coefficients, load_model_artifacts, unwrap_model_artifact

    _, glm_loss = load_model_artifacts("linear")
    coeffs = extract_linear_loss_coefficients(glm_loss)

    assert coeffs["x_feature_names"] == list(unwrap_model_artifact(glm_loss).feature_names_in_)
    assert len(coeffs["x_coef"]) == len(coeffs["x_feature_names"])
    assert np.isfinite(coeffs["intercept"])


def test_extract_model_based_coefficients_glm_and_xgb_support():
    from data.loader import extract_model_based_coefficients, load_model_artifacts

    glm_acc, glm_loss = load_model_artifacts("linear")
    coeffs = extract_model_based_coefficients(glm_acc, glm_loss)
    assert coeffs is not None
    assert set(coeffs) == {"acceptance", "loss"}

    xgb_acc, xgb_loss = load_model_artifacts("xgb")
    assert extract_model_based_coefficients(xgb_acc, xgb_loss) is None


def test_load_x_array_invalid_type():
    from data.loader import load_x_array

    with pytest.raises(ValueError):
        load_x_array("invalid")
