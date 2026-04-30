"""Tests for src/data/loader.py."""

import numpy as np
import pytest


def test_load_x_array_glm_shape():
    from data.loader import load_x_array, FEATURE_COLS_GLM
    x = load_x_array("glm", n_rows=50, seed=123)
    assert x.shape == (50, len(FEATURE_COLS_GLM))
    assert x.dtype == np.float64


def test_load_x_array_xgb_shape():
    from data.loader import load_x_array, FEATURE_COLS_XGB
    x = load_x_array("xgb", n_rows=50, seed=123)
    assert x.shape == (50, len(FEATURE_COLS_XGB))
    assert x.dtype == np.float64


def test_load_x_array_glm_has_more_cols_than_xgb():
    from data.loader import load_x_array
    glm_x = load_x_array("glm", n_rows=10, seed=123)
    xgb_x = load_x_array("xgb", n_rows=10, seed=123)
    assert glm_x.shape[1] > xgb_x.shape[1]


def test_sample_csv_row_indices_is_deterministic():
    from data.loader import sample_csv_row_indices

    idx_1 = sample_csv_row_indices("glm", n_rows=25, seed=123)
    idx_2 = sample_csv_row_indices("glm", n_rows=25, seed=123)

    assert np.array_equal(idx_1, idx_2)
    assert idx_1.shape == (25,)
    assert np.unique(idx_1).shape == (25,)


def test_sample_csv_row_indices_changes_with_seed():
    from data.loader import sample_csv_row_indices

    idx_1 = sample_csv_row_indices("glm", n_rows=25, seed=123)
    idx_2 = sample_csv_row_indices("glm", n_rows=25, seed=456)

    assert not np.array_equal(idx_1, idx_2)


def test_load_x_array_row_indices_are_ordered_and_reusable():
    from data.loader import load_x_array, sample_csv_row_indices

    row_indices = sample_csv_row_indices("glm", n_rows=20, seed=123)
    x_1 = load_x_array("glm", row_indices=row_indices)
    x_2 = load_x_array("glm", row_indices=row_indices)

    assert np.array_equal(x_1, x_2)


def test_load_observed_u_array_matches_requested_rows():
    from data.loader import _load_observed_u_array

    glm_u = _load_observed_u_array("glm", n_rows=25, seed=123)
    xgb_u = _load_observed_u_array("xgb", n_rows=25, seed=123)

    assert glm_u.shape == (25,)
    assert xgb_u.shape == (25,)
    assert np.all(np.isfinite(glm_u))
    assert np.all(np.isfinite(xgb_u))


def test_load_observed_u_array_uses_row_indices():
    from data.loader import load_observed_u_array, sample_csv_row_indices

    row_indices = sample_csv_row_indices("glm", n_rows=25, seed=123)
    u_1 = load_observed_u_array("glm", row_indices=row_indices)
    u_2 = load_observed_u_array("glm", row_indices=row_indices)

    assert np.array_equal(u_1, u_2)
    assert u_1.shape == (25,)


def test_load_model_artifacts_types():
    import sklearn.pipeline
    import xgboost

    from data.loader import ModelArtifactBundle, load_model_artifacts, unwrap_model_artifact

    glm_acc, glm_loss = load_model_artifacts("glm")
    assert isinstance(glm_acc, ModelArtifactBundle)
    assert isinstance(glm_loss, ModelArtifactBundle)
    assert glm_acc.preprocessor is not None
    assert glm_loss.preprocessor is not None
    assert isinstance(unwrap_model_artifact(glm_acc), sklearn.pipeline.Pipeline)
    assert hasattr(unwrap_model_artifact(glm_loss), "predict")

    xgb_acc, xgb_loss = load_model_artifacts("xgb")
    assert isinstance(xgb_acc, ModelArtifactBundle)
    assert isinstance(xgb_loss, ModelArtifactBundle)
    assert isinstance(unwrap_model_artifact(xgb_acc), xgboost.XGBClassifier)
    assert isinstance(unwrap_model_artifact(xgb_loss), xgboost.XGBRegressor)


def test_extract_glm_u_coef_is_finite():
    from data.loader import extract_glm_u_coef, load_model_artifacts

    glm_acc, _ = load_model_artifacts("glm")
    coef = extract_glm_u_coef(glm_acc)
    assert np.isfinite(coef)
    assert coef != 0.0


def test_extract_glm_churn_coefficients_matches_u_coef():
    from data.loader import extract_glm_churn_coefficients, extract_glm_u_coef, load_model_artifacts

    glm_acc, _ = load_model_artifacts("glm")
    coeffs = extract_glm_churn_coefficients(glm_acc)

    assert len(coeffs["x_feature_names"]) == len(coeffs["x_coef"])
    assert coeffs["x_feature_names"]
    assert all(name != "U" for name in coeffs["x_feature_names"])
    assert np.isfinite(coeffs["intercept"])
    assert coeffs["u_coef"] == pytest.approx(extract_glm_u_coef(glm_acc))


def test_extract_linear_loss_coefficients_has_expected_features():
    from data.loader import extract_linear_loss_coefficients, load_model_artifacts, unwrap_model_artifact

    _, glm_loss = load_model_artifacts("glm")
    coeffs = extract_linear_loss_coefficients(glm_loss)

    assert coeffs["x_feature_names"] == list(unwrap_model_artifact(glm_loss).feature_names_in_)
    assert len(coeffs["x_coef"]) == len(coeffs["x_feature_names"])
    assert np.isfinite(coeffs["intercept"])


def test_extract_model_based_coefficients_glm_and_xgb_support():
    from data.loader import extract_model_based_coefficients, load_model_artifacts

    glm_acc, glm_loss = load_model_artifacts("glm")
    coeffs = extract_model_based_coefficients(glm_acc, glm_loss)
    assert coeffs is not None
    assert set(coeffs) == {"churn", "loss"}

    xgb_acc, xgb_loss = load_model_artifacts("xgb")
    assert extract_model_based_coefficients(xgb_acc, xgb_loss) is None


def test_load_x_array_invalid_type():
    from data.loader import load_x_array

    with pytest.raises(ValueError):
        load_x_array("invalid")
