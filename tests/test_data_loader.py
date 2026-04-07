"""Tests for src/data/loader.py."""

import numpy as np
import pytest


def test_load_x_array_glm_shape():
    from data.loader import load_x_array, FEATURE_COLS_GLM
    x = load_x_array("glm", n_rows=50)
    assert x.shape == (50, len(FEATURE_COLS_GLM))
    assert x.dtype == np.float64


def test_load_x_array_xgb_shape():
    from data.loader import load_x_array, FEATURE_COLS_XGB
    x = load_x_array("xgb", n_rows=50)
    assert x.shape == (50, len(FEATURE_COLS_XGB))
    assert x.dtype == np.float64


def test_load_x_array_glm_has_more_cols_than_xgb():
    from data.loader import load_x_array
    glm_x = load_x_array("glm", n_rows=10)
    xgb_x = load_x_array("xgb", n_rows=10)
    assert glm_x.shape[1] > xgb_x.shape[1]


def test_load_model_artifacts_types():
    import sklearn.pipeline
    import xgboost

    from data.loader import load_model_artifacts

    glm_acc, glm_loss = load_model_artifacts("glm")
    assert isinstance(glm_acc, sklearn.pipeline.Pipeline)
    assert hasattr(glm_loss, "predict")  # LinearRegression

    xgb_acc, xgb_loss = load_model_artifacts("xgb")
    assert isinstance(xgb_acc, xgboost.XGBClassifier)
    assert isinstance(xgb_loss, xgboost.XGBRegressor)


def test_extract_glm_u_coef_is_finite():
    from data.loader import extract_glm_u_coef, load_model_artifacts

    glm_acc, _ = load_model_artifacts("glm")
    coef = extract_glm_u_coef(glm_acc)
    assert np.isfinite(coef)
    assert coef != 0.0


def test_extract_glm_churn_coefficients_matches_u_coef():
    from data.loader import (
        ACCEPTANCE_STATE_COLS,
        extract_glm_churn_coefficients,
        extract_glm_u_coef,
        load_model_artifacts,
    )

    glm_acc, _ = load_model_artifacts("glm")
    coeffs = extract_glm_churn_coefficients(glm_acc)

    assert coeffs["x_feature_names"] == list(ACCEPTANCE_STATE_COLS)
    assert len(coeffs["x_coef"]) == len(ACCEPTANCE_STATE_COLS)
    assert np.isfinite(coeffs["intercept"])
    assert coeffs["u_coef"] == pytest.approx(extract_glm_u_coef(glm_acc))


def test_extract_linear_loss_coefficients_has_expected_features():
    from data.loader import LOSS_FEATURE_COLS, extract_linear_loss_coefficients, load_model_artifacts

    _, glm_loss = load_model_artifacts("glm")
    coeffs = extract_linear_loss_coefficients(glm_loss)

    assert coeffs["x_feature_names"] == list(LOSS_FEATURE_COLS)
    assert len(coeffs["x_coef"]) == len(LOSS_FEATURE_COLS)
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
