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


def test_load_x_array_invalid_type():
    from data.loader import load_x_array

    with pytest.raises(ValueError):
        load_x_array("invalid")
