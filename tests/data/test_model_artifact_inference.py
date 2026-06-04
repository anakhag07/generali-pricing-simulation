"""Smoke tests for bundled model artifact inference on canonical data rows."""

import numpy as np
import pandas as pd


def _artifact_prediction_inputs(model_type: str, n_rows: int = 5):
    from data.loader import ACCEPTANCE_STATE_COLS, LOSS_FEATURE_COLS, load_model_artifacts, load_x_frame

    acceptance_model, loss_model = load_model_artifacts(model_type)
    x = load_x_frame(model_type, n_rows=n_rows, seed=123)
    acceptance_frame = x.loc[:, list(ACCEPTANCE_STATE_COLS)].copy()
    acceptance_frame["U"] = np.zeros(x.shape[0], dtype=float)
    loss_frame = x.loc[:, list(LOSS_FEATURE_COLS)].copy()
    return acceptance_model, loss_model, acceptance_frame, loss_frame


def test_glm_artifacts_predict_on_canonical_dataset_rows():
    _assert_artifacts_predict_on_canonical_rows("glm")


def test_xgb_artifacts_predict_on_canonical_dataset_rows():
    _assert_artifacts_predict_on_canonical_rows("xgb")


def test_model_based_objective_runs_on_canonical_dataset_rows():
    from data.loader import ACCEPTANCE_STATE_COLS, LOSS_FEATURE_COLS, PREMIUM_COL, extract_glm_u_coef, load_model_artifacts, load_x_frame
    from objective.objectives import ModelBasedObjective
    from objective.policy import ConstantPolicy

    theta = np.array([0.0], dtype=float)
    for model_type in ("glm", "xgb"):
        acceptance_model, loss_model = load_model_artifacts(model_type)
        u_coef = extract_glm_u_coef(acceptance_model) if model_type == "glm" else None
        objective = ModelBasedObjective(
            policy=ConstantPolicy(),
            acceptance_model=acceptance_model,
            loss_model=loss_model,
            acceptance_state_cols=tuple(ACCEPTANCE_STATE_COLS),
            loss_cols=tuple(LOSS_FEATURE_COLS),
            premium_col=PREMIUM_COL,
            u_coef=u_coef,
        )
        x = load_x_frame(model_type, n_rows=5, seed=123)

        value = objective.value(theta, x)
        value_at_u = objective.value_at_u(x, 0.0)
        mean_acceptance = objective.mean_acceptance(theta, x)

        assert np.isfinite(value)
        assert np.isfinite(value_at_u)
        assert 0.0 <= mean_acceptance <= 1.0


def _assert_artifacts_predict_on_canonical_rows(model_type: str):
    acceptance_model, loss_model, acceptance_frame, loss_frame = _artifact_prediction_inputs(model_type)

    acceptance_input = acceptance_model.model_frame(acceptance_frame)
    loss_input = loss_model.model_frame(loss_frame)
    acceptance = acceptance_model.model.predict_proba(acceptance_input)
    loss = loss_model.model.predict(loss_input)

    assert acceptance.shape == (len(acceptance_frame), 2)
    assert loss.shape == (len(loss_frame),)
    assert np.isfinite(acceptance).all()
    assert np.isfinite(loss).all()
    assert np.all((acceptance >= 0.0) & (acceptance <= 1.0))
