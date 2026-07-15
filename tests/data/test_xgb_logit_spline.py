"""Tests for portable XGBoost logit-spline artifact preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.xgb_logit_spline import (
    XGBLogitSplineAcceptance,
    canonical_row_indices_for_policy_ids,
    fit_logit_spline_artifact,
    load_xgb_logit_spline_artifact,
    save_xgb_logit_spline_artifact,
    source_churn_grid,
)
from scipy.interpolate import BSpline


class _UAcceptanceWrapper:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        acceptance = 0.9 - 0.5 * frame["U"].to_numpy(dtype=float)
        return np.column_stack([acceptance, 1.0 - acceptance])


def _artifact_data():
    action_grid = np.linspace(0.0, 0.16, 17)
    churn = np.vstack(
        [
            0.08 + 0.4 * action_grid,
            0.12 + 0.6 * action_grid,
        ]
    )
    return fit_logit_spline_artifact(
        policy_ids=["101", "202"],
        row_indices=[3, 7],
        action_grid=action_grid,
        churn_grid=churn,
        weights=np.ones(action_grid.size),
        source_sha256="abc123",
    )


def test_source_churn_grid_overwrites_historical_u() -> None:
    profiles = {
        "101": pd.DataFrame({"id": ["101"], "x": [1.0], "U": [0.13]}),
        "202": pd.DataFrame({"id": ["202"], "x": [2.0], "U": [0.07]}),
    }
    artifact = {
        "profiles": profiles,
        "MAX_PI_FIT": 2,
        "model_features": ["x", "U"],
        "smoothing_wrapper": _UAcceptanceWrapper(),
    }

    policy_ids, _, action_grid, churn = source_churn_grid(artifact)

    np.testing.assert_array_equal(policy_ids, np.asarray(["101", "202"]))
    np.testing.assert_allclose(action_grid, [0.0, 0.01, 0.02])
    np.testing.assert_allclose(churn[0], [0.1, 0.105, 0.11])
    np.testing.assert_allclose(churn[1], [0.1, 0.105, 0.11])


def test_fit_and_portable_round_trip_are_deterministic(tmp_path) -> None:
    first = _artifact_data()
    second = _artifact_data()
    np.testing.assert_allclose(first.knots, second.knots)
    np.testing.assert_allclose(first.coefficients, second.coefficients)

    output = tmp_path / "curves.npz"
    save_xgb_logit_spline_artifact(first, output)
    loaded = load_xgb_logit_spline_artifact(output)

    np.testing.assert_array_equal(loaded.policy_ids, first.policy_ids)
    np.testing.assert_array_equal(loaded.row_indices, first.row_indices)
    np.testing.assert_allclose(loaded.action_grid, first.action_grid)
    np.testing.assert_allclose(loaded.knots, first.knots)
    np.testing.assert_allclose(loaded.coefficients, first.coefficients)
    np.testing.assert_allclose(loaded.upper_slopes, first.upper_slopes)
    assert loaded.source_sha256 == "abc123"


def test_runtime_acceptance_derivative_matches_finite_difference() -> None:
    model = XGBLogitSplineAcceptance(_artifact_data())
    frame = pd.DataFrame({"id": ["101", "202"]})
    u = np.asarray([0.05, 0.12])
    eps = 1e-6

    derivative = model.d_acceptance_du(frame, u)
    derivative_fd = (
        model.predict_acceptance(frame, u + eps) - model.predict_acceptance(frame, u - eps)
    ) / (2.0 * eps)

    np.testing.assert_allclose(derivative, derivative_fd, rtol=1e-6, atol=1e-8)
    assert np.all(derivative < 0.0)


def test_runtime_boundary_rules_match_artifact_contract() -> None:
    artifact = _artifact_data()
    model = XGBLogitSplineAcceptance(artifact)
    frame = pd.DataFrame({"id": ["101", "101", "101"]})
    u = np.asarray([-0.1, 0.08, 0.18])

    acceptance = model.predict_acceptance(frame, u)
    derivative = model.d_acceptance_du(frame, u)

    assert acceptance[0] == pytest.approx(1.0 - artifact.churn_min[0])
    assert derivative[0] == 0.0
    assert derivative[1] < 0.0
    assert derivative[2] == pytest.approx(-artifact.upper_slopes[0])


def test_runtime_values_match_stored_bspline() -> None:
    artifact = _artifact_data()
    model = XGBLogitSplineAcceptance(artifact)
    u = np.asarray([0.03, 0.11])
    frame = pd.DataFrame({"id": ["101", "202"]})

    expected = []
    for index, u_value in enumerate(u):
        spline = BSpline(
            artifact.knots[index],
            artifact.coefficients[index],
            int(artifact.degrees[index]),
        )
        expected.append(1.0 / (1.0 + np.exp(float(spline(u_value)))))

    np.testing.assert_allclose(model.predict_acceptance(frame, u), expected)


def test_runtime_rejects_uncovered_policy_ids() -> None:
    model = XGBLogitSplineAcceptance(_artifact_data())
    with pytest.raises(ValueError, match="No fitted acceptance spline"):
        model.predict_acceptance(pd.DataFrame({"id": ["999"]}), np.asarray([0.1]))


def test_save_refuses_to_overwrite_by_default(tmp_path) -> None:
    output = tmp_path / "curves.npz"
    save_xgb_logit_spline_artifact(_artifact_data(), output)
    with pytest.raises(FileExistsError):
        save_xgb_logit_spline_artifact(_artifact_data(), output)


def test_fit_rejects_misaligned_weights() -> None:
    action_grid = np.linspace(0.0, 0.16, 17)
    with pytest.raises(ValueError, match="weights"):
        fit_logit_spline_artifact(
            policy_ids=["101"],
            row_indices=[3],
            action_grid=action_grid,
            churn_grid=np.full((1, 17), 0.1),
            weights=np.ones(16),
        )


def test_canonical_row_indices_preserve_requested_policy_order(tmp_path) -> None:
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame({"id": ["202", "999", "101"]}).to_csv(dataset, sep=";", index=False)

    row_indices = canonical_row_indices_for_policy_ids(dataset, ["101", "202"])

    np.testing.assert_array_equal(row_indices, [2, 0])
