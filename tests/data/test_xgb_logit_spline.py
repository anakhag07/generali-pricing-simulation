"""Tests for portable XGBoost logit-spline artifact preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.xgb_logit_spline import (
    canonical_row_indices_for_policy_ids,
    fit_logit_spline_artifact,
    load_xgb_logit_spline_artifact,
    save_xgb_logit_spline_artifact,
    source_churn_grid,
)


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
