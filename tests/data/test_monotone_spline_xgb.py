"""Behavior tests for the monotone-spline wrapper's public interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import PchipInterpolator

from data.monotone_spline_xgb import (
    MonotoneSplineArtifactData,
    MonotoneSplineXGBAcceptance,
    load_monotone_spline_artifact,
    save_monotone_spline_artifact,
)


class _RawModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        acceptance = np.clip(0.9 - frame["U"].to_numpy(float), 0.0, 1.0)
        return np.column_stack([1.0 - acceptance, acceptance])


class _BaseAcceptance:
    model = _RawModel()
    preprocessor = None
    x_feature_cols = ("x",)

    def model_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[:, ["x", "U"]]

    def policy_feature_dim(self) -> int:
        return 1


def _artifact() -> MonotoneSplineArtifactData:
    grid = np.asarray([0.0, 0.04, 0.08, 0.12, 0.16])
    values = np.asarray([[0.08, 0.10, 0.13, 0.17, 0.22], [0.15, 0.17, 0.20, 0.24, 0.29]])
    curves = [PchipInterpolator(grid, row) for row in values]
    return MonotoneSplineArtifactData(
        policy_ids=np.asarray(["101", "202"]),
        row_indices=np.asarray([3, 7]),
        action_grid=grid,
        coefficients=np.stack([curve.c for curve in curves]),
        churn_min=values[:, 0],
        churn_max=values[:, -1],
        upper_slopes=np.asarray([curve.derivative()(grid[-1]) for curve in curves]),
        base_artifact_sha256="base",
        source_fold=0,
    )


def test_array_only_round_trip(tmp_path: Path) -> None:
    path = save_monotone_spline_artifact(_artifact(), tmp_path / "curves.npz")
    loaded = load_monotone_spline_artifact(path)
    np.testing.assert_array_equal(loaded.policy_ids, ["101", "202"])
    np.testing.assert_allclose(loaded.coefficients, _artifact().coefficients)
    assert loaded.source_fold == 0
    with np.load(path, allow_pickle=False) as raw:
        assert all(raw[name].dtype != object for name in raw.files)


def test_cached_acceptance_is_bounded_monotone_and_has_derivative() -> None:
    model = MonotoneSplineXGBAcceptance(_artifact(), _BaseAcceptance())
    frame = pd.DataFrame({"id": ["101", "202"], "x": [1.0, 2.0]})
    actions = np.asarray([0.05, 0.11])
    eps = 1e-6
    acceptance = model.predict_acceptance(frame, actions)
    derivative = model.d_acceptance_du(frame, actions)
    finite_difference = (
        model.predict_acceptance(frame, actions + eps)
        - model.predict_acceptance(frame, actions - eps)
    ) / (2.0 * eps)
    assert np.all((acceptance >= 0.0) & (acceptance <= 1.0))
    assert np.all(derivative <= 0.0)
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-7, atol=1e-8)


def test_unknown_policy_falls_back_to_base_xgb() -> None:
    model = MonotoneSplineXGBAcceptance(_artifact(), _BaseAcceptance())
    frame = pd.DataFrame({"id": ["999"], "x": [1.0]})
    np.testing.assert_allclose(model.predict_acceptance(frame, np.asarray([0.08])), [0.82])
    np.testing.assert_allclose(model.d_acceptance_du(frame, np.asarray([0.08])), [-1.0])


def test_wrapper_rejects_curve_cache_from_another_xgb(tmp_path: Path) -> None:
    base = _BaseAcceptance()
    base.artifact_path = tmp_path / "acceptance.pkl"
    base.artifact_path.write_bytes(b"different-base")
    with pytest.raises(ValueError, match="not derived from the configured XGB"):
        MonotoneSplineXGBAcceptance(_artifact(), base)


def test_artifact_rejects_nonmonotone_or_out_of_bounds_curves() -> None:
    artifact = _artifact()
    invalid = artifact.coefficients.copy()
    invalid[0, -1, 0] = 1.5
    with pytest.raises(ValueError, match="probability range|not monotone"):
        MonotoneSplineArtifactData(
            policy_ids=artifact.policy_ids,
            row_indices=artifact.row_indices,
            action_grid=artifact.action_grid,
            coefficients=invalid,
            churn_min=artifact.churn_min,
            churn_max=artifact.churn_max,
            upper_slopes=artifact.upper_slopes,
        )
