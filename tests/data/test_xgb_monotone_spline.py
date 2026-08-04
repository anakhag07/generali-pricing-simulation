"""Tests for portable policy-specific monotone PCHIP acceptance curves."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import PchipInterpolator

from data.xgb_monotone_spline import (
    XGBMonotoneSplineAcceptance,
    XGBMonotoneSplineArtifactData,
    load_legacy_monotone_wrapper,
    load_xgb_monotone_spline_artifact,
    save_xgb_monotone_spline_artifact,
)


def _artifact() -> XGBMonotoneSplineArtifactData:
    action_grid = np.asarray([0.0, 0.04, 0.08, 0.12, 0.16])
    churn_values = np.asarray(
        [
            [0.08, 0.10, 0.13, 0.17, 0.22],
            [0.15, 0.17, 0.20, 0.24, 0.29],
        ]
    )
    curves = [PchipInterpolator(action_grid, values) for values in churn_values]
    return XGBMonotoneSplineArtifactData(
        policy_ids=np.asarray(["101", "202"]),
        row_indices=np.asarray([3, 7]),
        action_grid=action_grid,
        coefficients=np.stack([curve.c for curve in curves]),
        churn_min=churn_values[:, 0],
        churn_max=churn_values[:, -1],
        upper_slopes=np.asarray([curve.derivative()(action_grid[-1]) for curve in curves]),
        source_sha256="source",
        embedded_booster_sha256="booster",
        base_artifact_sha256="base",
        base_preprocessor_sha256="preprocessor",
    )


def test_portable_round_trip_uses_array_only_npz(tmp_path: Path) -> None:
    path = save_xgb_monotone_spline_artifact(_artifact(), tmp_path / "curves.npz")
    loaded = load_xgb_monotone_spline_artifact(path)

    np.testing.assert_array_equal(loaded.policy_ids, ["101", "202"])
    np.testing.assert_array_equal(loaded.row_indices, [3, 7])
    np.testing.assert_allclose(loaded.action_grid, _artifact().action_grid)
    np.testing.assert_allclose(loaded.coefficients, _artifact().coefficients)
    assert loaded.source_sha256 == "source"
    with np.load(path, allow_pickle=False) as raw:
        assert all(raw[name].dtype != object for name in raw.files)


def test_acceptance_is_bounded_monotone_and_derivative_matches_fd() -> None:
    model = XGBMonotoneSplineAcceptance(_artifact())
    frame = pd.DataFrame({"id": ["101", "202"]})
    actions = np.asarray([0.05, 0.11])
    eps = 1e-6

    acceptance = model.predict_acceptance(frame, actions)
    analytical = model.d_acceptance_du(frame, actions)
    finite_difference = (
        model.predict_acceptance(frame, actions + eps)
        - model.predict_acceptance(frame, actions - eps)
    ) / (2.0 * eps)

    assert np.all((acceptance >= 0.0) & (acceptance <= 1.0))
    assert np.all(analytical <= 0.0)
    np.testing.assert_allclose(analytical, finite_difference, rtol=1e-7, atol=1e-8)


def test_boundary_rules_are_bounded_and_clipping_has_zero_derivative() -> None:
    model = XGBMonotoneSplineAcceptance(_artifact())
    frame = pd.DataFrame({"id": ["101", "101", "101", "101"]})
    actions = np.asarray([-0.1, 0.0, 0.16, 10.0])

    acceptance = model.predict_acceptance(frame, actions)
    derivative = model.d_acceptance_du(frame, actions)

    np.testing.assert_allclose(acceptance[:2], 1.0 - _artifact().churn_min[0])
    assert acceptance[2] == pytest.approx(1.0 - _artifact().churn_max[0])
    assert acceptance[3] == 0.0
    assert derivative[0] == 0.0
    assert derivative[3] == 0.0
    assert np.all((acceptance >= 0.0) & (acceptance <= 1.0))


def test_predict_proba_and_unknown_policy_contract() -> None:
    model = XGBMonotoneSplineAcceptance(_artifact())
    frame = pd.DataFrame({"id": ["101"], "U": [0.08]})
    probabilities = model.predict_proba(frame)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    with pytest.raises(ValueError, match="No fitted monotone spline"):
        model.predict_acceptance(pd.DataFrame({"id": ["999"]}), np.asarray([0.08]))


def test_artifact_rejects_nonmonotone_or_out_of_bounds_curves() -> None:
    artifact = _artifact()
    invalid = artifact.coefficients.copy()
    invalid[0, -1, 0] = 1.5
    with pytest.raises(ValueError, match="probability range|not monotone"):
        XGBMonotoneSplineArtifactData(
            policy_ids=artifact.policy_ids,
            row_indices=artifact.row_indices,
            action_grid=artifact.action_grid,
            coefficients=invalid,
            churn_min=artifact.churn_min,
            churn_max=artifact.churn_max,
            upper_slopes=artifact.upper_slopes,
        )


def test_real_portable_artifact_matches_all_archived_source_curves() -> None:
    from data.dataset_metadata import ACCEPTANCE_MODEL_ARTIFACTS

    portable_path = ACCEPTANCE_MODEL_ARTIFACTS[
        "xgb_monotone_spline_20260728"
    ]["path"]
    source_path = (
        portable_path.parents[2]
        / "model_sources"
        / "acceptance"
        / "xgb_monotone_spline_20260728.source.pkl"
    )
    artifact = load_xgb_monotone_spline_artifact(portable_path)
    wrapper = load_legacy_monotone_wrapper(source_path)
    normalized = {str(policy_id): curve for policy_id, curve in wrapper._curves.items()}
    action_grid = np.asarray([-0.02, *np.linspace(0.0, 0.16, 17), 0.25])

    assert artifact.source_sha256 == hashlib.sha256(source_path.read_bytes()).hexdigest()
    np.testing.assert_array_equal(artifact.policy_ids, sorted(normalized))
    expected = np.vstack([normalized[policy_id](action_grid) for policy_id in artifact.policy_ids])
    model = XGBMonotoneSplineAcceptance(artifact)
    actual = model.predict_acceptance(
        pd.DataFrame({"id": np.repeat(artifact.policy_ids, action_grid.size)}),
        np.tile(action_grid, artifact.policy_ids.size),
    ).reshape(artifact.policy_ids.size, action_grid.size)
    np.testing.assert_allclose(actual, 1.0 - expected, rtol=0.0, atol=1e-14)
