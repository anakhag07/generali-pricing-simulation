"""Tests for portable per-policy shifted-sigmoid acceptance artifacts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data.xgb_sigmoid import (
    XGBSigmoidAcceptance,
    XGBSigmoidArtifactData,
    canonical_row_indices_for_policy_ids,
    extract_sigmoid_parameters,
    load_xgb_sigmoid_artifact,
    save_xgb_sigmoid_artifact,
)


def _artifact() -> XGBSigmoidArtifactData:
    return XGBSigmoidArtifactData(
        policy_ids=np.asarray(["101", "202"]),
        row_indices=np.asarray([3, 7]),
        parameters=np.asarray([[8.0, 0.08, -0.1], [5.0, 0.06, 0.0]]),
        source_sha256="source",
        embedded_booster_sha256="booster",
    )


def test_portable_round_trip(tmp_path) -> None:
    output = save_xgb_sigmoid_artifact(_artifact(), tmp_path / "curves.npz")
    loaded = load_xgb_sigmoid_artifact(output)
    np.testing.assert_array_equal(loaded.policy_ids, ["101", "202"])
    np.testing.assert_array_equal(loaded.row_indices, [3, 7])
    np.testing.assert_allclose(loaded.parameters, _artifact().parameters)
    assert loaded.source_sha256 == "source"
    assert loaded.embedded_booster_sha256 == "booster"


def test_acceptance_and_derivative_match_finite_difference() -> None:
    model = XGBSigmoidAcceptance(_artifact())
    frame = pd.DataFrame({"id": ["101", "202"]})
    u = np.asarray([0.05, 0.12])
    eps = 1e-6
    analytical = model.d_acceptance_du(frame, u)
    finite_difference = (
        model.predict_acceptance(frame, u + eps)
        - model.predict_acceptance(frame, u - eps)
    ) / (2.0 * eps)
    np.testing.assert_allclose(analytical, finite_difference, rtol=1e-7, atol=1e-8)
    assert np.all(analytical < 0.0)


def test_clipped_curve_has_zero_derivative() -> None:
    artifact = XGBSigmoidArtifactData(
        policy_ids=np.asarray(["101"]),
        row_indices=np.asarray([3]),
        parameters=np.asarray([[8.0, 0.08, 1.0]]),
    )
    model = XGBSigmoidAcceptance(artifact)
    frame = pd.DataFrame({"id": ["101"]})
    np.testing.assert_allclose(model.predict_acceptance(frame, np.asarray([0.05])), 0.0)
    np.testing.assert_allclose(model.d_acceptance_du(frame, np.asarray([0.05])), 0.0)


def test_runtime_rejects_uncovered_policy_ids() -> None:
    model = XGBSigmoidAcceptance(_artifact())
    with pytest.raises(ValueError, match="No fitted acceptance sigmoid"):
        model.predict_acceptance(pd.DataFrame({"id": ["999"]}), np.asarray([0.1]))


def test_extract_parameters_is_sorted_and_rejects_malformed_curves() -> None:
    wrapper = SimpleNamespace(
        _function_name="sigmoid_with_shift",
        _curves={
            "202": SimpleNamespace(params=(2.0, 0.2, 0.02)),
            "101": SimpleNamespace(params=(1.0, 0.1, 0.01)),
        },
    )
    policy_ids, parameters = extract_sigmoid_parameters(wrapper)
    np.testing.assert_array_equal(policy_ids, ["101", "202"])
    np.testing.assert_allclose(parameters[:, 0], [1.0, 2.0])
    wrapper._curves["101"] = SimpleNamespace(params=(1.0, 0.1))
    with pytest.raises(ValueError, match="invalid sigmoid parameters"):
        extract_sigmoid_parameters(wrapper)


def test_extract_rejects_unsupported_smoother() -> None:
    with pytest.raises(ValueError, match="Only sigmoid_with_shift"):
        extract_sigmoid_parameters(
            SimpleNamespace(_function_name="spline", _curves={"101": object()})
        )


def test_canonical_rows_resolve_in_requested_order(tmp_path) -> None:
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame({"id": ["202", "999", "101"]}).to_csv(dataset, sep=";", index=False)
    rows = canonical_row_indices_for_policy_ids(dataset, np.asarray(["101", "202"]))
    np.testing.assert_array_equal(rows, [2, 0])
