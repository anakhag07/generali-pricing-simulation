"""Replay and semantic checks for context-averaged envelope diagnostics."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.envelopes import diagnostics, study
from experiments.provenance import file_record


def _summary(rows, *, count=3, requested=None, actions=None):
    kind = "real_line" if actions is None else "finite_set"
    target_actions = {"type": kind}
    if actions is not None:
        target_actions["values"] = actions
    return {
        "target": {"kind": kind, "context_count": count, "context_dim": 1},
        "manifest": {
            "target": {"contexts": "calibration", "actions": target_actions},
            "coverage": {"mode": "analytical"},
            "envelope": {"type": "constant_radius", "radius": 3.},
        },
        "requested_repetitions": len(rows) if requested is None else requested,
        "repetitions": rows,
        "coverage": {"analytical": None, "empirical": None},
    }


def _row(status, *, index=0, contexts=None, domain="real_line", method="custom", **details):
    if contexts is not None:
        details["contexts"] = [
            {"context_index": i, "covered": covered}
            for i, covered in enumerate(contexts)
        ]
    return {
        "index": index, "status": status,
        "containment": {"status": status, "domain": domain,
                        "method": method, "details": details},
        "artifacts": {},
    }


def _grid(*, count=3, actions=(-1., 0., 1.)):
    actions = np.asarray(actions)
    shape = (count, len(actions))
    return {"X": np.arange(count, dtype=float)[:, None], "actions": actions,
            "center": np.zeros(shape), "truth": np.zeros(shape),
            "radius": np.full(shape, 3.), "excess": np.full(shape, 3.)}


def test_global_violation_does_not_imply_every_context_failed():
    summary = _summary([_row("violated")])
    result = diagnostics._coverage(summary, {0: _grid()})
    np.testing.assert_array_equal(result["statuses"], [[-1, -1, -1]])
    assert result["mean"] is None
    assert result["covered_fraction"] == 0
    assert result["unresolved_fraction"] == 1


def test_global_coverage_certifies_every_context_without_per_context_records():
    result = diagnostics._coverage(_summary([_row("covered")]), {})
    np.testing.assert_array_equal(result["statuses"], [[1, 1, 1]])
    assert result["mean"] == 1
    assert result["unresolved_fraction"] == 0


def test_legacy_constant_slope_mismatch_certifies_failure_on_every_real_fiber():
    row = _row("violated", method="constant_affine_all_real", slope_mismatch=True,
               context_count=3, action_slope_difference="1/4")
    result = diagnostics._coverage(_summary([row]), {0: _grid()})
    # The finite display is covered, but unequal slopes fail in the real-line tails.
    np.testing.assert_array_equal(result["statuses"], [[0, 0, 0]])
    assert result["mean"] == 0
    assert result["unresolved_fraction"] == 0


def test_context_average_is_fractional_and_distinct_from_joint_coverage():
    row = _row("violated", contexts=[True, False, True])
    result = diagnostics._coverage(_summary([row]), {})
    np.testing.assert_array_equal(result["statuses"], [[1, 0, 1]])
    assert result["mean"] == pytest.approx(2/3)
    assert result["covered_fraction"] == pytest.approx(2/3)


def test_custom_partial_certificates_preserve_unknown_contexts():
    row = _row("unknown", contexts=[True, None, False])
    result = diagnostics._coverage(_summary([row]), {})
    np.testing.assert_array_equal(result["statuses"], [[1, -1, 0]])
    assert result["mean"] is None
    assert result["covered_fraction"] == pytest.approx(1/3)
    assert result["unresolved_fraction"] == pytest.approx(1/3)


def test_context_certificates_cannot_contradict_the_global_decision():
    summary = _summary([_row("covered", contexts=[True, False, True])])
    with pytest.raises(ValueError, match="contradicts"):
        diagnostics._coverage(summary, {})


def test_failed_and_missing_repetitions_keep_the_requested_denominator():
    failed = {"index": 1, "status": "fit_failed", "containment": None,
              "artifacts": {}}
    summary = _summary([_row("covered"), failed], requested=3)
    result = diagnostics._coverage(summary, {})
    np.testing.assert_array_equal(result["statuses"],
                                  [[1, 1, 1], [-1, -1, -1], [-1, -1, -1]])
    assert result["mean"] is None
    assert result["covered_fraction"] == pytest.approx(1/3)
    assert result["unresolved_fraction"] == pytest.approx(2/3)


def test_finite_fallback_uses_only_exact_declared_target_actions():
    grid = _grid(actions=(-2., -1., 0., 1., 2.))
    grid["truth"][0, 0] = 99.  # Outside the declared target, hence irrelevant.
    grid["truth"][1, 3] = 4.   # One target violation makes this context fail.
    grid["excess"] = grid["radius"]-np.abs(grid["center"]-grid["truth"])
    row = _row("violated", domain="finite_set")
    summary = _summary([row], actions=[-1., 1.])
    result = diagnostics._coverage(summary, {0: grid})
    np.testing.assert_array_equal(result["statuses"], [[1, 0, 1]])
    assert result["mean"] == pytest.approx(2/3)


def test_finite_fallback_never_interpolates_an_unobserved_target_action():
    row = _row("unknown", domain="finite_set")
    summary = _summary([row], actions=[-.5, .5])
    result = diagnostics._coverage(summary, {0: _grid()})
    np.testing.assert_array_equal(result["statuses"], [[-1, -1, -1]])
    assert result["mean"] is None


def test_mean_radius_stays_three_when_mean_errors_cancel_and_all_contexts_fail():
    grid = _grid(count=2)
    grid["truth"][:] = [[4., 4., 4.], [-4., -4., -4.]]
    grid["excess"] = grid["radius"]-np.abs(grid["center"]-grid["truth"])
    averages = diagnostics._averages(grid)
    np.testing.assert_array_equal(averages["radius"], [3., 3., 3.])
    np.testing.assert_array_equal(averages["truth"], averages["center"])
    summary = _summary([_row("violated", domain="finite_set")], count=2,
                       actions=grid["actions"].tolist())
    result = diagnostics._coverage(summary, {0: grid})
    np.testing.assert_array_equal(result["statuses"], [[0, 0]])


def test_component_decomposition_averages_absolute_discrepancy_before_summing():
    grid = _grid(count=2)
    grid["auxiliary"] = np.array([[2., -1., 3.], [-2., 1., -3.]])
    grid["discrepancy"] = np.abs(grid["center"]-grid["auxiliary"])
    grid["width"] = np.array([[.5, 1., 2.], [1.5, 3., 4.]])
    grid["radius"] = grid["discrepancy"]+grid["width"]
    averages = diagnostics._averages(grid)
    np.testing.assert_allclose(averages["discrepancy"], [2., 1., 3.])
    np.testing.assert_allclose(averages["width"], [1., 2., 3.])
    np.testing.assert_allclose(averages["radius"], [3., 3., 6.])
    np.testing.assert_allclose(averages["radius"],
                               averages["discrepancy"]+averages["width"])
    np.testing.assert_array_equal(np.abs(averages["center"]-averages["auxiliary"]), 0.)


@pytest.mark.parametrize("inconsistent", ["component_sum", "absolute_discrepancy"])
def test_replay_rejects_component_errors_that_cancel_in_the_mean(tmp_path, inconsistent):
    contexts = np.array([[0.], [1.]])
    actions = np.array([-1., 1.])
    center = np.zeros((2, 2))
    auxiliary = np.full((2, 2), 2.)
    discrepancy = np.abs(center-auxiliary)
    width = np.ones((2, 2))
    radius = discrepancy+width
    # The average still agrees; every saved context must agree separately.
    if inconsistent == "component_sum":
        width += np.array([[.5], [-.5]])
    else:
        auxiliary += np.array([[.5], [-.5]])
    point_path, calibration_path = tmp_path/"points.npz", tmp_path/"calibration.npz"
    np.savez(point_path, X=np.repeat(contexts, len(actions), axis=0),
             actions=np.tile(actions, len(contexts)), center=center.ravel(),
             truth=center.ravel(), radius=radius.ravel(), excess=radius.ravel(),
             weights=np.full(4, .25), component_auxiliary=auxiliary.ravel(),
             component_discrepancy=discrepancy.ravel(), component_width=width.ravel())
    np.savez(calibration_path, X=contexts, actions=actions, y=np.zeros(2))
    row = {"artifacts": {"points": file_record(point_path),
                          "calibration": file_record(calibration_path)}}
    with pytest.raises(ValueError, match="components|discrepancy"):
        diagnostics._load_grid(row, {"context_count": 2, "context_dim": 1})


@pytest.mark.parametrize("actions, conflict, expected_actions", [
    ([1., -1., 0., 1.], False, [-1., 0., 1.]),
    ([0.], False, [0.]),
    ([1., -1., 0., 1.], True, None),
])
def test_saved_grid_replay_handles_action_order_duplicates_and_singletons(
        tmp_path, actions, conflict, expected_actions):
    contexts = np.array([[-2.], [3.]])
    actions = np.asarray(actions)
    center = contexts+actions[None, :]
    truth = center-.25
    radius = np.broadcast_to(3.+.2*actions**2, center.shape)
    if conflict:
        center[0, -1] += .1
    point_path, calibration_path = tmp_path/"points.npz", tmp_path/"calibration.npz"
    np.savez(point_path, X=np.repeat(contexts, len(actions), axis=0),
             actions=np.tile(actions, len(contexts)), center=center.ravel(),
             truth=truth.ravel(), radius=radius.ravel(),
             excess=(radius-np.abs(center-truth)).ravel(),
             weights=np.full(center.size, 1/center.size))
    np.savez(calibration_path, X=contexts, actions=np.zeros(2), y=np.zeros(2))
    row = {"artifacts": {"points": file_record(point_path),
                          "calibration": file_record(calibration_path)}}
    target = {"context_count": 2, "context_dim": 1}
    if conflict:
        with pytest.raises(ValueError, match="conflicting"):
            diagnostics._load_grid(row, target)
        return
    grid = diagnostics._load_grid(row, target)
    expected_actions = np.asarray(expected_actions)
    np.testing.assert_array_equal(grid["actions"], expected_actions)
    np.testing.assert_array_equal(grid["X"], contexts)
    np.testing.assert_allclose(grid["center"], contexts+expected_actions[None, :])
    np.testing.assert_allclose(grid["truth"], grid["center"]-.25)
    np.testing.assert_allclose(grid["radius"],
                               np.broadcast_to(3.+.2*expected_actions**2, grid["center"].shape))
    np.testing.assert_allclose(grid["excess"], grid["radius"]-.25)
    assert file_record(point_path) == row["artifacts"]["points"]


@pytest.mark.parametrize("contexts, expected", [
    ([[3., 0.], [0., 1.], [0., 0.], [2., 0.]], [2, 1, 0]),
    ([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]], [0, 1, 3]),
    ([[2.], [1.]], [1, 0]),
    ([[1.]], [0]),
])
def test_context_slices_use_stable_norm_order_and_lower_median(contexts, expected):
    np.testing.assert_array_equal(diagnostics._slice_indices(np.asarray(contexts)), expected)


def test_reports_replay_a_saved_fit_and_reject_tampered_inputs(tmp_path, monkeypatch):
    manifest_path = Path(__file__).parents[2]/"manifests"/"uncertainty-envelopes.json"
    payload = json.loads(manifest_path.read_text())
    payload["calibration"].update(n=8, context_dim=1)
    payload["problem"]["truth"]["coefficients"] = [0., .5, 1.]
    payload["problem"]["predictor"]["coefficients"] = [.2, .4, .8]
    payload["tightness"]["actions"]["count"] = 5
    local_manifest = tmp_path/"manifest.json"
    local_manifest.write_text(json.dumps(payload))
    specification = study.load_manifest(local_manifest)
    summary = study.run_study(specification, tmp_path/"runs", plots=False)
    project = tmp_path/"runs"/specification.name
    original = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("Report replay must not refit or regenerate data.")

    monkeypatch.setattr(study, "fit_linear_transfer", forbidden_fit)
    monkeypatch.setattr(study, "run_repetition", forbidden_fit)
    paths = diagnostics.write_diagnostic_reports(project)
    assert len(paths) == 4
    assert all(path.parent == project/"reports"/"diagnostics" for path in paths)
    assert all(path.suffix == ".pdf" and path.read_bytes().startswith(b"%PDF-") for path in paths)
    assert (project/"reports"/"diagnostics"/"diagnostics.json").is_file()
    assert all(path.read_bytes() == contents for path, contents in original.items())
    assert not list(project.rglob("*.png"))

    collected = study.collect(specification, tmp_path/"runs", plots=True)
    assert len(collected["plots"]) == 6
    assert all(file_record(record["path"]) == record for record in collected["plots"])
    provenance = json.loads((project/"reports"/"diagnostics"/"diagnostics.json").read_text())
    assert file_record(provenance["summary"]["path"]) == provenance["summary"]
    assert Path(provenance["summary"]["path"]).name == "source_summary.json"
    snapshot = json.loads(Path(provenance["summary"]["path"]).read_text())
    assert snapshot["repetitions"] == collected["repetitions"]

    # Repetition and original artifact identities must be verified before replay.
    point_path = Path(summary["repetitions"][0]["artifacts"]["points"]["path"])
    point_path.write_bytes(original[point_path]+b"tampered")
    with pytest.raises(ValueError):
        diagnostics.write_diagnostic_reports(project)
    point_path.write_bytes(original[point_path])

    edited = deepcopy(summary)
    edited["manifest"]["envelope"]["delta"] = .5
    (project/"summary.json").write_text(json.dumps(edited))
    with pytest.raises(ValueError):
        diagnostics.write_diagnostic_reports(project)

    edited = deepcopy(summary)
    edited["requested_repetitions"] += 1
    (project/"summary.json").write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="repetition|calibration"):
        diagnostics.write_diagnostic_reports(project)
