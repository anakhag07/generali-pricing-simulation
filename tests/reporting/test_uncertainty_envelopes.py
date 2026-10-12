from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.text import Text
import numpy as np
import pytest

from experiments.envelopes.core import aggregate_coverage
from experiments.envelopes import reporting
from experiments.envelopes.reporting import write_comparison, write_reports
from experiments.provenance import file_record


def _summary(tmp_path, *, empirical=False):
    points = tmp_path / "points.npz"
    np.savez(points, radius=[1., 2.], excess=[-.25, 1.], weights=[1., 3.])
    rows = [{"index": 0, "status": "violated", "metrics": {"radius_mean": 1.75},
             "artifacts": {"points": file_record(points)}}]
    summary = {
        "name": "uncertainty-envelopes",
        "target": {"kind": "real_line", "context_count": 3},
        "coverage": {"analytical": {"kind": "lower_bound", "value": .95},
                     "empirical": aggregate_coverage(rows, 1) if empirical else None},
        "repetitions": rows,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def _capture_figures(monkeypatch):
    saved = []
    original = Figure.savefig

    def capture(figure, *args, **kwargs):
        saved.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    return saved


def test_replay_preserves_inputs_and_single_containment_is_not_probability(tmp_path, monkeypatch):
    _summary(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    figures = _capture_figures(monkeypatch)
    paths = write_reports(tmp_path)
    assert [path.name for path in paths] == ["01_coverage.pdf", "02_radius_excess.pdf"]
    assert all(path.read_bytes().startswith(b"%PDF-") for path in paths)
    assert all(path.read_bytes() == data for path, data in before.items())
    probability, counts = figures[0].axes
    assert [tick.get_text() for tick in probability.get_xticklabels()] == ["Analytical\nlower bound"]
    assert probability.lines[0].get_ydata().tolist() == [.95]
    assert [patch.get_height() for patch in counts.patches] == [0, 1, 0, 0, 0]
    assert all(ax.get_xlabel() and ax.get_ylabel() for fig in figures for ax in fig.axes)
    assert not list(tmp_path.rglob("*.png"))


def test_histograms_keep_signed_excess_and_give_each_fit_equal_mass(tmp_path, monkeypatch):
    summary = _summary(tmp_path)
    second = tmp_path / "second.npz"
    np.savez(second, radius=[3.], excess=[2.], weights=[9.])
    summary["repetitions"].append({"index": 1, "status": "unknown", "metrics": {"radius_mean": 3.},
                                   "artifacts": {"points": file_record(second)}})
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    calls = []
    original = Axes.hist

    def capture(ax, values, *args, **kwargs):
        calls.append((np.array(values), np.array(kwargs["weights"])))
        return original(ax, values, *args, **kwargs)

    monkeypatch.setattr(Axes, "hist", capture)
    write_reports(tmp_path)
    np.testing.assert_allclose(calls[0][1], [.125, .375, .5])
    np.testing.assert_allclose(calls[1][0], [-.25, 1., 2.])


def test_empirical_probability_is_separate_from_analytical_bound(tmp_path, monkeypatch):
    _summary(tmp_path, empirical=True)
    figures = _capture_figures(monkeypatch)
    write_reports(tmp_path)
    probability = figures[0].axes[0]
    assert [tick.get_text() for tick in probability.get_xticklabels()] == [
        "Analytical\nlower bound", "Empirical\nestimate"]
    assert probability.lines[0].get_ydata().tolist() == [.95]
    assert probability.lines[1].get_ydata()[0] == 0.
    assert probability.get_legend().get_texts()[0].get_text() == "95% Wilson interval"


def test_unresolved_empirical_results_and_missing_cases_remain_visible(tmp_path, monkeypatch):
    summary = _summary(tmp_path)
    summary["repetitions"].extend([
        {"index": 1, "status": "unknown", "metrics": None, "artifacts": {}},
        {"index": 2, "status": "fit_failed", "metrics": None, "artifacts": {}},
    ])
    summary["coverage"]["empirical"] = aggregate_coverage(summary["repetitions"], 4)
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    figures = _capture_figures(monkeypatch)
    write_reports(tmp_path)
    probability, counts = figures[0].axes
    assert [patch.get_height() for patch in counts.patches] == [0, 1, 1, 1, 1]
    assert "Empirical\nunavailable" in [tick.get_text() for tick in probability.get_xticklabels()]
    assert len(probability.lines) == 1
    assert "Conditional on 1/4" in figures[1]._suptitle.get_text()


def test_empty_metric_panels_are_labeled(tmp_path, monkeypatch):
    summary = _summary(tmp_path)
    summary["repetitions"] = [{"index": 0, "status": "fit_failed", "metrics": None, "artifacts": {}}]
    summary["coverage"]["analytical"] = None
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    figures = _capture_figures(monkeypatch)
    write_reports(tmp_path)
    assert all("no available evaluations" in ax.get_title() for ax in figures[1].axes)


def test_tampered_point_artifact_is_rejected_before_reports(tmp_path):
    _summary(tmp_path)
    np.savez(tmp_path / "points.npz", radius=[99.], excess=[99.], weights=[1.])
    with pytest.raises(ValueError, match="hash/size mismatch"):
        write_reports(tmp_path)
    assert not (tmp_path / "reports").exists()


def test_empirical_aggregate_must_match_saved_outcomes(tmp_path):
    summary = _summary(tmp_path, empirical=True)
    summary["coverage"]["empirical"]["counts"]["covered"] = 1
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="counts disagree"):
        write_reports(tmp_path)


def _comparison_project(project, *, transfer=False, context_shift=0., predictor_context_coefficient=1.):
    directory = project / "repetitions" / "000000"
    directory.mkdir(parents=True)
    contexts = np.array([[1., 0., -1.], [.2, .3, .4], [-.5, -.6, -.7]]) + context_shift
    X = np.repeat(contexts, 5, axis=0)
    actions = np.tile(np.linspace(-2, 2, 5), 3)
    center = predictor_context_coefficient*X[:, 0] + actions
    truth = X[:, 0] + actions - .25
    radius = .5 + .1*actions**2 if transfer else np.full(len(actions), 3.)
    points = {"X": X, "actions": actions, "center": center, "truth": truth,
              "radius": radius, "excess": radius-np.abs(center-truth), "weights": np.full(len(actions), 1/len(actions))}
    if transfer:
        points["component_auxiliary"] = truth+.1
    np.savez(directory / "points.npz", **points)
    np.savez(directory / "calibration.npz", X=contexts, actions=np.zeros(3), y=np.zeros(3))
    method = {"type": "linear_known_variance_transfer"} if transfer else {"type": "constant_radius", "radius": 3.}
    manifest = {"envelope": method, "target": {"actions": {"type": "real_line"}},
                "problem": {"truth": {"type": "linear", "coefficients": [-.25, 1., 0., 0., 1.]},
                            "predictor": {"type": "linear", "coefficients": [0., predictor_context_coefficient, 0., 0., 1.]}}}
    contract = {"manifest": manifest}
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    (project / "contract.json").write_text(json.dumps({"sha256": digest, **contract}))
    analytical = ({"kind": "lower_bound", "value": .95, "event": "simultaneous coverage",
                   "conditioning": "fixed design", "assumptions": ["fixture"], "derivation": "fixture"}
                  if transfer else None)
    status = "covered" if np.all(points["excess"] >= 0) else "violated"
    row = {"index": 0, "status": status, "contract_sha256": digest,
           "analytical": analytical, "metrics": {"radius_mean": float(radius.mean()),
                                                  "excess_mean": float(points["excess"].mean())},
           "containment": {"status": status, "domain": "real_line", "method": "fixture"},
           "artifacts": {"points": file_record(directory / "points.npz"),
                         "calibration": file_record(directory / "calibration.npz")}}
    row_path = directory / "summary.json"
    row_path.write_text(json.dumps(row))
    (directory / "completion.json").write_text(json.dumps({"summary": file_record(row_path)}))
    summary = {"name": project.name, "manifest": manifest,
               "target": {"kind": "real_line", "context_count": 3, "context_dim": 3},
               "coverage": {"analytical": analytical, "empirical": None},
               "contract_sha256": digest, "complete": True, "successful": True,
               "requested_repetitions": 1, "repetitions": [row],
               "repetition_records": [file_record(row_path)]}
    (project / "summary.json").write_text(json.dumps(summary))
    return project


def test_comparison_replays_common_overlap_and_preserves_coverage_semantics(tmp_path, monkeypatch):
    projects = [_comparison_project(tmp_path / "constant"),
                _comparison_project(tmp_path / "writeup", transfer=True)]
    inputs = {path: path.read_bytes() for project in projects for path in project.rglob("*") if path.is_file()}
    figures = _capture_figures(monkeypatch)
    bands = []
    original = Axes.fill_between

    def capture(ax, actions, lower, upper, *args, **kwargs):
        bands.append(np.asarray(upper)-np.asarray(lower))
        return original(ax, actions, lower, upper, *args, **kwargs)

    monkeypatch.setattr(Axes, "fill_between", capture)
    destination = tmp_path / "comparison"
    paths = write_comparison(projects, destination)
    assert [path.name for path in paths] == ["01_coverage_comparison.pdf"]
    assert all(path.read_bytes().startswith(b"%PDF-") for path in paths)
    np.testing.assert_allclose(bands[0], 6.)
    assert [len(figure.axes) for figure in figures] == [2]
    assert all(ax.get_xlabel() and ax.get_ylabel() for figure in figures for ax in figure.axes)
    for figure in figures:
        # Check layout coordinates in memory without saving a raster preview.
        # Matplotlib retains tick-label objects outside the view; check only
        # text that is actually drawn, not those unused tick objects.
        drawn = []
        original_draw = Text.draw

        def record_text(text, renderer, *args, **kwargs):
            if text.get_visible() and text.get_text():
                drawn.append((text.get_text(), text.get_window_extent(renderer)))
            return original_draw(text, renderer, *args, **kwargs)

        canvas = FigureCanvasAgg(figure)
        with monkeypatch.context() as drawing:
            drawing.setattr(Text, "draw", record_text)
            canvas.draw()
        page = figure.bbox
        for label, box in drawn:
            assert box.x0 >= page.x0-.5 and box.y0 >= page.y0-.5, label
            assert box.x1 <= page.x1+.5 and box.y1 <= page.y1+.5, label
    assert figures[0].axes[0].get_shared_y_axes().joined(figures[0].axes[0], figures[0].axes[1])
    provenance = json.loads((destination / "comparison.json").read_text())
    assert provenance["projects"][0]["coverage"]["analytical"] is None
    assert provenance["projects"][1]["coverage"]["analytical"]["kind"] == "lower_bound"
    with np.load(destination / "adjusted_coverage.npz") as adjusted:
        assert adjusted["contexts"].shape == (3, 3)
        assert adjusted["lower"].shape == adjusted["upper"].shape == (2, 5)
        np.testing.assert_allclose(adjusted["truth"], adjusted["actions"])
        np.testing.assert_allclose(adjusted["mean_predictor"], adjusted["actions"]+.25)
        np.testing.assert_allclose(adjusted["upper"][0]-adjusted["lower"][0], 6.)
        assert adjusted["covered"].all() and not adjusted["empty"].any()
    assert all(path.read_bytes() == value for path, value in inputs.items())
    assert not list(destination.rglob("*.png"))


def _aligned_inputs():
    contexts = np.array([[-1.], [-.25], [.25], [1.]])
    actions = np.array([-2., 0., 2.])
    X = np.repeat(contexts, len(actions), axis=0)
    a = np.tile(actions, len(contexts))
    truth = X[:, 0]+a
    center = 2*X[:, 0]+a
    radii = np.array([[1.2, .5, 1.2], [.3, 1., .3], [.3, 1., .3], [1.2, 2., .5]])
    summary = {"target": {"context_count": 4, "context_dim": 1},
               "manifest": {"problem": {
                   "truth": {"type": "linear", "coefficients": [0., 1., 1.]},
                   "predictor": {"type": "linear", "coefficients": [0., 2., 1.]},
               }}}
    arrays = {"X": X, "actions": a, "truth": truth, "center": center,
              "radius": radii.ravel(), "excess": radii.ravel()-np.abs(center-truth),
              "weights": np.full(len(a), 1/len(a))}
    return summary, arrays


def test_common_overlap_checks_all_contexts_when_mean_prediction_hides_failure():
    summary, arrays = _aligned_inputs()
    aligned = reporting._aligned_comparison(summary, arrays)
    shape = (4, 3)
    individual_covered = np.abs(arrays["center"]-arrays["truth"]) <= arrays["radius"]
    np.testing.assert_array_equal(aligned["covered"], individual_covered.reshape(shape).all(axis=0))
    assert individual_covered.reshape(shape)[:3, -1].all()
    assert not individual_covered.reshape(shape)[3, -1]
    np.testing.assert_array_equal(aligned["covered"], [True, False, False])
    np.testing.assert_array_equal(aligned["empty"], [False, False, True])
    np.testing.assert_allclose(aligned["mean_predictor"], aligned["truth"])
    shifted_centers = arrays["center"].reshape(shape)-aligned["g"][:, None]
    radii = arrays["radius"].reshape(shape)
    np.testing.assert_allclose(aligned["lower"], np.max(shifted_centers-radii, axis=0))
    np.testing.assert_allclose(aligned["upper"], np.min(shifted_centers+radii, axis=0))
    assert len(aligned["contexts"]) == 4


def test_context_adjustment_is_invariant_to_shared_context_function_and_preserves_radii():
    summary, arrays = _aligned_inputs()
    before_radius = arrays["radius"].copy()
    original = reporting._aligned_comparison(summary, arrays)
    changed_summary, changed_arrays = deepcopy(summary), deepcopy(arrays)
    shift = -.3+2*arrays["X"][:, 0]
    changed_arrays["truth"] += shift
    changed_arrays["center"] += shift
    changed_summary["manifest"]["problem"]["truth"]["coefficients"] = [-.3, 3., 1.]
    changed_summary["manifest"]["problem"]["predictor"]["coefficients"] = [-.3, 4., 1.]
    changed = reporting._aligned_comparison(changed_summary, changed_arrays)
    for key in ("actions", "truth", "mean_predictor", "lower", "upper", "covered", "empty"):
        np.testing.assert_allclose(changed[key], original[key], atol=1e-14)
    np.testing.assert_allclose(changed["g"]-original["g"], [-2.3, -.8, .2, 1.7])
    np.testing.assert_array_equal(arrays["radius"], before_radius)
    np.testing.assert_array_equal(changed_arrays["radius"], before_radius)


def test_context_alignment_rejects_nonlinear_truth():
    summary, arrays = _aligned_inputs()
    summary["manifest"]["problem"]["truth"] = {"type": "python", "callable": "example:nonlinear"}
    with pytest.raises(ValueError, match="linear"):
        reporting._aligned_comparison(summary, arrays)


def test_empty_common_intersection_is_not_shaded_as_valid_band(tmp_path, monkeypatch):
    projects = [_comparison_project(tmp_path / "constant", predictor_context_coefficient=10.),
                _comparison_project(tmp_path / "writeup", transfer=True, predictor_context_coefficient=10.)]
    calls = []
    original = Axes.fill_between

    def capture(ax, actions, lower, upper, *args, **kwargs):
        calls.append((np.asarray(lower), np.asarray(upper), kwargs))
        return original(ax, actions, lower, upper, *args, **kwargs)

    monkeypatch.setattr(Axes, "fill_between", capture)
    write_comparison(projects, tmp_path / "comparison")
    assert calls
    for lower, upper, kwargs in calls:
        valid = lower <= upper
        where = np.asarray(kwargs.get("where", np.ones_like(valid)), dtype=bool)
        # An empty common interval must be masked, rather than filled with its
        # inverted bounds as though it were a valid simultaneous envelope.
        assert not np.any(where & ~valid)


def test_comparison_rejects_different_diagnostic_contexts(tmp_path):
    projects = [_comparison_project(tmp_path / "constant"),
                _comparison_project(tmp_path / "writeup", transfer=True, context_shift=.1)]
    with pytest.raises(ValueError, match="identical saved diagnostic X"):
        write_comparison(projects, tmp_path / "comparison")
    assert not (tmp_path / "comparison").exists()


def test_comparison_checks_completion_hash_before_reading_saved_results(tmp_path):
    projects = [_comparison_project(tmp_path / "constant"),
                _comparison_project(tmp_path / "writeup", transfer=True)]
    completion_path = projects[1] / "repetitions" / "000000" / "completion.json"
    completion = json.loads(completion_path.read_text())
    completion["summary"]["sha256"] = "0"*64
    completion_path.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="Saved artifact changed"):
        write_comparison(projects, tmp_path / "comparison")


def test_comparison_rejects_edited_radius_label_in_project_summary(tmp_path):
    projects = [_comparison_project(tmp_path / "constant"),
                _comparison_project(tmp_path / "writeup", transfer=True)]
    path = projects[0] / "summary.json"
    summary = json.loads(path.read_text())
    summary["manifest"]["envelope"]["radius"] = 30.
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="manifest or contract identity"):
        write_comparison(projects, tmp_path / "comparison")
