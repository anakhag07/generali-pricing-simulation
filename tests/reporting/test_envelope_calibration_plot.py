import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import numpy as np

from experiments.envelopes import reporting, study
from experiments.envelopes.core import CalibrationData
from experiments.provenance import file_record


def _action_view_inputs():
    contexts = np.array([1., -2., .5]) + np.array([
        [3., 0., 0.], [-3., 0., 0.], [0., 2., 0.],
        [0., -2., 0.], [0., 0., 1.], [0., 0., -1.],
    ])
    truth_beta = np.array([.25, 1., -.5, .75, 2.])
    predictor_beta = np.array([-.2, .7, -.1, .4, 1.7])
    auxiliary_beta = np.array([-.1, .9, -.3, .8, 1.9])
    logged_actions = np.array([-3., -2., -.1, .3, 2., 3.])
    logged_features = np.column_stack((np.ones(6), contexts, logged_actions))
    observations = logged_features @ truth_beta + np.array([.2, -.1, .3, -.2, .5, 0.])
    data = CalibrationData(contexts, logged_actions, observations)
    summary = {"target": {"context_count": 6, "context_dim": 3, "kind": "real_line"},
               "manifest": {"envelope": {"type": "linear_known_variance_transfer"}, "problem": {
                   "truth": {"type": "linear", "coefficients": truth_beta.tolist()},
                   "predictor": {"type": "linear", "coefficients": predictor_beta.tolist()},
               }}}
    X = np.repeat(contexts, 5, axis=0)
    actions = np.tile(np.linspace(-2., 2., 5), len(contexts))
    features = np.column_stack((np.ones(len(actions)), X, actions))
    center, truth, auxiliary = (features @ beta for beta in (predictor_beta, truth_beta, auxiliary_beta))
    width = 1. + .1*actions**2 + .05*X[:, 0]**2
    discrepancy = np.abs(center-auxiliary)
    radius = width+discrepancy
    arrays = {"X": X, "actions": actions, "truth": truth, "center": center,
              "auxiliary": auxiliary, "width": width, "discrepancy": discrepancy,
              "radius": radius, "excess": radius-np.abs(center-truth),
              "weights": np.full(len(actions), 1/len(actions))}
    return summary, arrays, data


def test_action_view_uses_same_context_adjustment_and_actual_radius_components():
    summary, arrays, data = _action_view_inputs()
    before = {key: value.copy() for key, value in arrays.items()}
    view = reporting._calibration_action_view(summary, arrays, data)
    beta = np.asarray(summary["manifest"]["problem"]["truth"]["coefficients"])
    expected_g = beta[0] + data.X @ beta[1:-1]
    np.testing.assert_allclose(view["g"], expected_g)
    np.testing.assert_allclose(view["truth"], beta[-1]*view["actions"])
    np.testing.assert_allclose(view["adjusted_responses"], data.y-expected_g)
    for source, target in (("radius", "mean_radius"), ("width", "mean_width"),
                           ("discrepancy", "mean_discrepancy")):
        np.testing.assert_allclose(view[target], arrays[source].reshape(6, 5).mean(axis=0))
    np.testing.assert_allclose(view["mean_radius"], view["mean_width"]+view["mean_discrepancy"])
    # The common overlap has its own geometric width, distinct from mean r.
    assert not np.allclose((view["upper"]-view["lower"])/2, view["mean_radius"])
    for key, values in before.items():
        np.testing.assert_array_equal(arrays[key], values)


def test_action_histogram_counts_include_window_endpoints_and_preserve_outside_draws():
    summary, arrays, data = _action_view_inputs()
    view = reporting._calibration_action_view(summary, arrays, data)
    np.testing.assert_array_equal(view["observed_in_window"], [False, True, True, True, True, False])
    np.testing.assert_allclose(view["histogram_edges"], np.linspace(-2., 2., 9))
    np.testing.assert_array_equal(view["histogram_counts"], [1, 0, 0, 1, 1, 0, 0, 1])
    assert view["histogram_counts"].sum() == 4
    assert np.count_nonzero(~view["observed_in_window"]) == 2
    np.testing.assert_array_equal(data.actions, [-3., -2., -.1, .3, 2., 3.])


def test_observed_responses_do_not_redefine_saved_mean_function_coverage():
    summary, arrays, data = _action_view_inputs()
    original = reporting._calibration_action_view(summary, arrays, data)
    changed_data = CalibrationData(data.X, data.actions, data.y+10.)
    changed = reporting._calibration_action_view(summary, arrays, changed_data)
    for key in ("truth", "mean_predictor", "lower", "upper", "covered", "empty",
                "mean_radius", "mean_width", "mean_discrepancy", "histogram_counts"):
        np.testing.assert_array_equal(changed[key], original[key])
    np.testing.assert_allclose(changed["adjusted_responses"], original["adjusted_responses"]+10.)


def test_calibration_figure_aligns_band_radii_and_observed_action_counts():
    summary, arrays, data = _action_view_inputs()
    before = {key: value.copy() for key, value in vars(data).items()}
    view = reporting._calibration_action_view(summary, arrays, data)
    figure = reporting._calibration_figure(data, view)
    try:
        assert len(figure.axes) == 3
        band, radii, counts = figure.axes
        assert all(ax.name == "rectilinear" for ax in figure.axes)
        assert band.get_shared_x_axes().joined(band, radii)
        assert band.get_shared_x_axes().joined(band, counts)
        assert all(ax.get_xlabel() and ax.get_ylabel() for ax in figure.axes)
        scatter = [artist for artist in band.collections if isinstance(artist, PathCollection)]
        assert len(scatter) == 1
        mask = view["observed_in_window"]
        np.testing.assert_allclose(scatter[0].get_offsets(),
                                   np.column_stack((data.actions[mask], view["adjusted_responses"][mask])))
        assert len(radii.lines) == 3
        assert len(counts.patches) == 1
        np.testing.assert_allclose(counts.patches[0].get_data().values, view["histogram_counts"])
        for key, values in before.items():
            np.testing.assert_array_equal(getattr(data, key), values)
    finally:
        plt.close(figure)


def _file_records(value):
    if isinstance(value, dict):
        if {"path", "sha256", "bytes"} <= value.keys():
            yield value
        for child in value.values():
            yield from _file_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _file_records(child)


def test_calibration_plot_replays_verified_inputs_without_mutating_them(tmp_path):
    source = Path(__file__).parents[2]/"scratch/retired_experiments/uncertainty_envelopes/manifests/uncertainty-envelopes.json"
    payload = json.loads(source.read_text())
    payload["calibration"]["n"] = 8
    payload["tightness"]["actions"]["count"] = 5
    manifest_path = tmp_path/"manifest.json"
    manifest_path.write_text(json.dumps(payload))
    manifest = study.load_manifest(manifest_path)
    summary = study.run_study(manifest, tmp_path/"runs", plots=False)
    assert summary["successful"]
    project = tmp_path/"runs"/manifest.name
    before = {path: path.read_bytes() for path in project.rglob("*") if path.is_file()}
    destination = tmp_path/"calibration-report"
    pdf = reporting.write_calibration_plot(project, destination)
    assert pdf == destination/"03_calibration_dataset.pdf"
    assert pdf.read_bytes().startswith(b"%PDF-")
    assert all(path.read_bytes() == contents for path, contents in before.items())
    provenance = json.loads((destination/"calibration_action_support.json").read_text())
    records = list(_file_records(provenance))
    recorded_paths = {record["path"] for record in records}
    row = summary["repetitions"][0]
    assert str(pdf.resolve()) in recorded_paths
    assert row["artifacts"]["calibration"]["path"] in recorded_paths
    assert row["artifacts"]["points"]["path"] in recorded_paths
    assert all(file_record(record["path"]) == record for record in records)
    assert not (destination/"calibration_pca.json").exists()
    assert not list(destination.rglob("*.png"))
