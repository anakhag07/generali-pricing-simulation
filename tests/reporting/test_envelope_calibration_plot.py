import json
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Path3DCollection
import numpy as np
import pytest

from experiments.envelopes import reporting, study
from experiments.envelopes.core import CalibrationData
from experiments.provenance import file_record


def _projection_inputs():
    context_mean = np.array([1., -2., .5])
    offsets = np.array([[3., 0., 0.], [-3., 0., 0.],
                        [0., 2., 0.], [0., -2., 0.],
                        [0., 0., 1.], [0., 0., -1.]])
    data = CalibrationData(context_mean+offsets, np.linspace(-2., 3., 6), np.arange(6.)+.4)
    summary = {"manifest": {"problem": {
        "truth": {"type": "linear", "coefficients": [.25, 1., -.5, .75, 2.]},
        "predictor": {"type": "linear", "coefficients": [-.2, .7, -.1, .4, 1.7]},
    }}}
    fit = {"metadata": {"construction": "linear_known_variance_transfer"},
           "state": {"beta": [-.1, .9, -.3, .8, 1.9]}}
    return summary, data, fit


def test_pca_uses_centered_contexts_only_and_known_leading_variance():
    summary, data, fit = _projection_inputs()
    result = reporting._calibration_projection(summary, data, fit)
    np.testing.assert_allclose(result["context_mean"], [1., -2., .5])
    np.testing.assert_allclose(result["pc1"], [1., 0., 0.])
    np.testing.assert_allclose(result["singular_values"]**2, [18., 8., 2.])
    np.testing.assert_allclose(result["scores"], [3., -3., 0., 0., 0., 0.])
    assert result["explained_variance_ratio"] == pytest.approx(18/28)
    other = CalibrationData(data.X, data.actions*10000.+77., data.y[::-1]*-500.)
    changed = reporting._calibration_projection(summary, other, fit)
    for key in ("context_mean", "pc1", "scores", "singular_values"):
        np.testing.assert_array_equal(changed[key], result[key])
    assert changed["explained_variance_ratio"] == result["explained_variance_ratio"]
    assert changed["planes"] == result["planes"]


def test_reference_planes_are_slices_of_saved_models_without_refitting():
    summary, data, fit = _projection_inputs()
    projection = reporting._calibration_projection(summary, data, fit)
    t, actions = np.array([-1.7, 0., 2.4]), np.array([.2, -3., 1.1])
    X = projection["context_mean"] + t[:, None]*projection["pc1"]
    coefficients = {name: np.asarray(summary["manifest"]["problem"][name]["coefficients"])
                    for name in ("truth", "predictor")}
    coefficients["auxiliary"] = np.asarray(fit["state"]["beta"])
    for name, beta in coefficients.items():
        plane = projection["planes"][name]
        values = plane["intercept"] + plane["action_slope"]*actions + plane["pc1_slope"]*t
        expected = beta[0] + X @ beta[1:-1] + beta[-1]*actions
        np.testing.assert_allclose(values, expected)


def test_calibration_figure_keeps_raw_responses_and_default_camera():
    summary, data, fit = _projection_inputs()
    before = {key: value.copy() for key, value in vars(data).items()}
    projection = reporting._calibration_projection(summary, data, fit)
    figure = reporting._calibration_figure(data, projection)
    try:
        assert len(figure.axes) == 1
        ax = figure.axes[0]
        assert ax.name == "3d"
        assert ax.elev == 30 and ax.azim == -60
        assert ax.get_xlabel() and ax.get_ylabel() and ax.get_zlabel()
        scatter = [artist for artist in ax.collections if isinstance(artist, Path3DCollection)]
        wireframes = [artist for artist in ax.collections if isinstance(artist, Line3DCollection)]
        assert len(scatter) == 1 and len(wireframes) == 3
        actions, scores, responses = scatter[0]._offsets3d
        np.testing.assert_array_equal(actions, data.actions)
        np.testing.assert_array_equal(scores, projection["scores"])
        np.testing.assert_array_equal(responses, data.y)
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
    source = Path(__file__).parents[2]/"manifests"/"uncertainty-envelopes.json"
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
    provenance = json.loads((destination/"calibration_pca.json").read_text())
    records = list(_file_records(provenance))
    recorded_paths = {record["path"] for record in records}
    row = summary["repetitions"][0]
    assert str(pdf.resolve()) in recorded_paths
    assert row["artifacts"]["calibration"]["path"] in recorded_paths
    assert row["artifacts"]["fit"]["path"] in recorded_paths
    assert all(file_record(record["path"]) == record for record in records)
    assert not list(destination.rglob("*.png"))
