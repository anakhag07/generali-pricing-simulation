"""Saved-data q/b replays, baseline identity, and nested envelope behavior."""

import json
from pathlib import Path

import pytest

from experiments import coverage_kernel_sweep as source
from experiments import coverage_kernel_parameter_sweep as sweep


MANIFESTS = Path(__file__).parents[2]/"manifests"


@pytest.fixture(scope="module")
def saved_experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("kernel-parameter-source")
    payload = json.loads((MANIFESTS/"coverage_kernel_dense_n_sweep.json").read_text())
    payload.update(name="kernel-source-test", datasets=2, N_values=[25, 50])
    path = root/"source-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = source.load_manifest(path)
    for index in range(2):
        source.run_dataset(manifest, index, root)
    source.collect(manifest, root)
    return root, payload


@pytest.mark.parametrize("axis,values", [
    ("q", [0.5, 1.0, 1.959963984540054, 2.5]),
    ("b", [0.05, 0.1, 0.2]),
])
def test_parameter_sweep_replays_saved_fit_and_baseline(saved_experiment, axis, values):
    root, source_payload = saved_experiment
    payload = json.loads((MANIFESTS/f"coverage_kernel_{axis}_sweep.json").read_text())
    payload.update(name=f"kernel-{axis}-test", source_name=source_payload["name"],
                   datasets=2, N_values=[25, 50], report_N_values=[25, 50])
    payload[f"{axis}_values"] = values
    path = root/f"{axis}-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = sweep.load_manifest(path)
    for index in range(2):
        saved = sweep.run_dataset(manifest, index, root)
        original = json.loads((root/source_payload["name"]/"datasets"/
                               f"dataset-{index:03d}"/"summary.json").read_text())
        assert len(saved["rows"]) == 2*len(values)
        for n in (25, 50):
            group = [row for row in saved["rows"] if row["N"] == n]
            baseline = next(row for row in group if row["baseline_replay"])
            source_row = next(row for row in original["rows"] if row["N"] == n)
            for key in ("lcb_action", "optimization_state", "covered", "regret", "optimizer_attempts"):
                assert baseline[key] == source_row[key]
            widths = [row["width_at_true_optimum"] for row in group]
            coverage = [row["covered"] for row in group]
            assert all(state is not None for state in coverage)
            if axis == "q":
                assert widths == sorted(widths)
                assert coverage == sorted(coverage)
            else:
                assert widths == sorted(widths, reverse=True)
                assert coverage == sorted(coverage, reverse=True)
    sweep.collect(manifest, root)
    project = root/manifest.name
    summary = json.loads((project/"summary.json").read_text())
    assert len(summary["metrics"]) == 2*len(values)
    assert len(summary["plots"]) == 3
    for record in summary["plots"]:
        assert Path(record["path"]).read_bytes().startswith(b"%PDF-")
    assert sweep.run_dataset(manifest, 0, root)["contract"] == json.loads(
        (project/"datasets"/"dataset-000"/"summary.json").read_text()
    )["contract"]


@pytest.mark.parametrize("axis", ["q", "b"])
def test_manifest_rejects_missing_baseline_at_source_check(saved_experiment, axis):
    root, source_payload = saved_experiment
    payload = json.loads((MANIFESTS/f"coverage_kernel_{axis}_sweep.json").read_text())
    payload.update(name=f"kernel-{axis}-missing", source_name=source_payload["name"],
                   datasets=2, N_values=[25, 50], report_N_values=[25, 50])
    payload[f"{axis}_values"] = [0.5] if axis == "q" else [0.2]
    path = root/f"{axis}-missing-manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = sweep.load_manifest(path)
    with pytest.raises(ValueError, match="exact original"):
        sweep.run_dataset(manifest, 0, root)
