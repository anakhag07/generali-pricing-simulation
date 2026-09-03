import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "design_bench.py"
SPEC = importlib.util.spec_from_file_location("design_bench_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
design_bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(design_bench)


def _arrays(dimension):
    x = np.arange(201 * dimension, dtype=np.float32).reshape(201, dimension)
    y = np.linspace(-1.0, 1.0, 201, dtype=np.float32)[:, None]
    return x, y


def test_supported_tasks_and_reference_configuration(tmp_path):
    assert design_bench._task_spec(design_bench.ANT_TASK)["dimension"] == 60
    assert design_bench._task_spec(design_bench.DKITTY_TASK)["dimension"] == 56
    with pytest.raises(ValueError, match="Unsupported task"):
        design_bench._task_spec("HopperController-Exact-v0")

    config = design_bench._baseline_config(
        design_bench._task_spec(design_bench.ANT_TASK), tmp_path, "reference"
    )
    assert config["epochs"] == 100
    assert config["hidden_size"] == 2048
    assert config["solver_samples"] == 128
    assert config["solver_steps"] == 200
    assert config["do_evaluation"] is False


def test_smoke_mode_only_reduces_runtime_knobs(tmp_path):
    spec = design_bench._task_spec(design_bench.DKITTY_TASK)
    reference = design_bench._baseline_config(spec, tmp_path, "reference")
    smoke = design_bench._baseline_config(spec, tmp_path, "smoke")
    differing = {key for key in reference if reference[key] != smoke[key]}
    assert differing == {"epochs", "solver_samples", "solver_steps"}


def test_dataset_identity_changes_when_raw_data_changes():
    spec = design_bench._task_spec(design_bench.ANT_TASK)
    x, y = _arrays(60)
    first = design_bench._manifest_id(design_bench._identity(spec, x, y))
    changed = x.copy()
    changed[0, 0] += 1.0
    second = design_bench._manifest_id(design_bench._identity(spec, changed, y))
    assert first != second


def test_export_and_evaluate_use_official_raw_task_api(monkeypatch, tmp_path):
    x, y = _arrays(60)
    seen = []

    class FakeTask:
        def __init__(self):
            self.x, self.y = x, y

        def predict(self, candidates):
            seen.append(candidates.copy())
            return np.sum(candidates, axis=1, keepdims=True)

    monkeypatch.setattr(design_bench, "_make_task", lambda spec: FakeTask())
    dataset = tmp_path / "dataset"
    design_bench.export_dataset(design_bench.ANT_TASK, dataset)
    candidates = x[:2]
    candidate_path = tmp_path / "candidates.npy"
    np.save(candidate_path, candidates, allow_pickle=False)
    output = tmp_path / "evaluation"
    design_bench.evaluate(dataset, candidate_path, output)

    np.testing.assert_array_equal(seen[0], candidates)
    np.testing.assert_array_equal(
        np.load(output / "scores.npy"),
        np.sum(candidates, axis=1)[:, None],
    )
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["dataset_manifest_id"] == manifest["manifest_id"]


def test_load_dataset_rejects_tampered_array(monkeypatch, tmp_path):
    x, y = _arrays(56)

    class FakeTask:
        pass

    task = FakeTask()
    task.x, task.y = x, y
    monkeypatch.setattr(design_bench, "_make_task", lambda spec: task)
    dataset = tmp_path / "dataset"
    design_bench.export_dataset(design_bench.DKITTY_TASK, dataset)
    changed = x.copy()
    changed[0, 0] += 1.0
    np.save(dataset / "x.npy", changed, allow_pickle=False)
    with pytest.raises(ValueError, match="manifest"):
        design_bench._load_dataset(dataset)


def test_direct_execution_does_not_shadow_the_design_bench_package():
    """Running this script must not make its own file satisfy ``import design_bench``.

    ``scripts/design_bench.py`` shares a name with the installed package it
    calls, so executing it directly would otherwise resolve ``design_bench`` to
    itself and lose ``design_bench.make``.
    """
    probe = (
        "import runpy, sys;"
        "sys.argv = ['design_bench.py'];"
        "runpy.run_path({script!r}, run_name='not_main');"
        "print(str(sys.path))"
    ).format(script=str(SCRIPT))
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(SCRIPT.parent),
        capture_output=True,
        text=True,
        check=True,
    )
    remaining = ast.literal_eval(completed.stdout.strip())
    assert all(
        Path(entry or ".").resolve() != SCRIPT.parent for entry in remaining
    ), "script directory still on sys.path; import design_bench would self-shadow"
