from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

import benchmarks.design_bench as design_bench_module
from benchmarks.design_bench import (
    ANT_TASK,
    DKITTY_TASK,
    BaselineRunArtifact,
    DatasetArtifact,
    DesignBenchBridge,
    DesignBenchBridgeError,
    DesignBenchTaskSpec,
    EvaluationArtifact,
)


def _arrays(dimension: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(201 * dimension, dtype=np.float32).reshape(201, dimension) / 100.0
    y = np.linspace(-1.0, 1.0, 201, dtype=np.float32)[:, None]
    return x, y


def _write_dataset(path: Path, task_name: str = ANT_TASK) -> DatasetArtifact:
    spec = DesignBenchTaskSpec(task_name)
    x, y = _arrays(spec.dimension)
    return DatasetArtifact.write(
        path,
        task=spec,
        x=x,
        y=y,
        design_bench_version="2.0.20",
        environment={"python": "fixture"},
    )


def _array_record(path: Path, array: np.ndarray) -> dict[str, object]:
    np.save(path, array, allow_pickle=False)
    digest = hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()
    return {
        "path": path.name,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": digest,
    }


def _load_legacy_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "design_bench_legacy.py"
    module_spec = importlib.util.spec_from_file_location("design_bench_legacy_test", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_task_spec_supports_only_ant_and_dkitty() -> None:
    assert DesignBenchTaskSpec(ANT_TASK).dimension == 60
    assert DesignBenchTaskSpec(DKITTY_TASK).dimension == 56
    assert DesignBenchTaskSpec(ANT_TASK).task_kwargs == {"relabel": False}

    with pytest.raises(ValueError, match="Unsupported Design-Bench task"):
        DesignBenchTaskSpec("HopperController-Exact-v0")


def test_dataset_artifact_round_trip_has_stable_identity(tmp_path: Path) -> None:
    first = _write_dataset(tmp_path / "first")
    spec = DesignBenchTaskSpec(ANT_TASK)
    x, y = _arrays(spec.dimension)
    second = DatasetArtifact.write(
        tmp_path / "second",
        task=spec,
        x=x.copy(),
        y=y.copy(),
        design_bench_version="2.0.20",
        environment={"python": "a different host"},
    )

    assert first.manifest_id == second.manifest_id
    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.y, second.y)
    assert first.manifest_id.startswith("sha256:")


def test_dataset_artifact_rejects_checksum_mismatch(tmp_path: Path) -> None:
    artifact = _write_dataset(tmp_path / "dataset")
    changed = artifact.x.copy()
    changed[0, 0] += 1.0
    np.save(artifact.directory / "x.npy", changed, allow_pickle=False)

    with pytest.raises(ValueError, match="checksum"):
        DatasetArtifact.load(artifact.directory)


def test_dataset_artifact_rejects_wrong_task_shape(tmp_path: Path) -> None:
    spec = DesignBenchTaskSpec(DKITTY_TASK)
    x, y = _arrays(60)

    with pytest.raises(ValueError, match=r"shape \(n, 56\)"):
        DatasetArtifact.write(
            tmp_path / "dataset",
            task=spec,
            x=x,
            y=y,
            design_bench_version="2.0.20",
        )


def test_bridge_validates_exported_task(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_run(command, **kwargs):
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        output = Path(command[command.index("--output") + 1])
        _write_dataset(output, ANT_TASK)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(design_bench_module.subprocess, "run", fake_run)
    artifact = DesignBenchBridge("legacy-python").export_dataset(
        DesignBenchTaskSpec(ANT_TASK), tmp_path / "dataset"
    )

    assert artifact.task.name == ANT_TASK


def test_bridge_propagates_external_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 7, "", "legacy dependency missing")

    monkeypatch.setattr(design_bench_module.subprocess, "run", fake_run)
    bridge = DesignBenchBridge("legacy-python")

    with pytest.raises(DesignBenchBridgeError, match="legacy dependency missing"):
        bridge.export_dataset(DesignBenchTaskSpec(ANT_TASK), tmp_path / "dataset")


def test_bridge_rejects_invalid_candidates_before_subprocess(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset", DKITTY_TASK)
    bridge = DesignBenchBridge("legacy-python")

    with pytest.raises(ValueError, match=r"shape \(n, 56\)"):
        bridge.evaluate(dataset, np.ones((2, 60), dtype=np.float32), tmp_path / "evaluation")


def test_evaluation_artifact_preserves_every_raw_score(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset", DKITTY_TASK)
    output = tmp_path / "evaluation"
    output.mkdir()
    candidates = dataset.x[:3].copy()
    scores = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
    metadata = {
        "schema_version": 1,
        "artifact_type": "design_bench_oracle_evaluation",
        "task": dataset.task.to_dict(),
        "dataset_manifest_id": dataset.manifest_id,
        "arrays": {
            "candidates": _array_record(output / "candidates.npy", candidates),
            "scores": _array_record(output / "scores.npy", scores),
        },
    }
    (output / "evaluation.json").write_text(json.dumps(metadata), encoding="utf-8")

    artifact = EvaluationArtifact.load(output)

    np.testing.assert_array_equal(artifact.candidates, candidates)
    np.testing.assert_array_equal(artifact.scores, scores)
    assert artifact.dataset_manifest_id == dataset.manifest_id


def test_baseline_artifact_preserves_normalized_and_raw_designs(tmp_path: Path) -> None:
    dataset = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "baseline"
    output.mkdir()
    normalized = np.zeros((2, dataset.task.dimension), dtype=np.float32)
    raw = np.full_like(normalized, 2.5)
    metadata = {
        "schema_version": 1,
        "artifact_type": "design_baselines_gradient_ascent_run",
        "method": "gradient_ascent",
        "task": dataset.task.to_dict(),
        "dataset_manifest_id": dataset.manifest_id,
        "mode": "smoke",
        "seed": 17,
        "arrays": {
            "normalized_solution": _array_record(output / "solution.npy", normalized),
            "raw_candidates": _array_record(output / "candidates.npy", raw),
        },
    }
    (output / "run.json").write_text(json.dumps(metadata), encoding="utf-8")

    artifact = BaselineRunArtifact.load(output)

    assert artifact.mode == "smoke"
    assert artifact.seed == 17
    np.testing.assert_array_equal(artifact.normalized_solution, normalized)
    np.testing.assert_array_equal(artifact.raw_candidates, raw)


def test_legacy_export_and_evaluation_use_raw_official_task_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_legacy_module()
    x, y = _arrays(60)
    seen: list[np.ndarray] = []

    class FakeTask:
        def __init__(self):
            self.x = x
            self.y = y

        def predict(self, candidates):
            seen.append(np.asarray(candidates).copy())
            return np.sum(candidates, axis=1, keepdims=True)

    monkeypatch.setattr(module, "_require_pinned_design_bench", lambda: "2.0.20")
    monkeypatch.setattr(module, "_make_design_bench_task", lambda task: FakeTask())

    dataset_dir = tmp_path / "dataset"
    module.export_dataset(ANT_TASK, dataset_dir)
    dataset = DatasetArtifact.load(dataset_dir)
    candidates_path = tmp_path / "proposals.npy"
    candidates = dataset.x[:2].copy()
    np.save(candidates_path, candidates, allow_pickle=False)

    output = tmp_path / "evaluation"
    module.evaluate(dataset_dir, candidates_path, output)
    evaluation = EvaluationArtifact.load(output)

    np.testing.assert_array_equal(seen[0], candidates)
    np.testing.assert_array_equal(evaluation.candidates, candidates)
    np.testing.assert_array_equal(
        evaluation.scores,
        np.sum(candidates, axis=1, keepdims=True),
    )


def test_legacy_runner_denormalizes_solution_in_task_coordinates() -> None:
    module = _load_legacy_module()

    class FakeNormalizedTask:
        def denormalize_x(self, values):
            return values * 3.0 + 2.0

    normalized = np.asarray([[0.0, 1.0]], dtype=np.float32)
    raw = module._denormalize_solution(FakeNormalizedTask(), normalized)

    np.testing.assert_array_equal(raw, np.asarray([[2.0, 5.0]], dtype=np.float32))


def test_manifest_id_rejects_metadata_tampering(tmp_path: Path) -> None:
    artifact = _write_dataset(tmp_path / "dataset")
    manifest_path = artifact.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["design_bench_version"] = "different"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest_id"):
        DatasetArtifact.load(artifact.directory)
