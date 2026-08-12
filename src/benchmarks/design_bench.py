"""Artifact and subprocess seam for the legacy Design-Bench environment.

This module deliberately does not import ``design_bench``, TensorFlow, or
``design_baselines``. Those packages run in a separate Python environment via
``scripts/design_bench_legacy.py``; this module validates the files that cross
that process seam.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Literal, Mapping, cast

import numpy as np


ANT_TASK = "AntMorphology-Exact-v0"
DKITTY_TASK = "DKittyMorphology-Exact-v0"
SUPPORTED_TASK_DIMENSIONS: dict[str, int] = {
    ANT_TASK: 60,
    DKITTY_TASK: 56,
}
SCHEMA_VERSION = 1
BaselineMode = Literal["reference", "smoke"]


class DesignBenchBridgeError(RuntimeError):
    """Raised when the external runner fails or returns an invalid artifact."""


@dataclass(frozen=True)
class DesignBenchTaskSpec:
    """One supported raw continuous Design-Bench task."""

    name: str

    def __post_init__(self) -> None:
        if self.name not in SUPPORTED_TASK_DIMENSIONS:
            supported = ", ".join(sorted(SUPPORTED_TASK_DIMENSIONS))
            raise ValueError(
                f"Unsupported Design-Bench task {self.name!r}; choose one of {supported}."
            )

    @property
    def dimension(self) -> int:
        return SUPPORTED_TASK_DIMENSIONS[self.name]

    @property
    def task_kwargs(self) -> dict[str, bool]:
        return {"relabel": False}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "task_kwargs": self.task_kwargs,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DesignBenchTaskSpec:
        spec = cls(str(payload.get("name", "")))
        if payload.get("dimension") != spec.dimension:
            raise ValueError("Task dimension does not match the supported Design-Bench task.")
        if payload.get("task_kwargs") != spec.task_kwargs:
            raise ValueError("Only the fixed task arguments {'relabel': False} are supported.")
        return spec


@dataclass(frozen=True)
class DatasetArtifact:
    """Verified raw ``task.x``/``task.y`` arrays and their stable identity."""

    directory: Path
    task: DesignBenchTaskSpec
    manifest_id: str
    design_bench_version: str
    x: np.ndarray
    y: np.ndarray
    manifest: dict[str, Any]

    @classmethod
    def load(cls, directory: str | Path) -> DatasetArtifact:
        root = Path(directory)
        manifest = _read_json(root / "manifest.json")
        _require_artifact_header(manifest, "design_bench_dataset")
        task = DesignBenchTaskSpec.from_dict(_mapping(manifest, "task"))
        arrays = _mapping(manifest, "arrays")
        x = _load_array(root, _mapping(arrays, "x"), name="x")
        y = _load_array(root, _mapping(arrays, "y"), name="y")
        _validate_dataset_arrays(task, x, y)

        expected_id = _dataset_manifest_id(manifest)
        if manifest.get("manifest_id") != expected_id:
            raise ValueError("Dataset manifest_id does not match its task, version, and arrays.")
        return cls(
            directory=root,
            task=task,
            manifest_id=expected_id,
            design_bench_version=str(manifest.get("design_bench_version", "")),
            x=x,
            y=y,
            manifest=manifest,
        )

    @classmethod
    def write(
        cls,
        directory: str | Path,
        *,
        task: DesignBenchTaskSpec,
        x: np.ndarray,
        y: np.ndarray,
        design_bench_version: str,
        environment: Mapping[str, Any] | None = None,
    ) -> DatasetArtifact:
        """Write a raw dataset artifact; primarily used by adapters and fixtures."""
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        _validate_dataset_arrays(task, x_arr, y_arr)
        root = _prepare_empty_directory(directory)
        np.save(root / "x.npy", x_arr, allow_pickle=False)
        np.save(root / "y.npy", y_arr, allow_pickle=False)
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": "design_bench_dataset",
            "task": task.to_dict(),
            "design_bench_version": str(design_bench_version),
            "arrays": {
                "x": _array_record("x.npy", x_arr),
                "y": _array_record("y.npy", y_arr),
            },
            "environment": dict(environment or {}),
        }
        manifest["manifest_id"] = _dataset_manifest_id(manifest)
        _write_json(root / "manifest.json", manifest)
        return cls.load(root)


@dataclass(frozen=True)
class EvaluationArtifact:
    """Raw candidates and their exact-oracle scores."""

    directory: Path
    task: DesignBenchTaskSpec
    dataset_manifest_id: str
    candidates: np.ndarray
    scores: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def load(cls, directory: str | Path) -> EvaluationArtifact:
        root = Path(directory)
        metadata = _read_json(root / "evaluation.json")
        _require_artifact_header(metadata, "design_bench_oracle_evaluation")
        task = DesignBenchTaskSpec.from_dict(_mapping(metadata, "task"))
        arrays = _mapping(metadata, "arrays")
        candidates = _load_array(root, _mapping(arrays, "candidates"), name="candidates")
        scores = _load_array(root, _mapping(arrays, "scores"), name="scores")
        _validate_candidates(task, candidates)
        _validate_scores(scores, candidates.shape[0])
        manifest_id = str(metadata.get("dataset_manifest_id", ""))
        if not manifest_id.startswith("sha256:"):
            raise ValueError("Evaluation is missing a valid dataset_manifest_id.")
        return cls(root, task, manifest_id, candidates, scores, metadata)


@dataclass(frozen=True)
class BaselineRunArtifact:
    """Official gradient-ascent output in normalized and raw coordinates."""

    directory: Path
    task: DesignBenchTaskSpec
    dataset_manifest_id: str
    mode: BaselineMode
    seed: int
    normalized_solution: np.ndarray
    raw_candidates: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def load(cls, directory: str | Path) -> BaselineRunArtifact:
        root = Path(directory)
        metadata = _read_json(root / "run.json")
        _require_artifact_header(metadata, "design_baselines_gradient_ascent_run")
        if metadata.get("method") != "gradient_ascent":
            raise ValueError("Only the official gradient_ascent baseline is supported.")
        task = DesignBenchTaskSpec.from_dict(_mapping(metadata, "task"))
        mode = str(metadata.get("mode", ""))
        if mode not in ("reference", "smoke"):
            raise ValueError(f"Invalid baseline mode {mode!r}.")
        arrays = _mapping(metadata, "arrays")
        solution = _load_array(root, _mapping(arrays, "normalized_solution"), name="solution")
        candidates = _load_array(root, _mapping(arrays, "raw_candidates"), name="candidates")
        _validate_candidates(task, solution)
        _validate_candidates(task, candidates)
        if solution.shape != candidates.shape:
            raise ValueError("Normalized and raw baseline candidates must have matching shapes.")
        manifest_id = str(metadata.get("dataset_manifest_id", ""))
        if not manifest_id.startswith("sha256:"):
            raise ValueError("Baseline run is missing a valid dataset_manifest_id.")
        return cls(
            root,
            task,
            manifest_id,
            cast(BaselineMode, mode),
            int(metadata["seed"]),
            solution,
            candidates,
            metadata,
        )


@dataclass(frozen=True)
class DesignBenchBridge:
    """Run Design-Bench operations through one isolated Python executable."""

    python_executable: str | Path
    legacy_script: str | Path | None = None

    def export_dataset(
        self,
        task: DesignBenchTaskSpec,
        output_dir: str | Path,
    ) -> DatasetArtifact:
        self._run("export-dataset", "--task", task.name, "--output", os.fspath(output_dir))
        artifact = DatasetArtifact.load(output_dir)
        if artifact.task != task:
            raise DesignBenchBridgeError(
                "External runner exported a different task than requested."
            )
        return artifact

    def evaluate(
        self,
        dataset: DatasetArtifact,
        candidates: np.ndarray | str | Path,
        output_dir: str | Path,
    ) -> EvaluationArtifact:
        if isinstance(candidates, (str, Path)):
            self._evaluate_path(dataset, Path(candidates), output_dir)
        else:
            candidate_array = np.asarray(candidates)
            _validate_candidates(dataset.task, candidate_array)
            with tempfile.TemporaryDirectory(prefix="design-bench-candidates-") as tmp:
                path = Path(tmp) / "candidates.npy"
                np.save(path, candidate_array, allow_pickle=False)
                self._evaluate_path(dataset, path, output_dir)
        artifact = EvaluationArtifact.load(output_dir)
        if artifact.dataset_manifest_id != dataset.manifest_id:
            raise DesignBenchBridgeError("Oracle evaluation used a different dataset manifest.")
        return artifact

    def run_gradient_ascent(
        self,
        dataset: DatasetArtifact,
        output_dir: str | Path,
        *,
        mode: BaselineMode,
        seed: int,
    ) -> BaselineRunArtifact:
        if mode not in ("reference", "smoke"):
            raise ValueError("mode must be 'reference' or 'smoke'.")
        if seed < 0:
            raise ValueError("seed must be non-negative.")
        self._run(
            "run-gradient-ascent",
            "--dataset",
            os.fspath(dataset.directory),
            "--output",
            os.fspath(output_dir),
            "--mode",
            mode,
            "--seed",
            str(seed),
        )
        artifact = BaselineRunArtifact.load(output_dir)
        if artifact.dataset_manifest_id != dataset.manifest_id:
            raise DesignBenchBridgeError("Baseline run used a different dataset manifest.")
        if artifact.mode != mode or artifact.seed != seed:
            raise DesignBenchBridgeError(
                "Baseline run metadata does not match the requested mode and seed."
            )
        return artifact

    def _evaluate_path(
        self,
        dataset: DatasetArtifact,
        candidates_path: Path,
        output_dir: str | Path,
    ) -> None:
        self._run(
            "evaluate",
            "--dataset",
            os.fspath(dataset.directory),
            "--candidates",
            os.fspath(candidates_path),
            "--output",
            os.fspath(output_dir),
        )

    def _run(self, *args: str) -> None:
        script = (
            Path(self.legacy_script)
            if self.legacy_script is not None
            else Path(__file__).resolve().parents[2] / "scripts" / "design_bench_legacy.py"
        )
        command = [os.fspath(self.python_executable), os.fspath(script), *args]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no runner output"
            raise DesignBenchBridgeError(
                f"Design-Bench runner failed with exit code {completed.returncode}: {detail}"
            )


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _array_record(filename: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": filename,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": _array_sha256(array),
    }


def _dataset_manifest_id(manifest: Mapping[str, Any]) -> str:
    identity = {
        "schema_version": manifest.get("schema_version"),
        "artifact_type": manifest.get("artifact_type"),
        "task": manifest.get("task"),
        "design_bench_version": manifest.get("design_bench_version"),
        "arrays": manifest.get("arrays"),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Artifact field {key!r} must be an object.")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read artifact metadata {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Artifact metadata {path} must contain a JSON object.")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _prepare_empty_directory(directory: str | Path) -> Path:
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Artifact output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _require_artifact_header(payload: Mapping[str, Any], artifact_type: str) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported artifact schema version {payload.get('schema_version')!r}.")
    if payload.get("artifact_type") != artifact_type:
        raise ValueError(f"Expected artifact_type={artifact_type!r}.")


def _load_array(root: Path, record: Mapping[str, Any], *, name: str) -> np.ndarray:
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid {name} artifact path {relative}.")
    try:
        array = np.load(root / relative, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load {name} array: {exc}") from exc
    expected_shape = tuple(record.get("shape", ()))
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape {array.shape} does not match manifest {expected_shape}.")
    if array.dtype.str != record.get("dtype"):
        raise ValueError(f"{name} dtype {array.dtype.str} does not match manifest.")
    if _array_sha256(array) != record.get("sha256"):
        raise ValueError(f"{name} checksum does not match manifest.")
    return array


def _validate_dataset_arrays(task: DesignBenchTaskSpec, x: np.ndarray, y: np.ndarray) -> None:
    _validate_candidates(task, x)
    _validate_scores(y, x.shape[0])
    if x.shape[0] <= 200:
        raise ValueError("Design-Bench dataset must contain more than the 200 validation examples.")


def _validate_candidates(task: DesignBenchTaskSpec, candidates: np.ndarray) -> None:
    if candidates.ndim != 2 or candidates.shape[1] != task.dimension:
        raise ValueError(
            f"Candidates for {task.name} must have shape (n, {task.dimension}); "
            f"received {candidates.shape}."
        )
    if candidates.shape[0] == 0:
        raise ValueError("At least one candidate is required.")
    if not np.issubdtype(candidates.dtype, np.floating):
        raise ValueError("Design-Bench morphology candidates must have a floating dtype.")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("Candidates must contain only finite values.")


def _validate_scores(scores: np.ndarray, expected_rows: int) -> None:
    if scores.shape != (expected_rows, 1):
        raise ValueError(f"Scores must have shape ({expected_rows}, 1); received {scores.shape}.")
    if not np.issubdtype(scores.dtype, np.floating):
        raise ValueError("Scores must have a floating dtype.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Scores must contain only finite values.")
