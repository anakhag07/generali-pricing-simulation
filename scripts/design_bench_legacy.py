#!/usr/bin/env python3
"""Python-3.7-compatible worker for the pinned Design-Bench environment.

This file is intentionally standalone: the main project requires modern Python,
so importing project modules here would break the legacy environment seam.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np


ANT_TASK = "AntMorphology-Exact-v0"
DKITTY_TASK = "DKittyMorphology-Exact-v0"
TASK_DIMENSIONS = {ANT_TASK: 60, DKITTY_TASK: 56}
SCHEMA_VERSION = 1
EXPECTED_DESIGN_BENCH_VERSION = "2.0.20"
DESIGN_BASELINES_COMMIT = "785dbcfa58107bfcc426257a1c2e69d7f71c3c27"


def _task_payload(name):
    if name not in TASK_DIMENSIONS:
        raise ValueError("Unsupported Design-Bench task {!r}.".format(name))
    return {
        "name": name,
        "dimension": TASK_DIMENSIONS[name],
        "task_kwargs": {"relabel": False},
    }


def _package_version(name):
    try:
        import pkg_resources

        return pkg_resources.get_distribution(name).version
    except Exception:
        return "unknown"


def _environment(include_baseline=False):
    packages = {
        "design_bench": _package_version("design-bench"),
        "numpy": _package_version("numpy"),
    }
    if include_baseline:
        packages.update(
            {
                "design_baselines": _package_version("design-baselines"),
                "morphing_agents": _package_version("morphing-agents"),
                "tensorflow": _package_version("tensorflow"),
                "tensorflow_probability": _package_version("tensorflow-probability"),
            }
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "expected_design_baselines_commit": DESIGN_BASELINES_COMMIT,
    }


def _require_pinned_design_bench():
    version = _package_version("design-bench")
    if version != EXPECTED_DESIGN_BENCH_VERSION:
        raise RuntimeError(
            "Expected design-bench=={}, found {}. Use the pinned legacy environment.".format(
                EXPECTED_DESIGN_BENCH_VERSION, version
            )
        )
    return version


def _array_sha256(array):
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _array_record(filename, array):
    return {
        "path": filename,
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "sha256": _array_sha256(array),
    }


def _dataset_manifest_id(manifest):
    identity = {
        "schema_version": manifest.get("schema_version"),
        "artifact_type": manifest.get("artifact_type"),
        "task": manifest.get("task"),
        "design_bench_version": manifest.get("design_bench_version"),
        "arrays": manifest.get("arrays"),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Artifact metadata must be a JSON object: {}".format(path))
    return payload


def _prepare_output(directory):
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise ValueError("Output directory is not empty: {}".format(root))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_recorded_array(root, record, name):
    relative = Path(str(record.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid {} array path: {}".format(name, relative))
    array = np.load(str(root / relative), allow_pickle=False)
    if list(array.shape) != record.get("shape"):
        raise ValueError("{} shape does not match the manifest.".format(name))
    if array.dtype.str != record.get("dtype"):
        raise ValueError("{} dtype does not match the manifest.".format(name))
    if _array_sha256(array) != record.get("sha256"):
        raise ValueError("{} checksum does not match the manifest.".format(name))
    return array


def _validate_candidates(task, candidates):
    expected_dim = TASK_DIMENSIONS[task["name"]]
    if candidates.ndim != 2 or candidates.shape[1] != expected_dim or candidates.shape[0] == 0:
        raise ValueError(
            "Candidates for {} must have shape (n, {}).".format(task["name"], expected_dim)
        )
    if not np.issubdtype(candidates.dtype, np.floating):
        raise ValueError("Morphology candidates must have a floating dtype.")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("Candidates must contain only finite values.")


def _validate_dataset_arrays(task, x, y):
    _validate_candidates(task, x)
    if y.shape != (x.shape[0], 1):
        raise ValueError("task.y must have shape (n, 1).")
    if x.shape[0] <= 200:
        raise ValueError("Dataset must contain more than 200 rows.")
    if not np.issubdtype(y.dtype, np.floating) or not np.all(np.isfinite(y)):
        raise ValueError("task.y must contain finite floating values.")


def _verify_live_task(task_payload, x, y):
    live_x = np.asarray(x)
    live_y = np.asarray(y)
    _validate_dataset_arrays(task_payload, live_x, live_y)
    return live_x, live_y


def _load_dataset_artifact(directory):
    root = Path(directory)
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported dataset artifact schema version.")
    if manifest.get("artifact_type") != "design_bench_dataset":
        raise ValueError("Not a Design-Bench dataset artifact.")
    task = _task_payload(manifest.get("task", {}).get("name"))
    if manifest.get("task") != task:
        raise ValueError("Dataset task metadata does not match the fixed task specification.")
    arrays = manifest.get("arrays", {})
    x = _load_recorded_array(root, arrays.get("x", {}), "x")
    y = _load_recorded_array(root, arrays.get("y", {}), "y")
    _validate_dataset_arrays(task, x, y)
    expected_id = _dataset_manifest_id(manifest)
    if manifest.get("manifest_id") != expected_id:
        raise ValueError("Dataset manifest_id does not match its contents.")
    if manifest.get("design_bench_version") != EXPECTED_DESIGN_BENCH_VERSION:
        raise ValueError("Dataset was not exported by the pinned Design-Bench version.")
    return root, manifest, task, x, y


def _make_design_bench_task(task_payload):
    import design_bench

    return design_bench.make(task_payload["name"], **task_payload["task_kwargs"])


def _assert_live_dataset_matches(manifest, task_payload, live_task):
    live_x, live_y = _verify_live_task(task_payload, live_task.x, live_task.y)
    records = manifest["arrays"]
    if _array_sha256(live_x) != records["x"]["sha256"]:
        raise ValueError("Live task.x does not match the exported dataset manifest.")
    if _array_sha256(live_y) != records["y"]["sha256"]:
        raise ValueError("Live task.y does not match the exported dataset manifest.")
    if live_x.dtype.str != records["x"]["dtype"] or list(live_x.shape) != records["x"]["shape"]:
        raise ValueError("Live task.x metadata does not match the exported dataset manifest.")
    if live_y.dtype.str != records["y"]["dtype"] or list(live_y.shape) != records["y"]["shape"]:
        raise ValueError("Live task.y metadata does not match the exported dataset manifest.")


def export_dataset(task_name, output):
    version = _require_pinned_design_bench()
    task_payload = _task_payload(task_name)
    task = _make_design_bench_task(task_payload)
    x, y = _verify_live_task(task_payload, task.x, task.y)
    root = _prepare_output(output)
    np.save(str(root / "x.npy"), x, allow_pickle=False)
    np.save(str(root / "y.npy"), y, allow_pickle=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "design_bench_dataset",
        "task": task_payload,
        "design_bench_version": version,
        "arrays": {
            "x": _array_record("x.npy", x),
            "y": _array_record("y.npy", y),
        },
        "environment": _environment(),
    }
    manifest["manifest_id"] = _dataset_manifest_id(manifest)
    _write_json(root / "manifest.json", manifest)


def evaluate(dataset, candidates, output):
    _require_pinned_design_bench()
    _, manifest, task_payload, _, _ = _load_dataset_artifact(dataset)
    task = _make_design_bench_task(task_payload)
    _assert_live_dataset_matches(manifest, task_payload, task)
    candidate_array = np.asarray(np.load(str(candidates), allow_pickle=False))
    _validate_candidates(task_payload, candidate_array)
    scores = np.asarray(task.predict(candidate_array))
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]
    if scores.shape != (candidate_array.shape[0], 1):
        raise ValueError("Exact oracle returned shape {}; expected (n, 1).".format(scores.shape))
    if not np.issubdtype(scores.dtype, np.floating) or not np.all(np.isfinite(scores)):
        raise ValueError("Exact oracle returned non-finite or non-floating scores.")

    root = _prepare_output(output)
    np.save(str(root / "candidates.npy"), candidate_array, allow_pickle=False)
    np.save(str(root / "scores.npy"), scores, allow_pickle=False)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "design_bench_oracle_evaluation",
        "task": task_payload,
        "dataset_manifest_id": manifest["manifest_id"],
        "arrays": {
            "candidates": _array_record("candidates.npy", candidate_array),
            "scores": _array_record("scores.npy", scores),
        },
        "environment": _environment(),
    }
    _write_json(root / "evaluation.json", metadata)


def _reference_config(task_payload, output):
    return {
        "logging_dir": os.fspath(Path(output).resolve()),
        "task": task_payload["name"],
        "task_kwargs": task_payload["task_kwargs"],
        "normalize_ys": True,
        "normalize_xs": True,
        "model_noise_std": 0.0,
        "val_size": 200,
        "batch_size": 128,
        "epochs": 100,
        "activations": [["leaky_relu", "leaky_relu"]],
        "hidden_size": 2048,
        "initial_max_std": 0.2,
        "initial_min_std": 0.1,
        "forward_model_lr": 0.0003,
        "aggregation_method": "mean",
        "solver_samples": 128,
        "do_evaluation": False,
        "solver_lr": 0.01,
        "solver_steps": 200,
    }


def _config_for_mode(task_payload, output, mode):
    config = _reference_config(task_payload, output)
    if mode == "smoke":
        config.update({"epochs": 1, "solver_samples": 2, "solver_steps": 1})
    elif mode != "reference":
        raise ValueError("mode must be 'reference' or 'smoke'.")
    return config


def _set_baseline_seed(seed, tensorflow_module):
    random.seed(seed)
    np.random.seed(seed)
    tensorflow_module.random.set_seed(seed)


def _denormalize_solution(task, normalized_solution):
    """Convert the official normalized solution back to raw task coordinates."""
    return np.asarray(task.denormalize_x(normalized_solution))


def run_gradient_ascent(dataset, output, mode, seed):
    _require_pinned_design_bench()
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    _, manifest, task_payload, _, _ = _load_dataset_artifact(dataset)

    import tensorflow as tf
    from design_baselines.data import StaticGraphTask
    import design_baselines.gradient_ascent as gradient_ascent_module

    verification_task = StaticGraphTask(task_payload["name"], **task_payload["task_kwargs"])
    _assert_live_dataset_matches(manifest, task_payload, verification_task)

    root = _prepare_output(output)
    _set_baseline_seed(seed, tf)
    config = _config_for_mode(task_payload, root, mode)
    requested_config = copy.deepcopy(config)

    # The upstream function constructs its own task. Intercept only that
    # construction so the exact arrays are verified immediately before its
    # normalization and training code consumes them.
    original_task_class = gradient_ascent_module.StaticGraphTask

    def verified_training_task(task_name, **task_kwargs):
        task = original_task_class(task_name, **task_kwargs)
        _assert_live_dataset_matches(manifest, task_payload, task)
        return task

    gradient_ascent_module.StaticGraphTask = verified_training_task
    try:
        gradient_ascent_module.gradient_ascent(config)
    finally:
        gradient_ascent_module.StaticGraphTask = original_task_class

    solution_path = root / "solution.npy"
    normalized_solution = np.asarray(np.load(str(solution_path), allow_pickle=False))
    _validate_candidates(task_payload, normalized_solution)

    normalization_task = StaticGraphTask(task_payload["name"], **task_payload["task_kwargs"])
    _assert_live_dataset_matches(manifest, task_payload, normalization_task)
    normalization_task.map_normalize_x()
    raw_candidates = _denormalize_solution(normalization_task, normalized_solution)
    _validate_candidates(task_payload, raw_candidates)
    np.save(str(root / "candidates.npy"), raw_candidates, allow_pickle=False)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "design_baselines_gradient_ascent_run",
        "method": "gradient_ascent",
        "task": task_payload,
        "dataset_manifest_id": manifest["manifest_id"],
        "mode": mode,
        "seed": seed,
        "config": requested_config,
        "effective_solver_lr": config["solver_lr"],
        "f_hat": {
            "implementation": "design_baselines.gradient_ascent.ForwardModel",
            "models": 1,
            "hidden_layers": [2048, 2048],
            "activation": "leaky_relu",
            "output_distribution": "Normal(mean, learned_std)",
            "loss": "gaussian_negative_log_likelihood",
            "validation_rows": 200,
            "normalizes_x": True,
            "normalizes_y": True,
            "training_source_checksum_verified": True,
        },
        "arrays": {
            "normalized_solution": _array_record("solution.npy", normalized_solution),
            "raw_candidates": _array_record("candidates.npy", raw_candidates),
        },
        "logs": {"format": "tensorflow_summary", "path": "."},
        "environment": _environment(include_baseline=True),
    }
    _write_json(root / "run.json", metadata)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    export = subparsers.add_parser("export-dataset")
    export.add_argument("--task", required=True)
    export.add_argument("--output", required=True)

    oracle = subparsers.add_parser("evaluate")
    oracle.add_argument("--dataset", required=True)
    oracle.add_argument("--candidates", required=True)
    oracle.add_argument("--output", required=True)

    baseline = subparsers.add_parser("run-gradient-ascent")
    baseline.add_argument("--dataset", required=True)
    baseline.add_argument("--output", required=True)
    baseline.add_argument("--mode", choices=("reference", "smoke"), required=True)
    baseline.add_argument("--seed", type=int, required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "export-dataset":
        export_dataset(args.task, args.output)
    elif args.command == "evaluate":
        evaluate(args.dataset, args.candidates, args.output)
    elif args.command == "run-gradient-ascent":
        run_gradient_ascent(args.dataset, args.output, args.mode, args.seed)
    else:
        raise SystemExit("A command is required.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("design_bench_legacy.py: {}".format(exc), file=sys.stderr)
        raise
