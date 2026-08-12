#!/usr/bin/env python3
"""Thin Design-Bench data, oracle, and gradient-ascent baseline runner.

Run this script with the pinned Design-Baselines Python 3.7 environment. Legacy
packages are imported inside commands, so they never enter the main project.
Surrogate design, optimizer integration, and comparison protocols are deferred.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np


ANT_TASK = "AntMorphology-Exact-v0"
DKITTY_TASK = "DKittyMorphology-Exact-v0"
TASK_DIMENSIONS = {ANT_TASK: 60, DKITTY_TASK: 56}
DESIGN_BENCH_VERSION = "2.0.20"
DESIGN_BASELINES_COMMIT = "785dbcfa58107bfcc426257a1c2e69d7f71c3c27"


def _sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def _task_spec(task_name):
    if task_name not in TASK_DIMENSIONS:
        raise ValueError("Unsupported task {!r}.".format(task_name))
    return {"name": task_name, "dimension": TASK_DIMENSIONS[task_name], "relabel": False}


def _validate_xy(spec, x, y):
    if x.ndim != 2 or x.shape[1] != spec["dimension"]:
        raise ValueError("task.x has the wrong shape for {}.".format(spec["name"]))
    if y.shape != (x.shape[0], 1):
        raise ValueError("task.y must have shape (n, 1).")
    if x.shape[0] <= 200 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Design-Bench arrays are empty, too small, or non-finite.")


def _validate_candidates(spec, candidates):
    if candidates.ndim != 2 or candidates.shape[1] != spec["dimension"]:
        raise ValueError("Candidates must have shape (n, {}).".format(spec["dimension"]))
    if candidates.shape[0] == 0 or not np.all(np.isfinite(candidates)):
        raise ValueError("Candidates must be non-empty and finite.")


def _identity(spec, x, y):
    return {
        "task": spec,
        "design_bench_version": DESIGN_BENCH_VERSION,
        "x": {"shape": list(x.shape), "dtype": x.dtype.str, "sha256": _sha256(x)},
        "y": {"shape": list(y.shape), "dtype": y.dtype.str, "sha256": _sha256(y)},
    }


def _manifest_id(identity):
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _empty_output(path):
    output = Path(path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    return output


def _installed_design_bench_version():
    import pkg_resources

    return pkg_resources.get_distribution("design-bench").version


def _make_task(spec):
    if _installed_design_bench_version() != DESIGN_BENCH_VERSION:
        raise RuntimeError(
            "This integration requires design-bench=={}.".format(DESIGN_BENCH_VERSION)
        )
    import design_bench

    return design_bench.make(spec["name"], relabel=False)


def _load_dataset(path):
    root = Path(path)
    with (root / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    spec = _task_spec(manifest["task"]["name"])
    x = np.load(str(root / "x.npy"), allow_pickle=False)
    y = np.load(str(root / "y.npy"), allow_pickle=False)
    _validate_xy(spec, x, y)
    identity = _identity(spec, x, y)
    if (
        manifest.get("identity") != identity
        or manifest.get("manifest_id") != _manifest_id(identity)
    ):
        raise ValueError("Dataset arrays or metadata do not match the manifest.")
    return manifest, spec, x, y


def _verified_task(manifest, spec, x, y):
    task = _make_task(spec)
    _verify_task_arrays(manifest, spec, x, y, task)
    return task


def _verify_task_arrays(manifest, spec, x, y, task):
    live_x, live_y = np.asarray(task.x), np.asarray(task.y)
    _validate_xy(spec, live_x, live_y)
    if _sha256(live_x) != _sha256(x) or _sha256(live_y) != _sha256(y):
        raise ValueError("Installed task.x/task.y differ from the exported dataset.")
    if manifest["manifest_id"] != _manifest_id(_identity(spec, live_x, live_y)):
        raise ValueError("Installed task does not match the dataset manifest.")


def export_dataset(task_name, output):
    spec = _task_spec(task_name)
    task = _make_task(spec)
    x, y = np.asarray(task.x), np.asarray(task.y)
    _validate_xy(spec, x, y)
    root = _empty_output(output)
    np.save(str(root / "x.npy"), x, allow_pickle=False)
    np.save(str(root / "y.npy"), y, allow_pickle=False)
    identity = _identity(spec, x, y)
    _write_json(
        root / "manifest.json",
        {"identity": identity, "manifest_id": _manifest_id(identity), "task": spec},
    )


def evaluate(dataset, candidates_path, output):
    manifest, spec, x, y = _load_dataset(dataset)
    task = _verified_task(manifest, spec, x, y)
    candidates = np.asarray(np.load(str(candidates_path), allow_pickle=False))
    _validate_candidates(spec, candidates)
    scores = np.asarray(task.predict(candidates)).reshape((-1, 1))
    if scores.shape[0] != candidates.shape[0] or not np.all(np.isfinite(scores)):
        raise ValueError("Exact oracle returned invalid scores.")
    root = _empty_output(output)
    np.save(str(root / "candidates.npy"), candidates, allow_pickle=False)
    np.save(str(root / "scores.npy"), scores, allow_pickle=False)
    _write_json(
        root / "evaluation.json",
        {"task": spec, "dataset_manifest_id": manifest["manifest_id"]},
    )


def _baseline_config(spec, output, mode):
    config = {
        "logging_dir": str(Path(output).resolve()),
        "task": spec["name"],
        "task_kwargs": {"relabel": False},
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
    if mode == "smoke":
        config.update({"epochs": 1, "solver_samples": 2, "solver_steps": 1})
    elif mode != "reference":
        raise ValueError("mode must be 'reference' or 'smoke'.")
    return config


def run_gradient_ascent(dataset, output, mode, seed):
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    manifest, spec, x, y = _load_dataset(dataset)
    root = _empty_output(output)
    import tensorflow as tf
    import design_baselines.gradient_ascent as baseline

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    config = _baseline_config(spec, root, mode)
    recorded_config = copy.deepcopy(config)
    original_task_class = baseline.StaticGraphTask
    training_tasks = []

    def verified_training_task(task_name, **task_kwargs):
        task = original_task_class(task_name, **task_kwargs)
        _verify_task_arrays(manifest, spec, x, y, task)
        training_tasks.append(task)
        return task

    baseline.StaticGraphTask = verified_training_task
    try:
        baseline.gradient_ascent(config)
    finally:
        baseline.StaticGraphTask = original_task_class
    if len(training_tasks) != 1:
        raise RuntimeError("Official baseline did not construct exactly one training task.")

    normalized = np.load(str(root / "solution.npy"), allow_pickle=False)
    raw = np.asarray(training_tasks[0].denormalize_x(normalized))
    _validate_candidates(spec, raw)
    np.save(str(root / "candidates.npy"), raw, allow_pickle=False)
    _write_json(
        root / "run.json",
        {
            "task": spec,
            "dataset_manifest_id": manifest["manifest_id"],
            "method": "design_baselines.gradient_ascent",
            "design_baselines_commit": DESIGN_BASELINES_COMMIT,
            "mode": mode,
            "seed": seed,
            "config": recorded_config,
            "normalized_solution": "solution.npy",
            "raw_candidates": "candidates.npy",
        },
    )


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    export = commands.add_parser("export-dataset")
    export.add_argument("--task", choices=tuple(TASK_DIMENSIONS), required=True)
    export.add_argument("--output", required=True)
    oracle = commands.add_parser("evaluate")
    oracle.add_argument("--dataset", required=True)
    oracle.add_argument("--candidates", required=True)
    oracle.add_argument("--output", required=True)
    baseline = commands.add_parser("run-gradient-ascent")
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
    main()
