#!/usr/bin/env python3
"""Run the complete Design-Bench Ant gradient-ascent smoke workflow.

Invoke this file with the pinned ``design-baselines`` environment, not the
project's normal Python environment. The default smoke mode trains for one
epoch, proposes two designs, and evaluates both with the exact Ant oracle.
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "design_bench.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("design_bench_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load {}.".format(RUNNER_PATH))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New empty directory for this smoke run.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Reuse an exported Ant dataset instead of exporting one.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--mode",
        choices=("smoke", "reference"),
        default="smoke",
        help="Smoke is the quick wiring check; reference uses official full settings.",
    )
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    runner = _load_runner()
    run_root = runner._empty_output(args.output)

    dataset_dir = args.dataset
    if dataset_dir is None:
        dataset_dir = run_root / "dataset"
        print("[1/4] Exporting raw Ant task.x/task.y to {}".format(dataset_dir))
        runner.export_dataset(runner.ANT_TASK, dataset_dir)
    else:
        dataset_dir = dataset_dir.resolve()
        print("[1/4] Reusing exported Ant dataset at {}".format(dataset_dir))

    manifest, task_spec, x, y = runner._load_dataset(dataset_dir)
    if task_spec["name"] != runner.ANT_TASK:
        raise ValueError("--dataset must refer to an AntMorphology-Exact-v0 export.")

    # This is the offline dataset D used to train the baseline.
    D = {"x": x, "y": y}
    print("[2/4] Loaded D: x shape={}, y shape={}".format(D["x"].shape, D["y"].shape))
    print("      manifest_id={}".format(manifest["manifest_id"]))

    baseline_dir = run_root / "gradient-ascent-{}".format(args.mode)
    print("[3/4] Running official gradient ascent ({})".format(args.mode))
    runner.run_gradient_ascent(
        dataset_dir,
        baseline_dir,
        mode=args.mode,
        seed=args.seed,
    )

    candidates_path = baseline_dir / "candidates.npy"
    evaluation_dir = run_root / "oracle-evaluation"
    print("[4/4] Evaluating raw candidates with task.predict")
    runner.evaluate(dataset_dir, candidates_path, evaluation_dir)

    candidates = np.load(str(candidates_path), allow_pickle=False)
    scores = np.load(str(evaluation_dir / "scores.npy"), allow_pickle=False)
    summary = {
        "dataset": str(dataset_dir),
        "manifest_id": manifest["manifest_id"],
        "mode": args.mode,
        "seed": args.seed,
        "candidate_shape": list(candidates.shape),
        "scores": scores[:, 0].tolist(),
    }
    with (run_root / "smoke-summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("Done. Raw candidates: {}".format(candidates_path))
    print("Oracle scores: {}".format(evaluation_dir / "scores.npy"))
    print("Summary: {}".format(run_root / "smoke-summary.json"))


if __name__ == "__main__":
    main()
