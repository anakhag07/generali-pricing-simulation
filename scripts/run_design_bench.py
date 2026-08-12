#!/usr/bin/env python3
"""Run isolated Design-Bench data, baseline, and oracle operations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from benchmarks.design_bench import DesignBenchBridge, DesignBenchTaskSpec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python",
        default=os.environ.get("DESIGN_BENCH_PYTHON"),
        help="Python executable in the pinned legacy environment (or set DESIGN_BENCH_PYTHON).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser("export-dataset", help="Export raw task.x/task.y arrays.")
    export.add_argument("--task", required=True)
    export.add_argument("--output", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate", help="Score raw candidate designs exactly.")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--candidates", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)

    baseline = subparsers.add_parser(
        "run-gradient-ascent", help="Run the official Design-Baselines gradient-ascent method."
    )
    baseline.add_argument("--dataset", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.add_argument("--mode", choices=("reference", "smoke"), required=True)
    baseline.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not args.python:
        raise SystemExit("Provide --python or set DESIGN_BENCH_PYTHON to the legacy executable.")
    bridge = DesignBenchBridge(args.python)

    if args.command == "export-dataset":
        artifact = bridge.export_dataset(DesignBenchTaskSpec(args.task), args.output)
        print(artifact.manifest_id)
    elif args.command == "evaluate":
        from benchmarks.design_bench import DatasetArtifact

        artifact = bridge.evaluate(DatasetArtifact.load(args.dataset), args.candidates, args.output)
        print(artifact.directory / "evaluation.json")
    elif args.command == "run-gradient-ascent":
        from benchmarks.design_bench import DatasetArtifact

        artifact = bridge.run_gradient_ascent(
            DatasetArtifact.load(args.dataset), args.output, mode=args.mode, seed=args.seed
        )
        print(artifact.directory / "run.json")
    else:  # pragma: no cover - argparse guarantees a known command
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
