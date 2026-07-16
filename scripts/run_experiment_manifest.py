"""Run a JSON experiment manifest through the shared experiment launcher."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from experiments.launch import LaunchContext, LaunchPlan, add_launch_args, run_launch_plan
from experiments.manifest import (
    load_experiment_manifest,
    manifest_requires_jax,
    run_experiment_manifest,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Path to a JSON experiment manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Print the expanded plan without running.")
    parser.add_argument("--runs-root", default=None, help="Optional output root override.")
    parser.add_argument(
        "--requires-jax",
        action="store_true",
        help="Force GPU Slurm profile/JAX GPU preflight for this manifest.",
    )
    add_launch_args(parser, default_launch="local", default_array=False)
    return parser.parse_args(argv)


def _manifest_name(path: str | Path) -> str:
    payload = load_experiment_manifest(path)
    return str(payload.get("name") or payload.get("project_name") or Path(path).stem)


def _plan_requires_jax(args: argparse.Namespace) -> bool:
    if args.requires_jax:
        return True
    return manifest_requires_jax(load_experiment_manifest(args.manifest))


def _run_task(index: int, context: LaunchContext, *, args: argparse.Namespace) -> dict[str, object]:
    del context
    if index != 0:
        raise IndexError("experiment manifest has exactly one launch task")
    result = run_experiment_manifest(
        args.manifest,
        dry_run=bool(args.dry_run),
        runs_root=args.runs_root,
    )
    _print_result(result)
    return {
        "manifest": str(args.manifest),
        "dry_run": bool(args.dry_run),
        "n_sweeps": len(result.sweeps),
        "n_variants": sum(len(sweep.variants) for sweep in result.sweeps),
        "n_skipped_variants": sum(len(sweep.skipped_variants) for sweep in result.sweeps),
        "n_executed_runs": result.executed_runs,
    }


def _run_all(context: LaunchContext, *, args: argparse.Namespace) -> None:
    _run_task(0, context, args=args)


def _build_launch_plan(args: argparse.Namespace) -> LaunchPlan:
    return LaunchPlan(
        name=_manifest_name(args.manifest),
        task_count=1,
        requires_jax=_plan_requires_jax(args),
        run_task=lambda index, context: _run_task(index, context, args=args),
        run_all=lambda context: _run_all(context, args=args),
        default_launch="local",
        default_array=False,
    )


def _print_result(result) -> None:
    for sweep in result.sweeps:
        status = "planned" if sweep.dry_run else "completed"
        print(
            f"{status} sweep '{sweep.name}': "
            f"{len(sweep.variants)} variants, "
            f"{len(sweep.skipped_variants)} skipped, "
            f"{sweep.executed_runs} executed runs, "
            f"project_dir={sweep.project_dir}"
        )
        if sweep.dry_run:
            for variant in sweep.variants:
                skip_text = " [complete]" if variant.name in sweep.skipped_variants else ""
                print(f"  - {variant.name}{skip_text}")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(_build_launch_plan(args), args=args, argv=original_argv)


if __name__ == "__main__":
    main()
