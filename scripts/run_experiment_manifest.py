"""Run a JSON experiment manifest through the shared launcher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from experiments.finite_policy_lcb import (
    FinitePolicyLCBManifest,
    load_finite_policy_lcb_manifest,
)
from experiments.launch import LaunchContext, LaunchPlan, add_launch_args, run_launch_plan
from experiments.manifest import (
    ExperimentManifest,
    collect_manifest_outputs,
    load_experiment_manifest,
    run_manifest_serial,
    run_manifest_variant,
)
from experiments.policy_lcb.launch import (
    POLICY_LCB_MANIFEST_KINDS,
    PolicyLCBManifest,
    build_policy_lcb_launch_plan,
    load_policy_lcb_manifest,
)
from experiments.policy_capacity import (
    POLICY_CAPACITY_MANIFEST_KIND,
    PolicyCapacityManifest,
    build_policy_capacity_launch_plan,
    load_policy_capacity_manifest,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="Path to the JSON experiment manifest.")
    parser.add_argument(
        "--runs-root",
        default=None,
        help="Override the results root for this manifest run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun variants/seeds even when all requested outputs already exist.",
    )
    add_launch_args(parser, default_launch=None, default_array=None)
    return parser.parse_args(argv)


def _apply_manifest_launch_defaults(
    args: argparse.Namespace,
    manifest: ExperimentManifest | PolicyLCBManifest | PolicyCapacityManifest,
) -> None:
    if args.array_max_parallel is None:
        args.array_max_parallel = manifest.launch.array_max_parallel


def _run_task(
    index: int,
    context: LaunchContext,
    *,
    manifest: ExperimentManifest,
    args: argparse.Namespace,
) -> dict[str, object]:
    if manifest.launch.array == "variant":
        return run_manifest_variant(
            manifest,
            index,
            runs_root=context.runs_root,
            force=bool(args.force),
        )
    if index != 0:
        raise IndexError("Non-array manifest runs have exactly one task.")
    return run_manifest_serial(
        manifest,
        runs_root=context.runs_root,
        force=bool(args.force),
    )


def _run_all(
    context: LaunchContext,
    *,
    manifest: ExperimentManifest,
    args: argparse.Namespace,
) -> None:
    payload = run_manifest_serial(
        manifest,
        runs_root=context.runs_root,
        force=bool(args.force),
    )
    print(
        f"Completed {payload['n_runs']} runs under {payload['project_dir']} "
        f"({payload['n_skipped_variants']} variants skipped)."
    )


def _collect(context: LaunchContext, *, manifest: ExperimentManifest) -> None:
    payload = collect_manifest_outputs(manifest, runs_root=context.runs_root)
    print(
        f"Collected {payload['n_final_rows']} final rows and "
        f"{payload['n_derived_rows']} derived rows under {payload['project_dir']}."
    )


def _build_launch_plan(
    args: argparse.Namespace,
    manifest: ExperimentManifest | None = None,
) -> LaunchPlan:
    resolved_manifest = manifest or load_experiment_manifest(args.manifest)
    task_count = len(resolved_manifest.variants) if resolved_manifest.launch.array == "variant" else 1
    return LaunchPlan(
        name=resolved_manifest.name,
        task_count=task_count,
        requires_jax=resolved_manifest.requires_jax(),
        run_task=lambda index, context: _run_task(
            index,
            context,
            manifest=resolved_manifest,
            args=args,
        ),
        run_all=lambda context: _run_all(context, manifest=resolved_manifest, args=args),
        collect=lambda context: _collect(context, manifest=resolved_manifest),
        runs_root=args.runs_root,
        default_launch=resolved_manifest.launch.mode,
        default_array=resolved_manifest.launch.array == "variant",
    )


def _build_policy_lcb_launch_plan(
    args: argparse.Namespace,
    manifest: PolicyLCBManifest | None = None,
) -> LaunchPlan:
    resolved_manifest = manifest or load_policy_lcb_manifest(args.manifest)
    return build_policy_lcb_launch_plan(
        resolved_manifest,
        runs_root=args.runs_root,
        force=bool(args.force),
    )


def _build_finite_policy_lcb_launch_plan(
    args: argparse.Namespace,
    manifest: FinitePolicyLCBManifest | None = None,
) -> LaunchPlan:
    """Compatibility wrapper for tests and callers of the finite-only helper."""
    resolved = manifest or load_finite_policy_lcb_manifest(args.manifest)
    return _build_policy_lcb_launch_plan(args, resolved)


def _manifest_kind(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Experiment manifest must be a JSON object.")
    return str(payload.get("kind") or "optimization")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    kind = _manifest_kind(args.manifest)
    if kind in POLICY_LCB_MANIFEST_KINDS:
        manifest = load_policy_lcb_manifest(Path(args.manifest))
        plan = _build_policy_lcb_launch_plan(args, manifest)
    elif kind == "optimization":
        manifest = load_experiment_manifest(Path(args.manifest))
        plan = _build_launch_plan(args, manifest)
    elif kind == POLICY_CAPACITY_MANIFEST_KIND:
        manifest = load_policy_capacity_manifest(Path(args.manifest))
        plan = build_policy_capacity_launch_plan(
            manifest,
            runs_root=args.runs_root,
            force=bool(args.force),
        )
    else:
        raise ValueError(f"Unsupported experiment manifest kind {kind!r}.")
    _apply_manifest_launch_defaults(args, manifest)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(plan, args=args, argv=original_argv)


if __name__ == "__main__":
    main()
