"""Run a generic seed-aware preset sweep through the shared launcher."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

from experiments.launch import LaunchContext, LaunchPlan, add_launch_args, run_launch_plan
from experiments.sweep_utils import run_sweep as run_seed_sweep


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_preset", help="Registered config preset name to sweep.")
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--run-seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--vary", nargs="+", default=["theta"])
    parser.add_argument("--anchor-seed", type=int, default=None)
    parser.add_argument("--display-keys", nargs="*", default=None)
    parser.add_argument("--per-seed-plots", action="store_true")
    parser.add_argument(
        "--requires-jax",
        action="store_true",
        help="Force GPU Slurm profile/JAX GPU preflight for this sweep.",
    )
    parser.add_argument(
        "--fixed-json",
        default=None,
        help="JSON mapping of fixed seed streams, or a path to a JSON file.",
    )
    override_group = parser.add_mutually_exclusive_group()
    override_group.add_argument(
        "--overrides-json",
        default=None,
        help="JSON mapping for one variant, or a path to a JSON file.",
    )
    override_group.add_argument(
        "--override-list-json",
        default=None,
        help="JSON list of override mappings, or a path to a JSON file.",
    )
    override_group.add_argument(
        "--override-grid-json",
        default=None,
        help="JSON mapping from override keys to value lists, or a path to a JSON file.",
    )
    add_launch_args(parser, default_launch="local", default_array=False)
    return parser.parse_args(argv)


def _json_arg(value: str | None) -> Any:
    if value is None:
        return None
    stripped = value.strip()
    if stripped.startswith(("{", "[")):
        return json.loads(value)
    path = Path(value)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(value)


def _override_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Sequence[Any]] | None, list[dict[str, Any]] | None]:
    if args.override_grid_json is not None:
        payload = _json_arg(args.override_grid_json)
        if not isinstance(payload, Mapping):
            raise ValueError("--override-grid-json must decode to a JSON object.")
        return {str(key): list(value) for key, value in payload.items()}, None
    if args.override_list_json is not None:
        payload = _json_arg(args.override_list_json)
        if not isinstance(payload, list) or not all(isinstance(item, Mapping) for item in payload):
            raise ValueError("--override-list-json must decode to a list of JSON objects.")
        return None, [dict(item) for item in payload]
    if args.overrides_json is not None:
        payload = _json_arg(args.overrides_json)
        if not isinstance(payload, Mapping):
            raise ValueError("--overrides-json must decode to a JSON object.")
        return None, [dict(payload)]
    return None, None


def _fixed_seeds(args: argparse.Namespace) -> dict[str, int | None] | None:
    payload = _json_arg(args.fixed_json)
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("--fixed-json must decode to a JSON object.")
    return {
        str(key): None if value is None else int(value)
        for key, value in payload.items()
    }


def _sweep_requires_jax(args: argparse.Namespace) -> bool:
    if args.requires_jax:
        return True
    override_grid, override_list = _override_inputs(args)
    if override_list is not None:
        return any(overrides.get("compute_backend") == "jax" for overrides in override_list)
    if override_grid is not None:
        return "jax" in set(override_grid.get("compute_backend", ()))
    return False


def _execute_sweep(args: argparse.Namespace):
    override_grid, override_list = _override_inputs(args)
    return run_seed_sweep(
        base_preset=args.base_preset,
        run_seeds=tuple(int(seed) for seed in args.run_seeds),
        override_grid=override_grid,
        override_list=override_list,
        vary=tuple(str(item) for item in args.vary),
        anchor_seed=args.anchor_seed,
        fixed=_fixed_seeds(args),
        per_seed_plots=bool(args.per_seed_plots),
        project_name=args.project_name,
        display_keys=args.display_keys,
    )


def _run_task(index: int, context: LaunchContext, *, args: argparse.Namespace) -> dict[str, object]:
    del context
    if index != 0:
        raise IndexError("generic run_sweep has exactly one task")
    sweep = _execute_sweep(args)
    return {"project_dir": str(sweep.project_dir), "n_runs": len(sweep.run_results)}


def _run_all(context: LaunchContext, *, args: argparse.Namespace) -> None:
    payload = _run_task(0, context, args=args)
    print(f"Completed {payload['n_runs']} runs under {payload['project_dir']}.")


def _build_launch_plan(args: argparse.Namespace) -> LaunchPlan:
    return LaunchPlan(
        name=args.project_name or f"{args.base_preset}-sweep",
        task_count=1,
        requires_jax=_sweep_requires_jax(args),
        run_task=lambda index, context: _run_task(index, context, args=args),
        run_all=lambda context: _run_all(context, args=args),
        default_launch="local",
        default_array=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(_build_launch_plan(args), args=args, argv=original_argv)


if __name__ == "__main__":
    main()
