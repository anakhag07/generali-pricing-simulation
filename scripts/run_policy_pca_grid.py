"""Run the GLM policy PCA-dimension by policy-class experiment grid."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

from data.loader import load_mean_observed_acceptance
from experiments.launch import LaunchContext, LaunchPlan, add_launch_args, run_launch_plan, task_payloads
from experiments.paths import results_root
from experiments.policy_pca_grid import (
    PCA_DIMS,
    POLICY_CLASSES,
    PolicyPcaGridSpec,
    run_policy_pca_grid,
    write_policy_pca_outputs,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--estimator", default="first_order")
    parser.add_argument("--t-steps", type=int, default=None)
    parser.add_argument("--project-name", default=None)
    parser.add_argument("--constrained", action="store_true")
    parser.add_argument("--acceptance-floor", type=float, default=None)
    parser.add_argument("--initial-constr-penalty", type=float, default=1.0)
    parser.add_argument("--quiet", action="store_true", help="Disable per-condition progress output.")
    add_launch_args(parser, default_launch="local", default_array=False)
    return parser.parse_args(argv)


def _spec_from_args(args: argparse.Namespace) -> PolicyPcaGridSpec:
    step_rule = "trust-constr" if args.constrained else "l-bfgs-b"
    t_steps = args.t_steps if args.t_steps is not None else (500 if args.constrained else 1000)
    project_name = args.project_name or (
        "policy-pca-grid-constrained" if args.constrained else "policy-pca-grid"
    )
    acceptance_floor = (
        args.acceptance_floor
        if args.acceptance_floor is not None
        else load_mean_observed_acceptance("glm") if args.constrained else None
    )

    return PolicyPcaGridSpec(
        pca_dims=PCA_DIMS,
        policy_classes=POLICY_CLASSES,
        seeds=tuple(args.seeds),
        n_samples=args.n_samples,
        estimator=args.estimator,
        step_rule=step_rule,
        t_steps=t_steps,
        initial_constr_penalty=args.initial_constr_penalty if args.constrained else None,
        acceptance_floor=acceptance_floor,
        project_name=project_name,
        verbose=not args.quiet,
    )


def _task_specs(spec: PolicyPcaGridSpec) -> list[tuple[int | None, str, int]]:
    return [
        (pca_dim, policy_class, int(seed))
        for pca_dim in spec.pca_dims
        for policy_class in spec.policy_classes
        for seed in spec.seeds
    ]


def _run_pca_task(index: int, context: LaunchContext, spec: PolicyPcaGridSpec) -> dict[str, object]:
    del context
    pca_dim, policy_class, seed = _task_specs(spec)[index]
    task_spec = replace(
        spec,
        pca_dims=(pca_dim,),
        policy_classes=(policy_class,),
        seeds=(seed,),
        project_name=f"{spec.project_name}/tasks/task_{index:03d}",
    )
    output = run_policy_pca_grid(task_spec)
    return {
        "pca_dim": "none" if pca_dim is None else int(pca_dim),
        "policy_class": policy_class,
        "seed": seed,
        "output_dir": str(output.output_dir),
        "final_rows": output.final_rows,
        "trace_rows": output.trace_rows,
    }


def _run_pca_serial(context: LaunchContext, spec: PolicyPcaGridSpec) -> None:
    del context
    output = run_policy_pca_grid(spec)
    print(f"Wrote policy PCA grid outputs to {output.output_dir}")


def _collect_pca_tasks(context: LaunchContext, spec: PolicyPcaGridSpec) -> None:
    payloads = task_payloads(context)
    final_rows = [row for payload in payloads for row in payload.get("final_rows", [])]
    trace_rows = [row for payload in payloads for row in payload.get("trace_rows", [])]
    if not final_rows:
        raise ValueError("No policy PCA grid rows were produced.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = results_root() if spec.output_root is None else Path(spec.output_root)
    output_dir = root / spec.project_name / f"policy_pca_grid_array_{timestamp}"
    write_policy_pca_outputs(final_rows, trace_rows, output_dir)
    print(f"Collected {len(payloads)} policy PCA grid array tasks into {output_dir}")


def _build_launch_plan(spec: PolicyPcaGridSpec) -> LaunchPlan:
    return LaunchPlan(
        name=spec.project_name,
        task_count=len(_task_specs(spec)),
        requires_jax=False,
        run_task=lambda index, context: _run_pca_task(index, context, spec),
        run_all=lambda context: _run_pca_serial(context, spec),
        collect=lambda context: _collect_pca_tasks(context, spec),
        runs_root=spec.output_root,
        default_launch="local",
        default_array=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    spec = _spec_from_args(args)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(_build_launch_plan(spec), args=args, argv=original_argv)


if __name__ == "__main__":
    main()
