"""Run the canonical XGB logit-spline convergence and policy experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Mapping

from experiments.config import CorrectnessSpec, ExperimentConfig
from experiments.configs import get_config
from experiments.execution import execute_experiment_run
from experiments.launch import LaunchContext, LaunchPlan, add_launch_args, run_launch_plan
from experiments.paths import results_root
from experiments.results import ExperimentResult


BASE_PRESET = "real_data_xgb_logit_spline_base"
PROJECT_NAME = "xgb-logit-spline-experiment"
RUN_NAME = "xgb_logit_spline_convergence"
DEFAULT_ESTIMATORS = ("first_order", "finite_difference")
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_INITIAL_U = 0.08
DEFAULT_FD_STEP = 1e-4
DEFAULT_T_STEPS = 500


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _test_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("test fraction must satisfy 0 <= value < 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-samples",
        type=_positive_int,
        default=None,
        help="Number of covered spline profiles to sample; omitted uses all 200.",
    )
    parser.add_argument(
        "--test-fraction",
        type=_test_fraction,
        default=DEFAULT_TEST_FRACTION,
        help="Held-out fraction used for policy diagnostics (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Experiment seed (default: 42).")
    parser.add_argument(
        "--t-steps",
        type=_positive_int,
        default=DEFAULT_T_STEPS,
        help=f"Maximum L-BFGS-B iterations (default: {DEFAULT_T_STEPS}).",
    )
    parser.add_argument(
        "--initial-u",
        type=float,
        default=DEFAULT_INITIAL_U,
        help="Initial constant action inside the spline support (default: 0.08).",
    )
    parser.add_argument(
        "--fd-step",
        type=_positive_float,
        default=DEFAULT_FD_STEP,
        help="Action-space central finite-difference step (default: 1e-4).",
    )
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=("first_order", "finite_difference", "spsa", "stein_difference"),
        default=DEFAULT_ESTIMATORS,
        help="Gradient estimators to compare.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step console output.")
    add_launch_args(parser, default_launch="local", default_array=False)
    return parser.parse_args(argv)


def _config_overrides(args: argparse.Namespace) -> dict[str, object]:
    test_fraction = float(args.test_fraction)
    return {
        "policy_kind": "softmax",
        "policy_preprocessing": "artifact",
        "feature_order": "linear",
        "constraint_mode": "none",
        "n_samples": args.n_samples,
        "train_fraction": 1.0 - test_fraction,
        "test_fraction": test_fraction,
        "seed": int(args.seed),
        "initial_u": float(args.initial_u),
        "step_rule": "l-bfgs-b",
        "t_steps": int(args.t_steps),
        "sigma": float(args.fd_step),
        "perturbation_space": "u",
        "enabled_estimators": tuple(args.estimators),
        "constant_u_baselines": (0.0, 0.08, 0.16),
        "grad_norm_tol": 1e-6,
        "plot": True,
        "verbose": not bool(args.quiet),
        "wandb_enabled": False,
    }


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    config = get_config(BASE_PRESET, overrides=_config_overrides(args))
    return replace(config, correctness=CorrectnessSpec(gradient_source="exact"))


def _last(values: object) -> float | None:
    if values is None:
        return None
    sequence = list(values)  # type: ignore[arg-type]
    return float(sequence[-1]) if sequence else None


def _convergence_rows(result: ExperimentResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for estimator, final in result.results.items():
        trace = result.traces[estimator]
        rows.append(
            {
                "estimator": estimator,
                "optimizer_success": trace.optimizer_success,
                "optimizer_status": trace.optimizer_status,
                "optimizer_message": trace.optimizer_message,
                "steps": len(trace.steps),
                "final_objective": float(final.value),
                "final_true_theta_grad_norm": _last(trace.true_theta_grad_norms),
                "mean_u": float(final.u),
                "mean_acceptance": (
                    float(final.mean_acceptance) if final.mean_acceptance is not None else None
                ),
                "runtime_sec": float(final.time),
            }
        )
    return rows


def _format_optional(value: object, format_spec: str = ".6g") -> str:
    if value is None:
        return "n/a"
    return format(float(value), format_spec)


def _print_convergence(rows: list[Mapping[str, Any]], run_dir: Path) -> None:
    print("\nConvergence summary")
    for row in rows:
        print(
            f"- {row['estimator']}: success={row['optimizer_success']}, "
            f"steps={row['steps']}, objective={_format_optional(row['final_objective'])}, "
            f"true_grad_norm={_format_optional(row['final_true_theta_grad_norm'])}, "
            f"mean_u={_format_optional(row['mean_u'])}, "
            f"mean_acceptance={_format_optional(row['mean_acceptance'])}"
        )
        if row.get("optimizer_message"):
            print(f"  optimizer: {row['optimizer_message']}")
    print(f"Run outputs: {run_dir}")


def _run_task(
    index: int,
    context: LaunchContext,
    args: argparse.Namespace,
) -> dict[str, object]:
    if index != 0:
        raise IndexError(f"XGB logit-spline experiment has only task 0, got {index}.")
    overrides = _config_overrides(args)
    config = _build_config(args)
    executed = execute_experiment_run(
        RUN_NAME,
        config,
        runs_root=context.runs_root,
        run_metadata={
            "preset_name": BASE_PRESET,
            "variant_name": RUN_NAME,
            "overrides": overrides,
        },
    )
    rows = _convergence_rows(executed.result)
    _print_convergence(rows, executed.run_context.run_dir)
    return {
        "run_dir": str(executed.run_context.run_dir),
        "convergence": rows,
    }


def _build_launch_plan(args: argparse.Namespace) -> LaunchPlan:
    return LaunchPlan(
        name=PROJECT_NAME,
        task_count=1,
        requires_jax=False,
        run_task=lambda index, context: _run_task(index, context, args),
        runs_root=str(results_root() / PROJECT_NAME),
        default_launch="local",
        default_array=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(_build_launch_plan(args), args=args, argv=original_argv)


if __name__ == "__main__":
    main()
