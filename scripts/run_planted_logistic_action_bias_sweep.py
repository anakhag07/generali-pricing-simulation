"""Run the planted-logistic deterministic action-bias sweep.

For each ``lambda_bias`` value, this script optimizes the surrogate
``M_hat(x, u) = M_star(x, u) - lambda_bias * u`` and evaluates the final policy
on the true planted-logistic objective. It also optimizes the true objective once
as an oracle baseline on the same deterministic sampled batch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from experiments.config import CorrectnessSpec  # noqa: E402
from experiments.configs import get_config  # noqa: E402
from experiments.execution import default_reporter_stack, execute_experiment_run  # noqa: E402
from experiments.paths import results_root  # noqa: E402
from experiments.policy_validation import policy_u_values  # noqa: E402
from experiments.sweep_reporting import timestamped_sweep_output_dir, write_rows_csv  # noqa: E402
from objective.objectives import BiasedObjective  # noqa: E402


BASE_PRESET = "planted_logistic_base"
PROJECT_NAME = "planted-logistic-action-bias-sweep"
ESTIMATOR = "first_order"
LAMBDA_BIAS_VALUES = (0.0, 0.01, 0.05, 0.1, 0.2)
N_SAMPLES = 1000
T_STEPS = 1000
OUTPUT_CSV = "planted_logistic_action_bias_sweep.csv"

FIELDNAMES = (
    "lambda_bias",
    "true_objective_at_oracle",
    "true_objective_at_biased_solution",
    "true_gap",
    "mean_action_oracle",
    "mean_action_biased_solution",
    "surrogate_objective_at_biased_solution",
    "optimism_gap",
    "oracle_run_dir",
    "biased_run_dir",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lambda-bias",
        type=float,
        nargs="+",
        default=list(LAMBDA_BIAS_VALUES),
        help="Bias strengths to sweep.",
    )
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--t-steps", type=int, default=T_STEPS)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--project-name", default=PROJECT_NAME)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to a timestamped directory under results/.",
    )
    parser.add_argument(
        "--per-run-plots",
        action="store_true",
        help="Enable normal per-run plots in each oracle/biased run directory.",
    )
    return parser.parse_args(argv)


def _common_overrides(args: argparse.Namespace | SimpleNamespace) -> dict[str, object]:
    return {
        "seed": int(args.seed),
        "n_samples": int(args.n_samples),
        "t_steps": int(args.t_steps),
        "step_rule": "l-bfgs-b",
        "enabled_estimators": (ESTIMATOR,),
        "correctness": CorrectnessSpec(gradient_source="exact"),
        "perturbation_space": "u",
        "plot": False,
        "verbose": False,
        "wandb_enabled": False,
    }


def _oracle_config(args: argparse.Namespace | SimpleNamespace):
    return get_config(BASE_PRESET, overrides=_common_overrides(args))


def _biased_objective(lambda_bias: float) -> BiasedObjective:
    base_objective = get_config(BASE_PRESET).objective
    return BiasedObjective(base_objective=base_objective, lambda_bias=float(lambda_bias))


def _biased_config(lambda_bias: float, args: argparse.Namespace | SimpleNamespace):
    overrides = {
        **_common_overrides(args),
        "objective": _biased_objective(float(lambda_bias)),
    }
    return get_config(BASE_PRESET, overrides=overrides)


def _run_name(lambda_bias: float) -> str:
    return f"lambda-bias-{_value_label(lambda_bias)}"


def _value_label(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _result_theta(executed, estimator: str = ESTIMATOR) -> np.ndarray:
    return np.asarray(executed.result.results[estimator].theta, dtype=float)


def _mean_policy_action(objective: object, theta: np.ndarray, x_samples: object) -> float:
    return float(np.mean(policy_u_values(objective, theta, x_samples)))


def _row_for_lambda(lambda_bias: float, oracle_executed, biased_executed) -> dict[str, object]:
    oracle_objective = oracle_executed.result.config.objective
    biased_objective = biased_executed.result.config.objective
    if not isinstance(biased_objective, BiasedObjective):
        raise ValueError("biased_executed must use BiasedObjective.")
    true_objective = biased_objective.base_objective

    oracle_theta = _result_theta(oracle_executed)
    biased_theta = _result_theta(biased_executed)
    true_at_oracle = float(oracle_objective.value(oracle_theta, oracle_executed.result.x_samples))
    true_at_biased = float(true_objective.value(biased_theta, biased_executed.result.x_samples))
    surrogate_at_biased = float(biased_objective.value(biased_theta, biased_executed.result.x_samples))

    return {
        "lambda_bias": float(lambda_bias),
        "true_objective_at_oracle": true_at_oracle,
        "true_objective_at_biased_solution": true_at_biased,
        "true_gap": true_at_biased - true_at_oracle,
        "mean_action_oracle": _mean_policy_action(
            oracle_objective,
            oracle_theta,
            oracle_executed.result.x_samples,
        ),
        "mean_action_biased_solution": _mean_policy_action(
            biased_objective,
            biased_theta,
            biased_executed.result.x_samples,
        ),
        "surrogate_objective_at_biased_solution": surrogate_at_biased,
        "optimism_gap": surrogate_at_biased - true_at_biased,
        "oracle_run_dir": str(oracle_executed.run_context.run_dir),
        "biased_run_dir": str(biased_executed.run_context.run_dir),
    }


def _output_dir(args: argparse.Namespace | SimpleNamespace) -> Path:
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    return timestamped_sweep_output_dir(
        project_name=str(args.project_name),
        dirname_prefix="action_bias_sweep",
        runs_root=results_root(),
    )


def _write_rows(output_dir: Path, rows: Sequence[dict[str, object]]) -> Path:
    output_path = output_dir / OUTPUT_CSV
    write_rows_csv(output_path, rows, FIELDNAMES)
    return output_path


def run_action_bias_sweep(args: argparse.Namespace | SimpleNamespace) -> tuple[Path, list[dict[str, object]]]:
    output_dir = _output_dir(args)
    runs_root = output_dir / "runs"

    def reporter_stack_factory(config):
        return default_reporter_stack(config, include_plots=bool(args.per_run_plots))

    oracle_executed = execute_experiment_run(
        "oracle-true-objective",
        _oracle_config(args),
        runs_root=runs_root,
        reporter_stack_factory=reporter_stack_factory,
        run_metadata={"preset_name": BASE_PRESET, "variant_name": "oracle-true-objective"},
    )

    rows: list[dict[str, object]] = []
    for lambda_bias in tuple(float(value) for value in args.lambda_bias):
        run_name = _run_name(lambda_bias)
        biased_executed = execute_experiment_run(
            run_name,
            _biased_config(lambda_bias, args),
            runs_root=runs_root,
            reporter_stack_factory=reporter_stack_factory,
            run_metadata={
                "preset_name": BASE_PRESET,
                "variant_name": run_name,
                "lambda_bias": float(lambda_bias),
            },
        )
        rows.append(_row_for_lambda(lambda_bias, oracle_executed, biased_executed))

    _write_rows(output_dir, rows)
    return output_dir, rows


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir, rows = run_action_bias_sweep(args)
    print(f"Completed {len(rows)} planted-logistic action-bias runs.")
    print(f"Wrote {output_dir / OUTPUT_CSV}.")


if __name__ == "__main__":
    main()
