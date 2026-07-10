"""Run a planted-logistic support-bias exploitation sweep.

The surrogate is exact inside an action-support band around the planted optimum
and optimistic only above support:

``M_hat(x, u) = M_star(x, u) - lambda_bias * max(0, u - (u_star + support_radius))``.

This tests whether optimizer exploitation is mediated by off-support action
selection rather than by a global linear action preference.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
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
from objective.objectives import BiasedObjective, UpperSupportHingeBias  # noqa: E402


BASE_PRESET = "planted_logistic_base"
PROJECT_NAME = "planted-logistic-support-bias-sweep"
ESTIMATOR = "first_order"
LAMBDA_BIAS_VALUES = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2)
SUPPORT_RADII = (0.02, 0.05, 0.1, 0.2)
N_SAMPLES = 1000
T_STEPS = 1000
OUTPUT_CSV = "planted_logistic_support_bias_sweep.csv"

FIELDNAMES = (
    "lambda_bias",
    "support_radius",
    "support_upper",
    "smooth_tau",
    "true_objective_at_oracle",
    "true_objective_at_biased_solution",
    "true_gap",
    "surrogate_objective_at_biased_solution",
    "optimism_gap",
    "mean_action_oracle",
    "mean_action_biased_solution",
    "support_violation_rate",
    "mean_support_excess",
    "max_support_excess",
    "theta_l2_from_oracle",
    "optimizer_success",
    "optimizer_status",
    "oracle_run_dir",
    "biased_run_dir",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-bias", type=float, nargs="+", default=list(LAMBDA_BIAS_VALUES))
    parser.add_argument("--support-radius", type=float, nargs="+", default=list(SUPPORT_RADII))
    parser.add_argument(
        "--smooth-tau",
        type=float,
        default=None,
        help="Optional smooth hinge temperature. Omit for hard support hinge.",
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


def _support_bias(lambda_bias: float, support_radius: float, args: argparse.Namespace | SimpleNamespace) -> UpperSupportHingeBias:
    base_objective = get_config(BASE_PRESET).objective
    return UpperSupportHingeBias(
        lambda_bias=float(lambda_bias),
        support_center=float(base_objective.optimal_u()),
        support_radius=float(support_radius),
        smooth_tau=None if args.smooth_tau is None else float(args.smooth_tau),
    )


def _biased_config(lambda_bias: float, support_radius: float, args: argparse.Namespace | SimpleNamespace):
    base_objective = get_config(BASE_PRESET).objective
    overrides = {
        **_common_overrides(args),
        "objective": BiasedObjective(
            base_objective=base_objective,
            bias=_support_bias(lambda_bias, support_radius, args),
        ),
    }
    return get_config(BASE_PRESET, overrides=overrides)


def _run_name(lambda_bias: float, support_radius: float) -> str:
    return f"lambda-{_value_label(lambda_bias)}__support-radius-{_value_label(support_radius)}"


def _value_label(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _result_theta(executed, estimator: str = ESTIMATOR) -> np.ndarray:
    return np.asarray(executed.result.results[estimator].theta, dtype=float)


def _support_metrics(objective: object, theta: np.ndarray, x_samples: object, support_upper: float) -> dict[str, float]:
    u_values = policy_u_values(objective, theta, x_samples)
    excess = np.maximum(0.0, u_values - float(support_upper))
    return {
        "mean_action": float(np.mean(u_values)),
        "support_violation_rate": float(np.mean(u_values > float(support_upper))),
        "mean_support_excess": float(np.mean(excess)),
        "max_support_excess": float(np.max(excess)),
    }


def _row_for_variant(
    lambda_bias: float,
    support_radius: float,
    oracle_executed,
    biased_executed,
) -> dict[str, object]:
    biased_objective = biased_executed.result.config.objective
    if not isinstance(biased_objective, BiasedObjective):
        raise ValueError("biased_executed must use BiasedObjective.")
    support_bias = biased_objective.bias
    if not isinstance(support_bias, UpperSupportHingeBias):
        raise ValueError("biased_executed must use UpperSupportHingeBias.")
    true_objective = biased_objective.base_objective
    x_samples = biased_executed.result.x_samples

    oracle_theta = _result_theta(oracle_executed)
    biased_theta = _result_theta(biased_executed)
    true_at_oracle = float(true_objective.value(oracle_theta, x_samples))
    true_at_biased = float(true_objective.value(biased_theta, x_samples))
    surrogate_at_biased = float(biased_objective.value(biased_theta, x_samples))
    support_metrics = _support_metrics(biased_objective, biased_theta, x_samples, support_bias.support_upper)
    oracle_metrics = _support_metrics(true_objective, oracle_theta, x_samples, support_bias.support_upper)
    trace = biased_executed.result.traces.get(ESTIMATOR)
    return {
        "lambda_bias": float(lambda_bias),
        "support_radius": float(support_radius),
        "support_upper": float(support_bias.support_upper),
        "smooth_tau": "" if support_bias.smooth_tau is None else float(support_bias.smooth_tau),
        "true_objective_at_oracle": true_at_oracle,
        "true_objective_at_biased_solution": true_at_biased,
        "true_gap": true_at_biased - true_at_oracle,
        "surrogate_objective_at_biased_solution": surrogate_at_biased,
        "optimism_gap": surrogate_at_biased - true_at_biased,
        "mean_action_oracle": oracle_metrics["mean_action"],
        "mean_action_biased_solution": support_metrics["mean_action"],
        "support_violation_rate": support_metrics["support_violation_rate"],
        "mean_support_excess": support_metrics["mean_support_excess"],
        "max_support_excess": support_metrics["max_support_excess"],
        "theta_l2_from_oracle": float(np.linalg.norm(biased_theta - oracle_theta)),
        "optimizer_success": "" if trace is None else bool(trace.optimizer_success),
        "optimizer_status": "" if trace is None else trace.optimizer_status,
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
        dirname_prefix="support_bias_sweep",
        runs_root=results_root(),
    )


def _write_outputs(output_dir: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    output_path = output_dir / OUTPUT_CSV
    write_rows_csv(output_path, rows, FIELDNAMES)
    if rows:
        _write_summary_plots(output_dir, rows)
    return output_path


def run_support_bias_sweep(args: argparse.Namespace | SimpleNamespace) -> tuple[Path, list[dict[str, object]]]:
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
    for support_radius in tuple(float(value) for value in args.support_radius):
        for lambda_bias in tuple(float(value) for value in args.lambda_bias):
            run_name = _run_name(lambda_bias, support_radius)
            biased_executed = execute_experiment_run(
                run_name,
                _biased_config(lambda_bias, support_radius, args),
                runs_root=runs_root,
                reporter_stack_factory=reporter_stack_factory,
                run_metadata={
                    "preset_name": BASE_PRESET,
                    "variant_name": run_name,
                    "lambda_bias": float(lambda_bias),
                    "support_radius": float(support_radius),
                },
            )
            rows.append(_row_for_variant(lambda_bias, support_radius, oracle_executed, biased_executed))

    _write_outputs(output_dir, rows)
    return output_dir, rows


def _write_summary_plots(output_dir: Path, rows: Sequence[Mapping[str, object]]) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    _plot_heatmap(
        plot_dir / "true_gap_heatmap.png",
        rows,
        value_key="true_gap",
        color_label=r"True gap $J_\star(\hat\theta_{\lambda,r})-J_\star(\hat\theta_{oracle})$",
        title=r"True degradation from off-support optimism",
    )
    _plot_heatmap(
        plot_dir / "mean_support_excess_heatmap.png",
        rows,
        value_key="mean_support_excess",
        color_label=r"Mean support excess $n^{-1}\sum_i(\pi_{\hat\theta}(x_i)-h)_+$",
        title=r"Off-support action excess",
    )
    _plot_support_excess_scatter(plot_dir / "true_gap_vs_support_excess.png", rows)


def _plot_heatmap(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    color_label: str,
    title: str,
) -> None:
    lambdas = sorted({float(row["lambda_bias"]) for row in rows})
    radii = sorted({float(row["support_radius"]) for row in rows})
    matrix = np.full((len(lambdas), len(radii)), np.nan, dtype=float)
    for row in rows:
        i = lambdas.index(float(row["lambda_bias"]))
        j = radii.index(float(row["support_radius"]))
        matrix[i, j] = float(row[value_key])
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(radii)), labels=[f"{value:g}" for value in radii])
    ax.set_yticks(np.arange(len(lambdas)), labels=[f"{value:g}" for value in lambdas])
    ax.set_xlabel(r"Support radius $r$ in $h=u^\star+r$")
    ax.set_ylabel(r"Bias strength $\lambda_{bias}$")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(color_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_support_excess_scatter(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    lambdas = np.asarray([float(row["lambda_bias"]) for row in rows], dtype=float)
    excess = np.asarray([float(row["mean_support_excess"]) for row in rows], dtype=float)
    true_gap = np.asarray([float(row["true_gap"]) for row in rows], dtype=float)
    radii = [float(row["support_radius"]) for row in rows]
    scatter = ax.scatter(excess, true_gap, c=lambdas, cmap="viridis", s=58, edgecolor="black", linewidth=0.4)
    for x_val, y_val, radius in zip(excess, true_gap, radii):
        ax.annotate(f"r={radius:g}", (x_val, y_val), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel(r"Mean support excess $n^{-1}\sum_i(\pi_{\hat\theta}(x_i)-h)_+$")
    ax.set_ylabel(r"True gap $J_\star(\hat\theta_{\lambda,r})-J_\star(\hat\theta_{oracle})$")
    ax.set_title(r"True degradation versus off-support action excess")
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"Bias strength $\lambda_{bias}$")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    output_dir, rows = run_support_bias_sweep(args)
    print(f"Completed {len(rows)} planted-logistic support-bias runs.")
    print(f"Wrote {output_dir / OUTPUT_CSV}.")
    print(f"Wrote plots under {output_dir / 'plots'}.")


if __name__ == "__main__":
    main()
