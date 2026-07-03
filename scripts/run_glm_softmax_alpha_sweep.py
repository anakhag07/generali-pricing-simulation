"""Run trust-constrained GLM softmax policy sweeps over symmetric bounds."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiments.configs import get_config
from experiments.execution import execute_experiment_run
from experiments.policy_artifacts import load_policy_artifact
from experiments.reporting.context import RunContext
from experiments.results import ExperimentResult, PolicyEvaluation
from reporting.visualization import _estimator_style

BASE_PRESET = "real_data_glm_base"
PROJECT_NAME = "glm-softmax-alpha-sweep"
ALPHA_VALUES = (0.5, 0.4, 0.3, 0.2, 0.15, 0.125, 0.1, 0.075)
RUN_OVERRIDES: dict[str, object] = {
    "policy_kind": "softmax",
    "policy_preprocessing": "no_pca",
    "feature_order": "linear",
    "constraint_mode": "trust_constr",
    "n_grad_samples": 8,
    "t_steps": 100,
    "enabled_estimators": ("first_order",),
    "plot": True,
    "verbose": False,
    "wandb_enabled": False,
    "n_samples": None,
    "train_fraction": 0.8,
    "test_fraction": 0.2,
    "initial_u": 0.0,
}

BIN_SUMMARY_SPLITS = ("train", "test", "all")
PLOT_SPLIT = "train"
ACCEPTANCE_THRESHOLD = 0.5
U_BIN_COUNT = 10

_FINAL_FIELDNAMES = [
    "run_name",
    "alpha",
    "action_low",
    "action_high",
    "estimator",
    "u",
    "mean_acceptance",
    "value",
    "expected_profit",
    "runtime_sec",
    "constraint_violation",
    "acceptance_multiplier",
    "constraint_penalty",
    "train_n_samples",
    "train_objective_value",
    "train_objective_sum",
    "train_expected_profit",
    "train_mean_u",
    "train_u_q25",
    "train_u_q75",
    "train_mean_acceptance",
    "train_projected_loss",
    "train_projected_revenue",
    "test_n_samples",
    "test_objective_value",
    "test_objective_sum",
    "test_expected_profit",
    "test_mean_u",
    "test_u_q25",
    "test_u_q75",
    "test_mean_acceptance",
    "test_projected_loss",
    "test_projected_revenue",
    "policy_artifact",
    "run_dir",
]

_BIN_FIELDNAMES = [
    "run_name",
    "alpha",
    "estimator",
    "split",
    "bin_type",
    "bin_index",
    "bin_name",
    "bin_label",
    "bin_low",
    "bin_high",
    "n_rows",
    "mean_u",
    "mean_acceptance",
    "mean_objective_contribution",
    "total_objective_contribution",
    "mean_expected_profit",
    "total_expected_profit",
    "mean_loss",
    "mean_premium",
    "mean_revenue",
    "policy_artifact",
]


def _alpha_label(alpha: float) -> str:
    return f"alpha_{float(alpha):.3f}".replace(".", "p")


def _run_alpha(alpha: float) -> tuple[float, str, ExperimentResult, RunContext]:
    alpha_value = float(alpha)
    run_name = _alpha_label(alpha_value)
    overrides = {
        **RUN_OVERRIDES,
        "softmax_action_bounds": (-alpha_value, alpha_value),
    }
    config = get_config(BASE_PRESET, overrides=overrides)
    executed = execute_experiment_run(
        run_name,
        config,
        runs_root=str(Path("outputs") / PROJECT_NAME),
    )
    return alpha_value, run_name, executed.result, executed.run_context


def _collect_final_rows(
    runs: Sequence[tuple[float, str, ExperimentResult, RunContext]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for alpha, run_name, result, run_context in runs:
        for estimator, estimator_result in result.results.items():
            policy_artifact = run_context.run_dir / "policies" / estimator / "policy.json"
            row: dict[str, object] = {
                "run_name": run_name,
                "alpha": float(alpha),
                "action_low": -float(alpha),
                "action_high": float(alpha),
                "estimator": estimator,
                "u": float(estimator_result.u),
                "mean_acceptance": _optional_float(estimator_result.mean_acceptance),
                "value": float(estimator_result.value),
                "expected_profit": -float(estimator_result.value),
                "runtime_sec": float(estimator_result.time),
                "constraint_violation": _optional_float(estimator_result.constraint_violation),
                "acceptance_multiplier": _optional_float(estimator_result.acceptance_multiplier),
                "constraint_penalty": _optional_float(estimator_result.constraint_penalty),
                "policy_artifact": str(policy_artifact) if policy_artifact.exists() else "",
                "run_dir": str(run_context.run_dir),
            }
            row.update(_evaluation_fields("train", result.train_metrics.get(estimator)))
            row.update(_evaluation_fields("test", result.test_metrics.get(estimator)))
            rows.append(row)
    return rows


def _collect_bin_rows_from_artifacts(
    final_rows: Sequence[Mapping[str, object]],
    *,
    splits: Sequence[str] = BIN_SUMMARY_SPLITS,
    u_bin_count: int = U_BIN_COUNT,
    acceptance_threshold: float = ACCEPTANCE_THRESHOLD,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for final_row in final_rows:
        artifact_path = str(final_row.get("policy_artifact", ""))
        if not artifact_path:
            continue
        artifact = load_policy_artifact(artifact_path)
        rows.extend(
            _collect_artifact_bin_rows(
                alpha=float(final_row["alpha"]),
                run_name=str(final_row["run_name"]),
                artifact=artifact,
                policy_artifact_path=artifact_path,
                splits=splits,
                u_bin_count=u_bin_count,
                acceptance_threshold=acceptance_threshold,
            )
        )
    return rows


def _collect_artifact_bin_rows(
    *,
    alpha: float,
    run_name: str,
    artifact: object,
    policy_artifact_path: str,
    splits: Sequence[str] = BIN_SUMMARY_SPLITS,
    u_bin_count: int = U_BIN_COUNT,
    acceptance_threshold: float = ACCEPTANCE_THRESHOLD,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in splits:
        try:
            values = _artifact_row_values(artifact, split=split)
        except ValueError as exc:
            if split == "test" and "no test" in str(exc).lower():
                continue
            raise
        rows.extend(
            _acceptance_bin_rows(
                alpha=alpha,
                run_name=run_name,
                estimator=str(getattr(artifact, "estimator")),
                split=split,
                values=values,
                policy_artifact_path=policy_artifact_path,
                threshold=acceptance_threshold,
            )
        )
        rows.extend(
            _u_bin_rows(
                alpha=alpha,
                run_name=run_name,
                estimator=str(getattr(artifact, "estimator")),
                split=split,
                values=values,
                policy_artifact_path=policy_artifact_path,
                bin_count=u_bin_count,
            )
        )
    return rows


def _artifact_row_values(artifact: object, *, split: str) -> dict[str, np.ndarray]:
    x_eval = artifact.load_x(split=split)  # type: ignore[attr-defined]
    objective = artifact.build_objective()  # type: ignore[attr-defined]
    theta = np.asarray(getattr(artifact, "theta"), dtype=float)
    u_values = np.asarray(objective.policy_value(theta, x_eval), dtype=float).reshape(-1)
    clip_fn = getattr(objective, "_clip_u", None)
    if callable(clip_fn):
        u_values = np.asarray(clip_fn(u_values), dtype=float).reshape(-1)
    acceptance = np.asarray(objective._acceptance_proba(x_eval, u_values), dtype=float).reshape(-1)
    loss = np.asarray(objective._loss_prediction(x_eval), dtype=float).reshape(-1)
    premium = np.asarray(objective._premium_values(x_eval), dtype=float).reshape(-1)
    revenue = (u_values + 1.0) * premium
    value_batch_fn = getattr(objective, "_value_batch", None)
    if callable(value_batch_fn):
        objective_contribution = np.asarray(value_batch_fn(x_eval, u_values), dtype=float).reshape(-1)
    else:
        objective_contribution = acceptance * (loss - revenue)
    expected_profit = -objective_contribution

    row_count = int(x_eval.shape[0])
    arrays = {
        "u": u_values,
        "acceptance": acceptance,
        "loss": loss,
        "premium": premium,
        "revenue": revenue,
        "objective_contribution": objective_contribution,
        "expected_profit": expected_profit,
    }
    for name, values in arrays.items():
        if values.shape != (row_count,):
            raise ValueError(f"{name} must have one value per evaluated row.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values.")
    return arrays


def _acceptance_bin_rows(
    *,
    alpha: float,
    run_name: str,
    estimator: str,
    split: str,
    values: Mapping[str, np.ndarray],
    policy_artifact_path: str,
    threshold: float,
) -> list[dict[str, object]]:
    acceptance = np.asarray(values["acceptance"], dtype=float)
    threshold_label = f"{float(threshold):.3f}".replace(".", "p")
    specs = [
        (
            0,
            f"acceptance_le_{threshold_label}",
            f"acceptance <= {float(threshold):.3f}",
            0.0,
            float(threshold),
            acceptance <= float(threshold),
        ),
        (
            1,
            f"acceptance_gt_{threshold_label}",
            f"acceptance > {float(threshold):.3f}",
            float(threshold),
            1.0,
            acceptance > float(threshold),
        ),
    ]
    return [
        _bin_summary_row(
            alpha=alpha,
            run_name=run_name,
            estimator=estimator,
            split=split,
            bin_type="acceptance_threshold",
            bin_index=index,
            bin_name=name,
            bin_label=label,
            bin_low=low,
            bin_high=high,
            mask=np.asarray(mask, dtype=bool),
            values=values,
            policy_artifact_path=policy_artifact_path,
        )
        for index, name, label, low, high, mask in specs
    ]


def _u_bin_rows(
    *,
    alpha: float,
    run_name: str,
    estimator: str,
    split: str,
    values: Mapping[str, np.ndarray],
    policy_artifact_path: str,
    bin_count: int,
) -> list[dict[str, object]]:
    u_values = np.asarray(values["u"], dtype=float)
    edges = _u_bin_edges(alpha, bin_count)
    rows: list[dict[str, object]] = []
    for index in range(edges.size - 1):
        low = float(edges[index])
        high = float(edges[index + 1])
        if index == edges.size - 2:
            mask = (u_values >= low) & (u_values <= high)
            label = f"{low:.3f} <= u <= {high:.3f}"
        else:
            mask = (u_values >= low) & (u_values < high)
            label = f"{low:.3f} <= u < {high:.3f}"
        rows.append(
            _bin_summary_row(
                alpha=alpha,
                run_name=run_name,
                estimator=estimator,
                split=split,
                bin_type="u",
                bin_index=index,
                bin_name=f"u_bin_{index:02d}",
                bin_label=label,
                bin_low=low,
                bin_high=high,
                mask=mask,
                values=values,
                policy_artifact_path=policy_artifact_path,
            )
        )
    return rows


def _u_bin_edges(alpha: float, bin_count: int) -> np.ndarray:
    alpha_value = float(alpha)
    bin_count_int = int(bin_count)
    if alpha_value <= 0.0:
        raise ValueError("alpha must be positive.")
    if bin_count_int <= 0:
        raise ValueError("bin_count must be positive.")
    return np.linspace(-alpha_value, alpha_value, bin_count_int + 1)


def _bin_summary_row(
    *,
    alpha: float,
    run_name: str,
    estimator: str,
    split: str,
    bin_type: str,
    bin_index: int,
    bin_name: str,
    bin_label: str,
    bin_low: float,
    bin_high: float,
    mask: np.ndarray,
    values: Mapping[str, np.ndarray],
    policy_artifact_path: str,
) -> dict[str, object]:
    mask_arr = np.asarray(mask, dtype=bool).reshape(-1)
    n_rows = int(np.sum(mask_arr))
    row: dict[str, object] = {
        "run_name": run_name,
        "alpha": float(alpha),
        "estimator": estimator,
        "split": split,
        "bin_type": bin_type,
        "bin_index": int(bin_index),
        "bin_name": bin_name,
        "bin_label": bin_label,
        "bin_low": float(bin_low),
        "bin_high": float(bin_high),
        "n_rows": n_rows,
        "policy_artifact": policy_artifact_path,
    }
    if n_rows == 0:
        row.update(
            {
                "mean_u": "",
                "mean_acceptance": "",
                "mean_objective_contribution": "",
                "total_objective_contribution": "",
                "mean_expected_profit": "",
                "total_expected_profit": "",
                "mean_loss": "",
                "mean_premium": "",
                "mean_revenue": "",
            }
        )
        return row
    objective_values = np.asarray(values["objective_contribution"], dtype=float)[mask_arr]
    expected_profit = np.asarray(values["expected_profit"], dtype=float)[mask_arr]
    row.update(
        {
            "mean_u": float(np.mean(np.asarray(values["u"], dtype=float)[mask_arr])),
            "mean_acceptance": float(np.mean(np.asarray(values["acceptance"], dtype=float)[mask_arr])),
            "mean_objective_contribution": float(np.mean(objective_values)),
            "total_objective_contribution": float(np.sum(objective_values)),
            "mean_expected_profit": float(np.mean(expected_profit)),
            "total_expected_profit": float(np.sum(expected_profit)),
            "mean_loss": float(np.mean(np.asarray(values["loss"], dtype=float)[mask_arr])),
            "mean_premium": float(np.mean(np.asarray(values["premium"], dtype=float)[mask_arr])),
            "mean_revenue": float(np.mean(np.asarray(values["revenue"], dtype=float)[mask_arr])),
        }
    )
    return row


def _evaluation_fields(prefix: str, evaluation: PolicyEvaluation | None) -> dict[str, object]:
    fields = {
        f"{prefix}_n_samples": "",
        f"{prefix}_objective_value": "",
        f"{prefix}_objective_sum": "",
        f"{prefix}_expected_profit": "",
        f"{prefix}_mean_u": "",
        f"{prefix}_u_q25": "",
        f"{prefix}_u_q75": "",
        f"{prefix}_mean_acceptance": "",
        f"{prefix}_projected_loss": "",
        f"{prefix}_projected_revenue": "",
    }
    if evaluation is None:
        return fields
    fields.update(
        {
            f"{prefix}_n_samples": int(evaluation.n_samples),
            f"{prefix}_objective_value": float(evaluation.objective_value),
            f"{prefix}_objective_sum": float(evaluation.objective_sum),
            f"{prefix}_expected_profit": -float(evaluation.objective_value),
            f"{prefix}_mean_u": float(evaluation.mean_u),
            f"{prefix}_u_q25": float(evaluation.u_q25),
            f"{prefix}_u_q75": float(evaluation.u_q75),
            f"{prefix}_mean_acceptance": _optional_float(evaluation.mean_acceptance),
            f"{prefix}_projected_loss": _optional_float(evaluation.projected_loss),
            f"{prefix}_projected_revenue": _optional_float(evaluation.projected_revenue),
        }
    )
    return fields


def _write_rows(rows: Sequence[Mapping[str, object]], csv_path: Path, fieldnames: Sequence[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_plots(
    final_rows: Sequence[Mapping[str, object]],
    bin_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_alpha_metric(
        final_rows,
        output_dir,
        metric_key="value",
        ylabel="Final objective value",
        title="Final objective by softmax bound",
        filename="alpha_vs_objective.png",
    )
    _plot_alpha_metric(
        final_rows,
        output_dir,
        metric_key="expected_profit",
        ylabel="Final expected profit = -objective",
        title="Final expected profit by softmax bound",
        filename="alpha_vs_expected_profit.png",
    )
    _plot_acceptance_bin_profit(bin_rows, output_dir)
    _plot_u_bin_profit_by_alpha(bin_rows, output_dir)


def _plot_alpha_metric(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    metric_key: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    valid_rows = [row for row in rows if _has_float(row.get(metric_key))]
    if not valid_rows:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    alphas = sorted({float(row["alpha"]) for row in valid_rows})
    for estimator, estimator_rows in _rows_by_estimator(valid_rows):
        sorted_rows = sorted(estimator_rows, key=lambda row: float(row["alpha"]))
        style = _estimator_style(estimator)
        ax.plot(
            [float(row["alpha"]) for row in sorted_rows],
            [float(row[metric_key]) for row in sorted_rows],
            label=str(style["label"]),
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=float(style.get("marker_size", 6.0)),
            alpha=0.9,
        )
    _configure_alpha_axis(ax, alphas)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def _plot_acceptance_bin_profit(
    bin_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    split: str = PLOT_SPLIT,
    filename: str = "alpha_profit_by_acceptance_bin.png",
) -> None:
    rows = [
        row
        for row in bin_rows
        if row.get("bin_type") == "acceptance_threshold"
        and row.get("split") == split
        and _has_float(row.get("mean_expected_profit"))
    ]
    if not rows:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    alphas = sorted({float(row["alpha"]) for row in rows})
    line_styles = {"acceptance_le_0p500": "-", "acceptance_gt_0p500": "--"}
    for estimator, estimator_rows in _rows_by_estimator(rows):
        style = _estimator_style(estimator)
        bin_names = sorted({str(row["bin_name"]) for row in estimator_rows})
        for bin_name in bin_names:
            selected = sorted(
                [row for row in estimator_rows if str(row["bin_name"]) == bin_name],
                key=lambda row: float(row["alpha"]),
            )
            label = str(selected[0]["bin_label"])
            if len(bin_names) > 1:
                label = f"{style['label']} / {label}"
            ax.plot(
                [float(row["alpha"]) for row in selected],
                [float(row["mean_expected_profit"]) for row in selected],
                label=label,
                color=style["color"],
                linestyle=line_styles.get(bin_name, "-"),
                marker=style["marker"],
                linewidth=1.8,
                markersize=float(style.get("marker_size", 6.0)),
                alpha=0.9,
            )
    _configure_alpha_axis(ax, alphas)
    ax.axhline(0.0, color="#636363", linewidth=1.0, linestyle=":", alpha=0.8)
    ax.set_ylabel("Mean expected profit")
    ax.set_title(f"Expected profit by predicted acceptance bin ({split})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def _plot_u_bin_profit_by_alpha(
    bin_rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    *,
    split: str = PLOT_SPLIT,
) -> None:
    rows = [
        row
        for row in bin_rows
        if row.get("bin_type") == "u"
        and row.get("split") == split
        and _has_float(row.get("mean_expected_profit"))
    ]
    if not rows:
        return
    for alpha in sorted({float(row["alpha"]) for row in rows}):
        alpha_rows = [row for row in rows if float(row["alpha"]) == alpha]
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        for estimator, estimator_rows in _rows_by_estimator(alpha_rows):
            selected = sorted(estimator_rows, key=lambda row: int(row["bin_index"]))
            centers = [0.5 * (float(row["bin_low"]) + float(row["bin_high"])) for row in selected]
            profits = [float(row["mean_expected_profit"]) for row in selected]
            style = _estimator_style(estimator)
            ax.plot(
                centers,
                profits,
                label=str(style["label"]),
                color=style["color"],
                marker=style["marker"],
                linewidth=1.8,
                markersize=float(style.get("marker_size", 6.0)),
                alpha=0.9,
            )
        ax.axhline(0.0, color="#636363", linewidth=1.0, linestyle=":", alpha=0.8)
        ax.set_xlim(-alpha, alpha)
        ax.set_xlabel("Final policy u bin center")
        ax.set_ylabel("Mean expected profit")
        ax.set_title(f"Expected profit by final u bin, alpha={alpha:.3f} ({split})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"profit_by_u_bins_{_alpha_label(alpha)}.png", dpi=200)
        plt.close(fig)


def _configure_alpha_axis(ax: plt.Axes, alphas: Sequence[float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(list(alphas))
    ax.set_xticklabels([_alpha_tick(alpha) for alpha in alphas])
    ax.set_xlabel("Softmax bound alpha, with u in [-alpha, alpha]")


def _rows_by_estimator(
    rows: Sequence[Mapping[str, object]],
) -> list[tuple[str, list[Mapping[str, object]]]]:
    estimators = sorted({str(row["estimator"]) for row in rows})
    return [
        (estimator, [row for row in rows if str(row["estimator"]) == estimator])
        for estimator in estimators
    ]


def _alpha_tick(alpha: float) -> str:
    return f"{float(alpha):.3f}".rstrip("0").rstrip(".")


def _has_float(value: object) -> bool:
    if value == "" or value is None:
        return False
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _optional_float(value: object) -> float | str:
    return "" if value is None else float(value)


def main() -> None:
    runs: list[tuple[float, str, ExperimentResult, RunContext]] = []
    for alpha in ALPHA_VALUES:
        runs.append(_run_alpha(alpha))

    final_rows = _collect_final_rows(runs)
    if not final_rows:
        raise ValueError("No alpha sweep rows were produced. Check enabled estimators.")
    bin_rows = _collect_bin_rows_from_artifacts(final_rows)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / PROJECT_NAME / f"alpha_sweep_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(final_rows, output_dir / "softmax_alpha_sweep.csv", _FINAL_FIELDNAMES)
    _write_rows(bin_rows, output_dir / "softmax_alpha_bin_summary.csv", _BIN_FIELDNAMES)
    _write_plots(final_rows, bin_rows, output_dir)

    print(f"Completed {len(runs)} GLM softmax alpha sweep runs for preset '{BASE_PRESET}'.")
    print(f"Wrote alpha sweep summaries and plots to {output_dir}.")


if __name__ == "__main__":
    main()
