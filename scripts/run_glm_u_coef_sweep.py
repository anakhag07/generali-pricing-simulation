"""Run a GLM acceptance beta_u sweep matching the main.py real-data setup."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from data.loader import extract_glm_u_coef
from experiments.results import ExperimentResult
from experiments.sweep_utils import run_preset_sweep
from reporting.visualization import _plot_sweep_pareto_frontier, _plot_sweep_tradeoffs

BASE_PRESET = "real_data_glm_base"
PROJECT_NAME = "glm-u-coef-sweep"
DISPLAY_KEYS = ("u_coef",)
U_COEFS = (-4.0, -5.0, -8.0, -10.0, -20.0)

OVERRIDE_GRID = {
    "u_coef": list(U_COEFS),
    "policy_kind": ["softmax"],
    "policy_preprocessing": ["no_pca"],
    "feature_order": ["linear"],
    "constraint_mode": ["trust_constr"],
    "n_samples": [700000],
    "n_grad_samples": [8],
    "t_steps": [100],
    "enabled_estimators": [("first_order", "finite_difference", "stein_difference")],
    "plot": [True],
    "wandb_enabled": [False],
}

_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def _policy_u_values(result: ExperimentResult, estimator: str) -> np.ndarray:
    objective = result.config.objective
    theta = result.results[estimator].theta
    policy_value = getattr(objective, "policy_value", None)
    if not callable(policy_value):
        return np.asarray([], dtype=float)
    u_values = np.asarray(policy_value(theta, result.x_samples), dtype=float).reshape(-1)
    clip_fn = getattr(objective, "_clip_u", None)
    if callable(clip_fn):
        u_values = np.asarray(clip_fn(u_values), dtype=float).reshape(-1)
    return u_values


def _acceptance_values(result: ExperimentResult, u_values: np.ndarray) -> np.ndarray:
    acceptance_fn = getattr(result.config.objective, "_acceptance_proba", None)
    if not callable(acceptance_fn) or u_values.size == 0:
        return np.asarray([], dtype=float)
    return np.asarray(acceptance_fn(result.x_samples, u_values), dtype=float).reshape(-1)


def _summary(prefix: str, values: np.ndarray) -> dict[str, float | str]:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            f"{prefix}_mean": "",
            f"{prefix}_q05": "",
            f"{prefix}_q25": "",
            f"{prefix}_q50": "",
            f"{prefix}_q75": "",
            f"{prefix}_q95": "",
        }
    quantiles = np.quantile(finite, _QUANTILES)
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_q05": float(quantiles[0]),
        f"{prefix}_q25": float(quantiles[1]),
        f"{prefix}_q50": float(quantiles[2]),
        f"{prefix}_q75": float(quantiles[3]),
        f"{prefix}_q95": float(quantiles[4]),
    }


def _collect_rows(results: Sequence[tuple[str, ExperimentResult]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run_name, result in results:
        objective = result.config.objective
        u_coef = float(objective.u_coef)
        artifact_u_coef = extract_glm_u_coef(objective.acceptance_model)
        for estimator, estimator_result in result.results.items():
            u_values = _policy_u_values(result, estimator)
            acceptance_values = _acceptance_values(result, u_values)
            row: dict[str, object] = {
                "run_name": run_name,
                "estimator": estimator,
                "u_coef": u_coef,
                "artifact_u_coef": float(artifact_u_coef),
                "u": float(estimator_result.u),
                "mean_acceptance": (
                    float(estimator_result.mean_acceptance)
                    if estimator_result.mean_acceptance is not None
                    else ""
                ),
                "value": float(estimator_result.value),
                "runtime_sec": float(estimator_result.time),
                "constraint_violation": (
                    float(estimator_result.constraint_violation)
                    if estimator_result.constraint_violation is not None
                    else ""
                ),
            }
            row.update(_summary("u", u_values))
            row.update(_summary("acceptance", acceptance_values))
            rows.append(row)
    return rows


def _write_rows(rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    csv_path = output_dir / "glm_u_coef_sweep.csv"
    fieldnames = [
        "run_name",
        "estimator",
        "u_coef",
        "artifact_u_coef",
        "u",
        "mean_acceptance",
        "value",
        "runtime_sec",
        "constraint_violation",
        "u_mean",
        "u_q05",
        "u_q25",
        "u_q50",
        "u_q75",
        "u_q95",
        "acceptance_mean",
        "acceptance_q05",
        "acceptance_q25",
        "acceptance_q50",
        "acceptance_q75",
        "acceptance_q95",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_frontier_plots(rows: Sequence[Mapping[str, object]], output_dir: Path) -> None:
    plot_dir = str(output_dir)
    _plot_sweep_tradeoffs(
        rows,
        plot_dir,
        sweep_key="u_coef",
        sweep_label="GLM acceptance beta_u",
        filename="u_coef_vs_u_acceptance.png",
    )
    _plot_sweep_pareto_frontier(
        rows,
        plot_dir,
        sweep_key="u_coef",
        sweep_label="GLM acceptance beta_u",
        y_key="value",
        y_label="Final objective value",
        filename="pareto_objective_acceptance.png",
    )
    _plot_sweep_pareto_frontier(
        rows,
        plot_dir,
        sweep_key="u_coef",
        sweep_label="GLM acceptance beta_u",
        y_key="u",
        y_label="Final u",
        filename="pareto_u_acceptance.png",
    )


def main() -> None:
    results = run_preset_sweep(
        base_preset=BASE_PRESET,
        override_grid=OVERRIDE_GRID,
        project_name=PROJECT_NAME,
        display_keys=DISPLAY_KEYS,
    )
    rows = _collect_rows(results)
    if not rows:
        raise ValueError("No GLM u_coef sweep rows were produced. Check u_coef overrides.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / PROJECT_NAME / f"u_coef_frontier_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(rows, output_dir)
    _write_frontier_plots(rows, output_dir)

    print(f"Completed {len(results)} GLM u_coef sweep runs for preset '{BASE_PRESET}'.")
    print(f"Wrote sweep summary and frontier plots to {output_dir}.")


if __name__ == "__main__":
    main()
