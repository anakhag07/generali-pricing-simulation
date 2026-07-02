"""Run a GLM acceptance beta_u sweep for the trust-constrained GLM setup."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from data.loader import extract_glm_u_coef
from experiments.results import ExperimentResult
from experiments.sweep_reporting import (
    timestamped_sweep_output_dir,
    write_rows_csv,
    write_sweep_frontier_plots,
)
from experiments.sweep_utils import run_preset_sweep

BASE_PRESET = "real_data_glm_base"
PROJECT_NAME = "glm-u-coef-sweep"
DISPLAY_KEYS = ("u_coef",)
U_COEFS = (-4.0, -5.0, -8.0, -10.0, -20.0)
N_SAMPLES = 200000

OVERRIDE_GRID = {
    "u_coef": list(U_COEFS),
    "policy_kind": ["softmax"],
    "policy_preprocessing": ["no_pca"],
    "feature_order": ["linear"],
    "constraint_mode": ["trust_constr"],
    "n_samples": [N_SAMPLES],
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


def _collect_rows(results) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sweep_result in results:
        result = sweep_result.result
        objective = result.config.objective
        u_coef = float(objective.u_coef)
        artifact_u_coef = extract_glm_u_coef(objective.acceptance_model)
        for estimator, estimator_result in result.results.items():
            u_values = _policy_u_values(result, estimator)
            acceptance_values = _acceptance_values(result, u_values)
            row: dict[str, object] = {
                "run_name": sweep_result.run_name,
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


_FIELDNAMES = [
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

    output_dir = timestamped_sweep_output_dir(
        project_name=PROJECT_NAME,
        dirname_prefix="u_coef_frontier",
    )
    write_rows_csv(output_dir / "glm_u_coef_sweep.csv", rows, _FIELDNAMES)
    write_sweep_frontier_plots(
        rows,
        output_dir,
        sweep_key="u_coef",
        sweep_label="GLM acceptance beta_u",
        tradeoff_filename="u_coef_vs_u_acceptance.png",
    )

    print(f"Completed {len(results)} GLM u_coef sweep runs for preset '{BASE_PRESET}'.")
    print(f"Wrote sweep summary and frontier plots to {output_dir}.")


if __name__ == "__main__":
    main()
