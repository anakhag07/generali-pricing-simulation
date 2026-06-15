"""Evaluate a saved policy under model or observed historical objectives."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from data.dataset_metadata import (
    ID_COLS,
    LOSS_TARGET_COL,
    OBSERVED_CHURN_COL,
    OBSERVED_U_COL,
    PREMIUM_COL,
)
from data.loader import (
    FEATURE_COLS_GLM,
    FEATURE_COLS_XGB,
    dataset_csv_path,
    eligible_csv_row_indices,
    sample_csv_row_indices,
)
from experiments.configs import get_config
from experiments.policy_artifacts import PolicyArtifact, load_policy_artifact


@dataclass(frozen=True)
class HistoricalPolicyEvaluation:
    """Historical-acceptance objective values for a saved policy."""

    estimator: str
    split: str
    theta: np.ndarray
    row_indices: np.ndarray
    ids: pd.DataFrame
    historical_u: np.ndarray
    policy_u: np.ndarray
    is_churn: np.ndarray
    historical_acceptance: np.ndarray
    observed_loss: np.ndarray
    premium: np.ndarray
    policy_revenue: np.ndarray
    objective_contribution: np.ndarray
    objective_kind: str = "historical"


@dataclass(frozen=True)
class ModelPolicyEvaluation:
    """Model-objective values for a saved policy artifact."""

    estimator: str
    split: str
    theta: np.ndarray
    row_indices: np.ndarray
    policy_u: np.ndarray
    model_acceptance: np.ndarray
    model_loss: np.ndarray
    premium: np.ndarray
    policy_revenue: np.ndarray
    objective_contribution: np.ndarray
    objective_kind: str = "model"


def load_summary_payload(summary_json: Path) -> dict[str, Any]:
    """Load a run summary JSON payload."""
    with summary_json.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "config" not in payload or "estimators" not in payload:
        raise ValueError("summary_json must contain 'config' and 'estimators' sections.")
    return payload


def load_estimator_theta(payload: Mapping[str, Any], estimator: str) -> np.ndarray:
    """Return the final theta for an estimator from a summary payload."""
    estimators = payload.get("estimators", {})
    if estimator not in estimators:
        available = ", ".join(sorted(str(name) for name in estimators))
        raise ValueError(f"Estimator '{estimator}' not found. Available: {available}.")
    theta = np.asarray(estimators[estimator]["theta"], dtype=float)
    if theta.ndim != 1 or theta.size == 0:
        raise ValueError(f"Estimator '{estimator}' theta must be a non-empty 1D array.")
    if not np.isfinite(theta).all():
        raise ValueError(f"Estimator '{estimator}' theta contains non-finite values.")
    return theta


def infer_model_type(payload: Mapping[str, Any]) -> str:
    """Infer GLM/XGB model type from summary config."""
    config = payload["config"]
    state_dim = int(config["state_dim"])
    if state_dim == len(FEATURE_COLS_GLM):
        return "glm"
    if state_dim == len(FEATURE_COLS_XGB):
        return "xgb"
    raise ValueError(f"Could not infer model type from state_dim={state_dim}.")


def reconstruct_run_row_indices(payload: Mapping[str, Any], model_type: str) -> np.ndarray:
    """Reconstruct the sampled canonical CSV row positions for a saved run."""
    config = payload["config"]
    n_samples = int(config["n_samples"])
    seed = int(config["seed"])
    eligible = eligible_csv_row_indices(model_type)
    if n_samples == eligible.size:
        try:
            _validate_reconstructed_indices(eligible, config)
            return eligible
        except ValueError:
            pass
    row_indices = sample_csv_row_indices(model_type, n_rows=n_samples, seed=seed)
    _validate_reconstructed_indices(row_indices, config)
    return row_indices


def split_run_row_indices(
    payload: Mapping[str, Any],
    row_indices: np.ndarray,
    split: str,
) -> np.ndarray:
    """Return selected CSV row positions for all/train/test from a summary payload."""
    if split == "all":
        return np.asarray(row_indices, dtype=int).copy()
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train', 'test', or 'all'.")
    config = payload["config"]
    selected = np.asarray(row_indices, dtype=int)
    test_fraction = float(config.get("test_fraction", 0.0))
    if test_fraction == 0.0:
        if split == "test":
            raise ValueError("Saved run has no test split rows.")
        return selected.copy()
    if selected.size < 2:
        raise ValueError("train/test split requires at least two rows.")
    rng = np.random.default_rng(int(config["seed"]))
    shuffled = rng.permutation(selected.size).astype(int)
    n_test = int(round(test_fraction * selected.size))
    n_test = min(max(n_test, 1), selected.size - 1)
    indices = shuffled[:n_test] if split == "test" else shuffled[n_test:]
    return selected[indices].copy()


def _validate_reconstructed_indices(row_indices: np.ndarray, config: Mapping[str, Any]) -> None:
    shape = config.get("x_fixed_row_indices_shape")
    if shape is not None and list(row_indices.shape) != list(shape):
        raise ValueError(
            "Reconstructed row-index shape does not match summary: "
            f"got {list(row_indices.shape)}, expected {shape}."
        )
    head = config.get("x_fixed_row_indices_head")
    if head is not None and row_indices[: len(head)].astype(int).tolist() != list(head):
        raise ValueError("Reconstructed row-index head does not match summary; dataset or seed may differ.")
    min_index = config.get("x_fixed_row_indices_min")
    if min_index is not None and int(row_indices.min()) != int(min_index):
        raise ValueError("Reconstructed row-index minimum does not match summary.")
    max_index = config.get("x_fixed_row_indices_max")
    if max_index is not None and int(row_indices.max()) != int(max_index):
        raise ValueError("Reconstructed row-index maximum does not match summary.")


def build_config_for_saved_policy(payload: Mapping[str, Any], row_indices: np.ndarray, model_type: str) -> object:
    """Rebuild a real-data config matching the saved policy feature mapping."""
    config_payload = payload["config"]
    objective_payload = config_payload["objective"]
    preset = payload.get("run", {}).get("experiment_name") or f"real_data_{model_type}_base"
    overrides = {
        "row_indices": np.asarray(row_indices, dtype=int),
        "n_samples": int(np.asarray(row_indices, dtype=int).size),
        "policy_kind": _policy_kind(objective_payload),
        "feature_order": _feature_order(objective_payload),
        "policy_preprocessing": _policy_preprocessing(objective_payload),
        "loss_source": "observed",
        "constraint_mode": "none",
        "enabled_estimators": ("first_order",),
        "plot": False,
        "verbose": False,
        "wandb_enabled": False,
    }
    u_coef = objective_payload.get("u_coef")
    if u_coef is not None and model_type == "glm":
        overrides["u_coef"] = float(u_coef)
    policy_payload = objective_payload.get("policy", {})
    if policy_payload.get("type") == "SoftmaxPolicy":
        action_low = policy_payload.get("action_low")
        action_high = policy_payload.get("action_high")
        if action_low is not None and action_high is not None:
            overrides["softmax_action_bounds"] = (float(action_low), float(action_high))
    return get_config(str(preset), overrides=overrides)


def build_config_for_policy_artifact(artifact: PolicyArtifact, split: str = "all") -> object:
    """Build a lightweight config-like object from a reloadable policy artifact."""
    return SimpleNamespace(
        objective=artifact.build_objective(),
        x_fixed=artifact.load_x(split=split),
    )


def _policy_kind(objective_payload: Mapping[str, Any]) -> str:
    policy_type = str(objective_payload.get("policy", {}).get("type", ""))
    mapping = {
        "ConstantPolicy": "constant",
        "LinearPolicy": "linear",
        "SoftmaxPolicy": "softmax",
        "MLPPolicy": "mlp",
    }
    if policy_type not in mapping:
        raise ValueError(f"Unsupported saved policy type '{policy_type}'.")
    return mapping[policy_type]


def _feature_order(objective_payload: Mapping[str, Any]) -> str:
    feature_map_type = str(
        objective_payload.get("policy", {}).get("feature_map", {}).get("type", "IdentityFeatureMap")
    )
    mapping = {
        "IdentityFeatureMap": "linear",
        "QuadraticFeatureMap": "quadratic",
        "CubicFeatureMap": "cubic",
        "QuarticFeatureMap": "quartic",
    }
    if feature_map_type not in mapping:
        raise ValueError(f"Unsupported saved feature map type '{feature_map_type}'.")
    return mapping[feature_map_type]


def _policy_preprocessing(objective_payload: Mapping[str, Any]) -> str:
    return "no_pca" if objective_payload.get("policy_preprocessor") is not None else "artifact"


def load_historical_rows(row_indices: np.ndarray) -> pd.DataFrame:
    """Load observed historical diagnostics for canonical CSV row positions."""
    usecols = [*ID_COLS, OBSERVED_U_COL, OBSERVED_CHURN_COL, LOSS_TARGET_COL, PREMIUM_COL]
    df = pd.read_csv(dataset_csv_path(), sep=";", usecols=usecols)
    rows = df.iloc[np.asarray(row_indices, dtype=int)].reset_index(drop=True)
    missing = rows.loc[:, usecols].isna().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        raise ValueError(f"Historical rows contain missing values: {missing.astype(int).to_dict()}")
    return rows


def evaluate_historical_policy_objective(
    *,
    config: object,
    theta: np.ndarray,
    row_indices: np.ndarray,
    historical_rows: pd.DataFrame,
    estimator: str,
    split: str = "all",
    n_rows: int | None = None,
) -> HistoricalPolicyEvaluation:
    """Evaluate saved policy prices with historical acceptance and observed loss."""
    if n_rows is not None:
        if int(n_rows) <= 0:
            raise ValueError("n_rows must be positive when provided.")
        row_indices = row_indices[: int(n_rows)]
        historical_rows = historical_rows.iloc[: int(n_rows)].reset_index(drop=True)

    x_fixed = getattr(config, "x_fixed", None)
    if x_fixed is None:
        raise ValueError("Rebuilt config must provide x_fixed real-data rows.")
    x_eval = x_fixed.iloc[: row_indices.shape[0]].reset_index(drop=True) if hasattr(x_fixed, "iloc") else x_fixed[: row_indices.shape[0]]
    objective = getattr(config, "objective")
    policy_u = np.asarray(objective.policy_value(np.asarray(theta, dtype=float), x_eval), dtype=float).reshape(-1)
    clip_u = getattr(objective, "_clip_u", None)
    if callable(clip_u):
        policy_u = np.asarray(clip_u(policy_u), dtype=float).reshape(-1)

    historical_u = historical_rows[OBSERVED_U_COL].to_numpy(dtype=float)
    is_churn = historical_rows[OBSERVED_CHURN_COL].to_numpy(dtype=float)
    observed_loss = historical_rows[LOSS_TARGET_COL].to_numpy(dtype=float)
    premium = historical_rows[PREMIUM_COL].to_numpy(dtype=float)
    historical_acceptance = 1.0 - is_churn
    policy_revenue = (policy_u + 1.0) * premium
    objective_contribution = historical_acceptance * (observed_loss - policy_revenue)

    for name, values in {
        "historical_u": historical_u,
        "is_churn": is_churn,
        "observed_loss": observed_loss,
        "premium": premium,
        "policy_u": policy_u,
        "objective_contribution": objective_contribution,
    }.items():
        if values.shape != (row_indices.shape[0],):
            raise ValueError(f"{name} must have one value per evaluated row.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values.")

    return HistoricalPolicyEvaluation(
        estimator=estimator,
        split=split,
        theta=np.asarray(theta, dtype=float),
        row_indices=np.asarray(row_indices, dtype=int),
        ids=historical_rows.loc[:, list(ID_COLS)].copy(),
        historical_u=historical_u,
        policy_u=policy_u,
        is_churn=is_churn,
        historical_acceptance=historical_acceptance,
        observed_loss=observed_loss,
        premium=premium,
        policy_revenue=policy_revenue,
        objective_contribution=objective_contribution,
    )


def evaluate_model_policy_objective(
    *,
    artifact: PolicyArtifact,
    split: str = "all",
    n_rows: int | None = None,
) -> ModelPolicyEvaluation:
    """Evaluate a saved policy under the model objective used for training."""
    row_indices = artifact.row_indices(split)
    x_eval = artifact.load_x(split=split)
    if n_rows is not None:
        if int(n_rows) <= 0:
            raise ValueError("n_rows must be positive when provided.")
        n_eval = int(n_rows)
        row_indices = row_indices[:n_eval]
        x_eval = x_eval.iloc[:n_eval].reset_index(drop=True) if hasattr(x_eval, "iloc") else x_eval[:n_eval]
    objective = artifact.build_objective()
    theta = np.asarray(artifact.theta, dtype=float)
    policy_u = np.asarray(objective.policy_value(theta, x_eval), dtype=float).reshape(-1)
    clip_u = getattr(objective, "_clip_u", None)
    if callable(clip_u):
        policy_u = np.asarray(clip_u(policy_u), dtype=float).reshape(-1)
    model_acceptance = np.asarray(objective._acceptance_proba(x_eval, policy_u), dtype=float).reshape(-1)
    model_loss = np.asarray(objective._loss_prediction(x_eval), dtype=float).reshape(-1)
    premium = objective._premium_values(x_eval)
    policy_revenue = (policy_u + 1.0) * premium
    objective_contribution = model_acceptance * (model_loss - policy_revenue)

    for name, values in {
        "policy_u": policy_u,
        "model_acceptance": model_acceptance,
        "model_loss": model_loss,
        "premium": premium,
        "objective_contribution": objective_contribution,
    }.items():
        if values.shape != (row_indices.shape[0],):
            raise ValueError(f"{name} must have one value per evaluated row.")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values.")

    return ModelPolicyEvaluation(
        estimator=artifact.estimator,
        split=split,
        theta=theta,
        row_indices=np.asarray(row_indices, dtype=int),
        policy_u=policy_u,
        model_acceptance=model_acceptance,
        model_loss=model_loss,
        premium=premium,
        policy_revenue=policy_revenue,
        objective_contribution=objective_contribution,
    )


def evaluation_summary(evaluation: HistoricalPolicyEvaluation) -> dict[str, Any]:
    """Return aggregate metrics for a historical-policy evaluation."""
    return {
        "objective_kind": evaluation.objective_kind,
        "split": evaluation.split,
        "estimator": evaluation.estimator,
        "n_rows": int(evaluation.row_indices.size),
        "theta": [float(value) for value in evaluation.theta.tolist()],
        "mean_objective": float(np.mean(evaluation.objective_contribution)),
        "total_objective": float(np.sum(evaluation.objective_contribution)),
        "mean_policy_u": float(np.mean(evaluation.policy_u)),
        "mean_historical_u": float(np.mean(evaluation.historical_u)),
        "mean_historical_acceptance": float(np.mean(evaluation.historical_acceptance)),
        "mean_is_churn": float(np.mean(evaluation.is_churn)),
        "mean_observed_loss": float(np.mean(evaluation.observed_loss)),
        "mean_premium": float(np.mean(evaluation.premium)),
        "mean_policy_revenue": float(np.mean(evaluation.policy_revenue)),
        "mean_policy_revenue_if_accepted": float(
            np.mean(evaluation.historical_acceptance * evaluation.policy_revenue)
        ),
        "accepted_rows": int(np.sum(evaluation.historical_acceptance)),
        "csv_row_index_min": int(np.min(evaluation.row_indices)),
        "csv_row_index_max": int(np.max(evaluation.row_indices)),
    }


def model_evaluation_summary(evaluation: ModelPolicyEvaluation) -> dict[str, Any]:
    """Return aggregate metrics for a model-objective policy replay."""
    return {
        "objective_kind": evaluation.objective_kind,
        "split": evaluation.split,
        "estimator": evaluation.estimator,
        "n_rows": int(evaluation.row_indices.size),
        "theta": [float(value) for value in evaluation.theta.tolist()],
        "objective_value": float(np.mean(evaluation.objective_contribution)),
        "objective_sum": float(np.sum(evaluation.objective_contribution)),
        "mean_u": float(np.mean(evaluation.policy_u)),
        "mean_acceptance": float(np.mean(evaluation.model_acceptance)),
        "projected_loss": float(np.mean(evaluation.model_loss)),
        "projected_revenue": float(np.mean(evaluation.policy_revenue)),
        "mean_premium": float(np.mean(evaluation.premium)),
        "csv_row_index_min": int(np.min(evaluation.row_indices)),
        "csv_row_index_max": int(np.max(evaluation.row_indices)),
    }


def write_outputs(evaluation: HistoricalPolicyEvaluation, output_dir: Path, *, write_per_row: bool = True) -> list[Path]:
    """Write aggregate JSON and optional row-level CSV outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(evaluation_summary(evaluation), handle, indent=2, sort_keys=True)
    outputs.append(summary_path)
    if write_per_row:
        csv_path = output_dir / "per_row.csv"
        _write_per_row_csv(evaluation, csv_path)
        outputs.append(csv_path)
    return outputs


def write_model_outputs(
    evaluation: ModelPolicyEvaluation,
    output_dir: Path,
    *,
    write_per_row: bool = True,
) -> list[Path]:
    """Write model-objective aggregate JSON and optional row-level CSV outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(model_evaluation_summary(evaluation), handle, indent=2, sort_keys=True)
    outputs.append(summary_path)
    if write_per_row:
        csv_path = output_dir / "per_row.csv"
        _write_model_per_row_csv(evaluation, csv_path)
        outputs.append(csv_path)
    return outputs


def _write_per_row_csv(evaluation: HistoricalPolicyEvaluation, csv_path: Path) -> None:
    fieldnames = [
        "csv_row_index",
        *ID_COLS,
        "historical_u",
        "policy_u",
        "is_churn",
        "historical_acceptance",
        "Y_G_Loss",
        "X_policy_premium",
        "policy_revenue",
        "objective_contribution",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(evaluation.row_indices.size):
            row = {
                "csv_row_index": int(evaluation.row_indices[idx]),
                "historical_u": float(evaluation.historical_u[idx]),
                "policy_u": float(evaluation.policy_u[idx]),
                "is_churn": float(evaluation.is_churn[idx]),
                "historical_acceptance": float(evaluation.historical_acceptance[idx]),
                "Y_G_Loss": float(evaluation.observed_loss[idx]),
                "X_policy_premium": float(evaluation.premium[idx]),
                "policy_revenue": float(evaluation.policy_revenue[idx]),
                "objective_contribution": float(evaluation.objective_contribution[idx]),
            }
            for col in ID_COLS:
                row[col] = evaluation.ids.iloc[idx][col]
            writer.writerow(row)


def _write_model_per_row_csv(evaluation: ModelPolicyEvaluation, csv_path: Path) -> None:
    fieldnames = [
        "csv_row_index",
        "policy_u",
        "model_acceptance",
        "model_loss",
        "X_policy_premium",
        "policy_revenue",
        "objective_contribution",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(evaluation.row_indices.size):
            writer.writerow(
                {
                    "csv_row_index": int(evaluation.row_indices[idx]),
                    "policy_u": float(evaluation.policy_u[idx]),
                    "model_acceptance": float(evaluation.model_acceptance[idx]),
                    "model_loss": float(evaluation.model_loss[idx]),
                    "X_policy_premium": float(evaluation.premium[idx]),
                    "policy_revenue": float(evaluation.policy_revenue[idx]),
                    "objective_contribution": float(evaluation.objective_contribution[idx]),
                }
            )


def format_theta(theta: np.ndarray) -> str:
    """Format theta values for terminal verification."""
    return "[" + ", ".join(f"{float(value):.12g}" for value in theta.tolist()) + "]"


def _default_output_dir(summary_json: Path, estimator: str) -> Path:
    return summary_json.parent / "historical_policy_objective" / estimator


def _default_artifact_output_dir(policy_artifact: Path, estimator: str) -> Path:
    return policy_artifact.parent / "historical_policy_objective" / estimator


def _default_artifact_model_output_dir(policy_artifact: Path, estimator: str) -> Path:
    return policy_artifact.parent / "model_policy_objective" / estimator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Path to a run summary.json containing final estimator theta.",
    )
    parser.add_argument(
        "--policy-artifact",
        type=Path,
        default=None,
        help="Path to a saved policies/<estimator>/policy.json artifact. Preferred over summary reconstruction.",
    )
    parser.add_argument("--estimator", default="first_order", help="Estimator theta to evaluate.")
    parser.add_argument(
        "--objective",
        choices=("historical", "model"),
        default="historical",
        help="Objective to evaluate: observed historical diagnostic or trained model objective.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default="all",
        help="Saved run rows to evaluate. 'all' means all selected rows before train/test split.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="Evaluate only the first N rows from the saved run sample. Policy preprocessing still uses all saved-run rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults beside summary.json under historical_policy_objective/<estimator>/.",
    )
    parser.add_argument(
        "--skip-per-row",
        action="store_true",
        help="Write only aggregate summary.json, not per_row.csv.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.policy_artifact is None and args.summary_json is None:
        raise SystemExit("Either --policy-artifact or --summary-json is required.")
    if args.objective == "model" and args.policy_artifact is None:
        raise SystemExit("--objective model requires --policy-artifact for exact replay.")
    estimator = args.estimator
    if args.policy_artifact is not None:
        artifact = load_policy_artifact(args.policy_artifact)
        estimator = artifact.estimator
        theta = artifact.theta
        model_type = artifact.objective.model_type
        row_indices = artifact.row_indices(args.split)
        config = build_config_for_policy_artifact(artifact, split=args.split)
        source_path = args.policy_artifact
    else:
        payload = load_summary_payload(args.summary_json)
        if payload.get("policy_artifacts"):
            print("Warning: summary_json contains policy_artifacts; prefer --policy-artifact for exact policy replay.")
        theta = load_estimator_theta(payload, estimator)
        model_type = infer_model_type(payload)
        selected_row_indices = reconstruct_run_row_indices(payload, model_type)
        row_indices = split_run_row_indices(payload, selected_row_indices, args.split)
        config = build_config_for_saved_policy(payload, row_indices, model_type)
        source_path = args.summary_json
    if args.objective == "model":
        model_evaluation = evaluate_model_policy_objective(
            artifact=artifact,
            split=args.split,
            n_rows=args.n_rows,
        )
        if args.output_dir is not None:
            output_dir = args.output_dir
        else:
            output_dir = _default_artifact_model_output_dir(args.policy_artifact, estimator)
        outputs = write_model_outputs(model_evaluation, output_dir, write_per_row=not args.skip_per_row)
        summary = model_evaluation_summary(model_evaluation)
        print(f"Read theta for estimator '{estimator}' from {source_path}.")
        print(f"Theta used ({estimator}): {format_theta(theta)}")
        print(f"Evaluated {summary['n_rows']} {args.split} rows with the trained model objective.")
        print(f"Mean model objective: {summary['objective_value']:.6f}")
        print(f"Total model objective: {summary['objective_sum']:.6f}")
        print(f"Mean policy u: {summary['mean_u']:.6f}")
        print(f"Mean model acceptance: {summary['mean_acceptance']:.6f}")
        for path in outputs:
            print(f"Wrote {path}")
        return
    historical_rows = load_historical_rows(row_indices)
    evaluation = evaluate_historical_policy_objective(
        config=config,
        theta=theta,
        row_indices=row_indices,
        historical_rows=historical_rows,
        estimator=estimator,
        split=args.split,
        n_rows=args.n_rows,
    )
    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.policy_artifact is not None:
        output_dir = _default_artifact_output_dir(args.policy_artifact, estimator)
    else:
        output_dir = _default_output_dir(args.summary_json, estimator)
    outputs = write_outputs(evaluation, output_dir, write_per_row=not args.skip_per_row)
    summary = evaluation_summary(evaluation)

    print(f"Read theta for estimator '{estimator}' from {source_path}.")
    print(f"Theta used ({estimator}): {format_theta(theta)}")
    print(f"Evaluated {summary['n_rows']} {args.split} rows with historical_acceptance = 1 - is_churn.")
    print(f"Mean historical objective: {summary['mean_objective']:.6f}")
    print(f"Total historical objective: {summary['total_objective']:.6f}")
    print(f"Mean policy u: {summary['mean_policy_u']:.6f}")
    print(f"Mean historical acceptance: {summary['mean_historical_acceptance']:.6f}")
    for path in outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
