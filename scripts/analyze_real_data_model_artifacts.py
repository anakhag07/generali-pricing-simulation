"""Compare candidate Generali datasets and model artifacts with the canonical set.

This is a descriptive EDA tool. It does not modify model registries, runtime
configuration, or optimization behavior. Pickles are executable inputs; only
run it on trusted artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score

from data.dataset_metadata import DATASET_PATH, REQUIRED_DATASET_COLUMNS
from data.feature_processor import FeatureProcessor
from data.xgb_logit_spline import load_xgb_logit_spline_acceptance
from experiments.paths import results_root


_REPO_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_MODEL_DIR = _REPO_ROOT / "src" / "data" / "models"
_KEY_NUMERIC_COLUMNS = (
    "is_churn",
    "U",
    "Y_G_Loss",
    "X_policy_premium",
    "X_upcoming_premium",
    "X_age",
    "X_vehicle_age",
    "X_claim_tot_value",
)
_KNOWN_CATEGORICAL_COLUMNS = (
    "X_fuel_type_vehicle",
    "X_gender",
    "X_customer_segment",
    "X_installment",
    "X_vehicle_type",
    "X_district",
    "X_distr_channel",
    "X_vehicle_weight",
    "X_vehicle_power",
)
_ACCEPTANCE_TARGET = "is_churn"
_LOSS_TARGET = "Y_G_Loss"
_ACTION_COLUMN = "U"
_ID_COLUMN = "id"


@dataclass(frozen=True)
class ArtifactView:
    """Normalized read-only view over a CV or selected-best-fold artifact."""

    name: str
    family: str
    role: str
    path: Path
    raw: Mapping[str, Any]
    model: Any
    preprocessor: FeatureProcessor
    x_feature_cols: tuple[str, ...]
    u_cols: tuple[str, ...]
    source_format: str
    selected_fold: int


class _LegacyParametricCurve:
    """Compatibility target for the legacy pickled parametric curves."""

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return self.func(np.asarray(values, dtype=float), *self.params)


class _LegacySmoothedXGBoostWrapper:
    """Compatibility target exposing only the state needed for EDA."""

    def covered_policies(self) -> set[str]:
        return set(self._curves)


def _legacy_sigmoid_with_shift(
    x: np.ndarray,
    k: float,
    m: float,
    d: float,
) -> np.ndarray:
    return np.clip(d + 1.0 / (1.0 + np.exp(-k * (x - m))), 0.0, 1.0)


class _AnalysisUnpickler(pickle.Unpickler):
    """Map trusted source-repository classes onto local compatibility targets."""

    def find_class(self, module: str, name: str) -> Any:
        if name == "FeatureProcessor" and module in {"__main__", "preprocessing"}:
            return FeatureProcessor
        if name == "SmoothedXGBoostWrapper" and module in {
            "__main__",
            "black_box_objective",
            "smoothing_wrapper",
        }:
            return _LegacySmoothedXGBoostWrapper
        if name == "_ParametricCurve" and module.endswith("curve_specifications"):
            return _LegacyParametricCurve
        if name == "_sigmoid_with_shift" and module.endswith("curve_specifications"):
            return _legacy_sigmoid_with_shift
        return super().find_class(module, name)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pickle(path: str | Path) -> Any:
    artifact_path = Path(path)
    with artifact_path.open("rb") as handle:
        return _AnalysisUnpickler(handle).load()


def normalize_artifact(
    raw: Mapping[str, Any],
    *,
    name: str,
    family: str,
    role: str,
    path: str | Path,
) -> ArtifactView:
    """Normalize supported CV and selected-best-fold dictionaries."""
    if not isinstance(raw, Mapping):
        raise TypeError(f"{name} must contain a dictionary, got {type(raw).__name__}.")

    if "model" in raw:
        source_format = "selected_best_fold"
        model = raw["model"]
        selected_fold = int(raw.get("best_fold", 0))
        preprocessor_bundle = raw.get("preprocessor")
    elif "trained_models" in raw:
        models = raw["trained_models"]
        if not isinstance(models, Sequence) or not models:
            raise ValueError(f"{name} has no trained_models.")
        source_format = "cv_first_fold"
        model = models[0]
        selected_fold = 0
        trained_preprocessors = raw.get("trained_preprocessors")
        if isinstance(trained_preprocessors, Sequence) and trained_preprocessors:
            preprocessor_bundle = trained_preprocessors[0]
        else:
            preprocessor_bundle = raw.get("preprocessor")
    else:
        raise ValueError(f"{name} is neither a CV nor selected-best-fold artifact.")

    u_cols: Sequence[str] = ()
    x_feature_cols: Sequence[str] | None = None
    if isinstance(preprocessor_bundle, Mapping):
        preprocessor = preprocessor_bundle.get("preprocessor")
        x_feature_cols = (
            preprocessor_bundle.get("x_feature_cols")
            or preprocessor_bundle.get("feature_cols")
        )
        u_cols = tuple(preprocessor_bundle.get("u_cols", ()))
    else:
        preprocessor = preprocessor_bundle
        x_feature_cols = raw.get("x_feature_cols")

    if not isinstance(preprocessor, FeatureProcessor):
        raise TypeError(f"{name} does not contain a supported FeatureProcessor.")

    model_features = tuple(str(value) for value in raw.get("model_features", ()))
    if not u_cols and _ACTION_COLUMN in model_features:
        u_cols = (_ACTION_COLUMN,)
    if x_feature_cols is None:
        x_feature_cols = tuple(col for col in model_features if col not in set(u_cols))
    if not x_feature_cols:
        raise ValueError(f"{name} does not identify raw X feature columns.")

    return ArtifactView(
        name=name,
        family=family,
        role=role,
        path=Path(path),
        raw=raw,
        model=model,
        preprocessor=preprocessor,
        x_feature_cols=tuple(str(col) for col in x_feature_cols),
        u_cols=tuple(str(col) for col in u_cols),
        source_format=source_format,
        selected_fold=selected_fold,
    )


def load_artifact_view(
    path: str | Path,
    *,
    name: str,
    family: str,
    role: str,
) -> ArtifactView:
    """Load a trusted pickle and return a normalized artifact view."""
    return normalize_artifact(
        _load_pickle(path),
        name=name,
        family=family,
        role=role,
        path=path,
    )


def _json_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return json.dumps(value, default=str, sort_keys=True)


def _comparison_row(
    section: str,
    metric: str,
    column: str,
    reference: Any,
    candidate: Any,
) -> dict[str, object]:
    reference_text = _json_value(reference)
    candidate_text = _json_value(candidate)
    return {
        "section": section,
        "metric": metric,
        "column": column,
        "reference": reference_text,
        "candidate": candidate_text,
        "match": reference_text == candidate_text,
    }


def _dataset_rows(frame: pd.DataFrame) -> dict[tuple[str, str], Any]:
    values: dict[tuple[str, str], Any] = {
        ("structure", "rows"): int(frame.shape[0]),
        ("structure", "columns"): int(frame.shape[1]),
        ("structure", "column_order"): list(frame.columns),
        ("integrity", "duplicate_rows"): int(frame.duplicated().sum()),
    }
    for id_col in ("dummy_id", _ID_COLUMN):
        if id_col in frame:
            values[("integrity", f"{id_col}_unique")] = int(frame[id_col].nunique(dropna=False))
            values[("integrity", f"{id_col}_duplicates")] = int(frame[id_col].duplicated().sum())

    missing_required = [col for col in REQUIRED_DATASET_COLUMNS if col not in frame]
    if missing_required:
        values[("eligibility", "missing_required_columns")] = missing_required
        values[("eligibility", "eligible_rows")] = 0
    else:
        eligible = frame.loc[:, list(REQUIRED_DATASET_COLUMNS)].notna().all(axis=1)
        values[("eligibility", "missing_required_columns")] = []
        values[("eligibility", "eligible_rows")] = int(eligible.sum())
        values[("eligibility", "excluded_rows")] = int((~eligible).sum())
    return values


def compare_dataset_frames(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    reference_sha256: str = "",
    candidate_sha256: str = "",
    reference_size: int | None = None,
    candidate_size: int | None = None,
) -> pd.DataFrame:
    """Return a long-form exact and descriptive dataset comparison."""
    rows: list[dict[str, object]] = [
        _comparison_row("file", "sha256", "", reference_sha256, candidate_sha256),
        _comparison_row("file", "size_bytes", "", reference_size, candidate_size),
    ]
    reference_values = _dataset_rows(reference)
    candidate_values = _dataset_rows(candidate)
    for section, metric in sorted(set(reference_values) | set(candidate_values)):
        rows.append(
            _comparison_row(
                section,
                metric,
                "",
                reference_values.get((section, metric)),
                candidate_values.get((section, metric)),
            )
        )

    all_columns = list(dict.fromkeys([*reference.columns, *candidate.columns]))
    for column in all_columns:
        reference_series = reference[column] if column in reference else None
        candidate_series = candidate[column] if column in candidate else None
        rows.append(
            _comparison_row(
                "dtype",
                "inferred_dtype",
                column,
                str(reference_series.dtype) if reference_series is not None else None,
                str(candidate_series.dtype) if candidate_series is not None else None,
            )
        )
        rows.append(
            _comparison_row(
                "missingness",
                "missing_count",
                column,
                int(reference_series.isna().sum()) if reference_series is not None else None,
                int(candidate_series.isna().sum()) if candidate_series is not None else None,
            )
        )
        rows.append(
            _comparison_row(
                "cardinality",
                "unique_count",
                column,
                int(reference_series.nunique(dropna=False))
                if reference_series is not None
                else None,
                int(candidate_series.nunique(dropna=False))
                if candidate_series is not None
                else None,
            )
        )
        if column in _KNOWN_CATEGORICAL_COLUMNS:
            ref_levels = (
                sorted(reference_series.dropna().astype(str).unique().tolist())
                if reference_series is not None
                else None
            )
            cand_levels = (
                sorted(candidate_series.dropna().astype(str).unique().tolist())
                if candidate_series is not None
                else None
            )
            rows.append(
                _comparison_row(
                    "categorical_levels",
                    "levels",
                    column,
                    ref_levels,
                    cand_levels,
                )
            )

    quantiles = {"min": 0.0, "p01": 0.01, "p05": 0.05, "p25": 0.25, "p50": 0.5,
                 "p75": 0.75, "p95": 0.95, "p99": 0.99, "max": 1.0}
    for column in _KEY_NUMERIC_COLUMNS:
        if column not in reference.columns and column not in candidate.columns:
            continue
        ref_numeric = (
            pd.to_numeric(reference[column], errors="coerce") if column in reference else None
        )
        cand_numeric = (
            pd.to_numeric(candidate[column], errors="coerce") if column in candidate else None
        )
        for metric in ("count", "mean", "std"):
            ref_value = getattr(ref_numeric, metric)() if ref_numeric is not None else None
            cand_value = getattr(cand_numeric, metric)() if cand_numeric is not None else None
            rows.append(
                _comparison_row(
                    "numeric_summary",
                    metric,
                    column,
                    float(ref_value) if ref_value is not None else None,
                    float(cand_value) if cand_value is not None else None,
                )
            )
        for metric, quantile in quantiles.items():
            ref_value = ref_numeric.quantile(quantile) if ref_numeric is not None else None
            cand_value = cand_numeric.quantile(quantile) if cand_numeric is not None else None
            rows.append(
                _comparison_row(
                    "numeric_summary",
                    metric,
                    column,
                    float(ref_value) if ref_value is not None else None,
                    float(cand_value) if cand_value is not None else None,
                )
            )
    return pd.DataFrame(rows)


def compare_dataset_files(
    reference_path: str | Path,
    candidate_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read two CSVs and return comparison, reference frame, and candidate frame."""
    reference_path = Path(reference_path)
    candidate_path = Path(candidate_path)
    reference_sha = sha256_file(reference_path)
    candidate_sha = sha256_file(candidate_path)
    reference = pd.read_csv(reference_path, sep=";")
    candidate = (
        reference
        if reference_sha == candidate_sha
        else pd.read_csv(candidate_path, sep=";")
    )
    summary = compare_dataset_frames(
        reference,
        candidate,
        reference_sha256=reference_sha,
        candidate_sha256=candidate_sha,
        reference_size=reference_path.stat().st_size,
        candidate_size=candidate_path.stat().st_size,
    )
    return summary, reference, candidate


def _artifact_model_frame(
    artifact: ArtifactView,
    raw_frame: pd.DataFrame,
    *,
    u: np.ndarray | None = None,
) -> pd.DataFrame:
    missing = [col for col in artifact.x_feature_cols if col not in raw_frame]
    if missing:
        raise ValueError(f"{artifact.name} input is missing columns: {missing}")
    transformed = artifact.preprocessor.transform(
        raw_frame.loc[:, list(artifact.x_feature_cols)].copy()
    ).reset_index(drop=True)
    if artifact.u_cols:
        if artifact.u_cols != (_ACTION_COLUMN,):
            raise ValueError(f"{artifact.name} uses unsupported action columns {artifact.u_cols}.")
        action = (
            raw_frame[_ACTION_COLUMN].to_numpy(dtype=float)
            if u is None
            else np.asarray(u, dtype=float)
        )
        if action.shape != (raw_frame.shape[0],):
            raise ValueError("u must contain one action per row.")
        transformed = pd.concat(
            [transformed, pd.DataFrame({_ACTION_COLUMN: action})],
            axis=1,
        )
    if hasattr(artifact.model, "feature_names_in_"):
        transformed = transformed.reindex(
            columns=[str(value) for value in artifact.model.feature_names_in_]
        )
    return transformed


def predict_artifact(
    artifact: ArtifactView,
    raw_frame: pd.DataFrame,
    *,
    u: np.ndarray | None = None,
) -> np.ndarray:
    """Predict direct acceptance or loss from a normalized artifact."""
    model_frame = _artifact_model_frame(artifact, raw_frame, u=u)
    if artifact.role == "acceptance":
        prediction = artifact.model.predict_proba(model_frame)[:, 1]
    elif artifact.role == "loss":
        prediction = artifact.model.predict(model_frame)
    else:
        raise ValueError(f"Unknown artifact role {artifact.role!r}.")
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (raw_frame.shape[0],) or not np.isfinite(prediction).all():
        raise ValueError(f"{artifact.name} returned invalid predictions.")
    return prediction


def artifact_inventory(artifacts: Sequence[ArtifactView]) -> pd.DataFrame:
    """Describe artifact schema, model, preprocessing, and training metadata."""
    rows: list[dict[str, object]] = []
    parameter_names = (
        "objective",
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "reg_alpha",
        "reg_lambda",
        "random_state",
    )
    for artifact in artifacts:
        try:
            params = artifact.model.get_params()
        except AttributeError:
            params = {}
        selected_params = {
            name: params[name] for name in parameter_names if name in params
        }
        feature_names = [
            str(value) for value in getattr(artifact.model, "feature_names_in_", ())
        ]
        feature_importance: list[tuple[str, float]] = []
        if hasattr(artifact.model, "feature_importances_") and feature_names:
            importance = np.asarray(artifact.model.feature_importances_, dtype=float)
            order = np.argsort(importance)[::-1][:10]
            feature_importance = [
                (feature_names[index], float(importance[index])) for index in order
            ]
        rows.append(
            {
                "artifact": artifact.name,
                "family": artifact.family,
                "role": artifact.role,
                "path": str(artifact.path),
                "sha256": sha256_file(artifact.path),
                "size_bytes": artifact.path.stat().st_size,
                "source_format": artifact.source_format,
                "model_class": type(artifact.model).__name__,
                "selected_fold_zero_based": artifact.selected_fold,
                "stored_models": len(artifact.raw.get("trained_models", ())) or 1,
                "n_folds": artifact.raw.get("n_folds"),
                "dataset": artifact.raw.get("dataset"),
                "timestamp": artifact.raw.get("timestamp"),
                "target": artifact.raw.get("target"),
                "best_iteration": getattr(artifact.model, "best_iteration", None),
                "n_model_features": len(feature_names),
                "model_features": _json_value(feature_names),
                "x_feature_cols": _json_value(artifact.x_feature_cols),
                "u_cols": _json_value(artifact.u_cols),
                "numeric_cols": _json_value(
                    getattr(artifact.preprocessor, "numeric_cols_", ())
                ),
                "categorical_cols": _json_value(
                    getattr(artifact.preprocessor, "categorical_cols_", ())
                ),
                "model_params": _json_value(selected_params),
                "top_feature_importance": _json_value(feature_importance),
            }
        )
    return pd.DataFrame(rows)


def artifact_cv_metrics(artifacts: Sequence[ArtifactView]) -> pd.DataFrame:
    """Flatten stored fold-level metrics without treating them as a new holdout."""
    rows: list[dict[str, object]] = []
    for artifact in artifacts:
        cv_results = artifact.raw.get("cv_results")
        if not isinstance(cv_results, pd.DataFrame):
            continue
        for row_index, values in cv_results.reset_index(drop=True).iterrows():
            base = {
                "artifact": artifact.name,
                "family": artifact.family,
                "role": artifact.role,
                "row_index": int(row_index),
            }
            base.update({str(key): value for key, value in values.items()})
            rows.append(base)
    return pd.DataFrame(rows)


def deterministic_common_sample(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    artifacts: Sequence[ArtifactView],
    *,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    """Select a deterministic complete sample whose IDs occur in both datasets."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive.")
    required = {
        _ACTION_COLUMN,
        _ACCEPTANCE_TARGET,
        _LOSS_TARGET,
        *[col for artifact in artifacts for col in artifact.x_feature_cols],
    }
    missing = sorted(required.difference(candidate.columns))
    if missing:
        raise ValueError(f"Candidate dataset is missing analysis columns: {missing}")
    eligible = candidate.loc[:, sorted(required)].notna().all(axis=1)
    common = candidate.loc[eligible].copy()
    if _ID_COLUMN in reference and _ID_COLUMN in candidate:
        reference_ids = set(reference[_ID_COLUMN].dropna().astype(str))
        common = common[common[_ID_COLUMN].astype(str).isin(reference_ids)]
    population = int(common.shape[0])
    if sample_size > population:
        raise ValueError(
            f"sample_size={sample_size} exceeds {population} complete common rows."
        )
    return common.sample(sample_size, random_state=seed).reset_index(drop=True), population


def prediction_metrics(
    artifacts: Sequence[ArtifactView],
    sample: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Score descriptive acceptance/loss metrics on the fixed common sample."""
    rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}
    acceptance_target = 1 - sample[_ACCEPTANCE_TARGET].to_numpy(dtype=int)
    loss_target = sample[_LOSS_TARGET].to_numpy(dtype=float)
    for artifact in artifacts:
        prediction = predict_artifact(artifact, sample)
        predictions[artifact.name] = prediction
        row: dict[str, object] = {
            "artifact": artifact.name,
            "family": artifact.family,
            "role": artifact.role,
            "n": sample.shape[0],
            "mean_prediction": float(np.mean(prediction)),
            "std_prediction": float(np.std(prediction)),
            "p01_prediction": float(np.quantile(prediction, 0.01)),
            "p50_prediction": float(np.quantile(prediction, 0.50)),
            "p99_prediction": float(np.quantile(prediction, 0.99)),
            "evaluation_status": "descriptive_non_holdout",
        }
        if artifact.role == "acceptance":
            row.update(
                {
                    "roc_auc": float(roc_auc_score(acceptance_target, prediction)),
                    "log_loss": float(log_loss(acceptance_target, prediction)),
                    "brier_score": float(
                        brier_score_loss(acceptance_target, prediction)
                    ),
                    "mae": np.nan,
                    "rmse": np.nan,
                    "pct_negative": np.nan,
                }
            )
        else:
            residual = prediction - loss_target
            row.update(
                {
                    "roc_auc": np.nan,
                    "log_loss": np.nan,
                    "brier_score": np.nan,
                    "mae": float(mean_absolute_error(loss_target, prediction)),
                    "rmse": float(np.sqrt(np.mean(np.square(residual)))),
                    "pct_negative": float(np.mean(prediction < 0.0) * 100.0),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows), predictions


def pairwise_prediction_metrics(
    artifacts: Sequence[ArtifactView],
    predictions: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Compare models with the same role on identical sampled rows."""
    rows: list[dict[str, object]] = []
    for role in ("acceptance", "loss"):
        names = [artifact.name for artifact in artifacts if artifact.role == role]
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                left = np.asarray(predictions[left_name], dtype=float)
                right = np.asarray(predictions[right_name], dtype=float)
                delta = left - right
                rows.append(
                    {
                        "role": role,
                        "left_artifact": left_name,
                        "right_artifact": right_name,
                        "n": left.size,
                        "mean_delta": float(np.mean(delta)),
                        "mae": float(np.mean(np.abs(delta))),
                        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
                        "correlation": float(np.corrcoef(left, right)[0, 1]),
                        "max_abs_delta": float(np.max(np.abs(delta))),
                    }
                )
    return pd.DataFrame(rows)


def summarize_action_predictions(
    *,
    model_name: str,
    cohort: str,
    action_grid: np.ndarray,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """Summarize a policy-by-action acceptance prediction matrix."""
    prediction_matrix = np.asarray(predictions, dtype=float)
    action_grid = np.asarray(action_grid, dtype=float)
    if prediction_matrix.ndim != 2 or prediction_matrix.shape[1] != action_grid.size:
        raise ValueError("predictions must have shape (n_rows, n_actions).")
    if not np.isfinite(prediction_matrix).all():
        raise ValueError("Action-grid predictions must be finite.")
    rows = []
    for index, action in enumerate(action_grid):
        values = prediction_matrix[:, index]
        rows.append(
            {
                "model": model_name,
                "cohort": cohort,
                "u": float(action),
                "n": values.size,
                "mean_acceptance": float(np.mean(values)),
                "std_acceptance": float(np.std(values)),
                "p10_acceptance": float(np.quantile(values, 0.10)),
                "p50_acceptance": float(np.quantile(values, 0.50)),
                "p90_acceptance": float(np.quantile(values, 0.90)),
            }
        )
    return pd.DataFrame(rows)


def raw_acceptance_action_grid(
    artifacts: Sequence[ArtifactView],
    sample: pd.DataFrame,
    action_grid: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Evaluate each raw acceptance artifact across a shared action grid."""
    summaries = []
    matrices: dict[str, np.ndarray] = {}
    for artifact in artifacts:
        if artifact.role != "acceptance":
            continue
        matrix = np.column_stack(
            [
                predict_artifact(
                    artifact,
                    sample,
                    u=np.full(sample.shape[0], action, dtype=float),
                )
                for action in action_grid
            ]
        )
        matrices[artifact.name] = matrix
        summaries.append(
            summarize_action_predictions(
                model_name=artifact.name,
                cohort="deterministic_common_sample",
                action_grid=action_grid,
                predictions=matrix,
            )
        )
    return pd.concat(summaries, ignore_index=True), matrices


def extract_sigmoid_parameters(
    wrapper: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract sorted policy IDs and ``(k, m, d)`` from a legacy wrapper."""
    if getattr(wrapper, "_function_name", None) != "sigmoid_with_shift":
        raise ValueError("Only sigmoid_with_shift smoothing artifacts are supported.")
    curves = getattr(wrapper, "_curves", None)
    if not isinstance(curves, Mapping) or not curves:
        raise ValueError("Smoothing artifact must contain a non-empty _curves mapping.")
    policy_ids = np.asarray(sorted(str(policy_id) for policy_id in curves), dtype=str)
    parameters = []
    for policy_id in policy_ids:
        curve = curves[policy_id]
        values = np.asarray(getattr(curve, "params", ()), dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"Policy {policy_id} has invalid sigmoid parameters.")
        parameters.append(values)
    return policy_ids, np.stack(parameters)


def sigmoid_acceptance_matrix(
    parameters: np.ndarray,
    action_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate direct acceptance for legacy churn sigmoid parameters."""
    parameters = np.asarray(parameters, dtype=float)
    action_grid = np.asarray(action_grid, dtype=float)
    if parameters.ndim != 2 or parameters.shape[1] != 3:
        raise ValueError("parameters must have shape (n_policies, 3).")
    k = parameters[:, 0, None]
    m = parameters[:, 1, None]
    d = parameters[:, 2, None]
    churn = np.clip(d + 1.0 / (1.0 + np.exp(-k * (action_grid - m))), 0.0, 1.0)
    return 1.0 - churn


def _unique_rows_for_policy_ids(
    frame: pd.DataFrame,
    policy_ids: np.ndarray,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    id_values = frame[_ID_COLUMN].astype("string").astype(str)
    positions: dict[str, list[int]] = {}
    requested = set(policy_ids.tolist())
    for position, policy_id in enumerate(id_values):
        if policy_id in requested:
            positions.setdefault(policy_id, []).append(position)
    missing = [policy_id for policy_id in policy_ids if policy_id not in positions]
    duplicate = [
        policy_id for policy_id in policy_ids if len(positions.get(policy_id, ())) != 1
    ]
    valid = [
        frame.iloc[positions[policy_id][0]]
        for policy_id in policy_ids
        if policy_id in positions and len(positions[policy_id]) == 1
    ]
    return pd.DataFrame(valid).reset_index(drop=True), missing, duplicate


def _embedded_raw_acceptance(
    wrapper: Any,
    frame: pd.DataFrame,
) -> np.ndarray:
    bundle = getattr(wrapper, "_prep", None)
    model = getattr(wrapper, "_model", None)
    if not isinstance(bundle, Mapping) or model is None:
        raise ValueError("Smoothing wrapper lacks its embedded raw XGB state.")
    preprocessor = bundle.get("preprocessor")
    x_feature_cols = bundle.get("x_feature_cols")
    if not isinstance(preprocessor, FeatureProcessor) or not x_feature_cols:
        raise ValueError("Smoothing wrapper has invalid embedded preprocessing state.")
    transformed = preprocessor.transform(frame.loc[:, list(x_feature_cols)]).reset_index(
        drop=True
    )
    transformed = pd.concat(
        [transformed, frame.loc[:, [_ACTION_COLUMN]].reset_index(drop=True)],
        axis=1,
    )
    if hasattr(model, "feature_names_in_"):
        transformed = transformed.reindex(
            columns=[str(value) for value in model.feature_names_in_]
        )
    return np.asarray(model.predict_proba(transformed)[:, 1], dtype=float)


def _metric_rows(
    artifact: str,
    metrics: Mapping[str, Any],
    *,
    detail: str = "",
) -> list[dict[str, object]]:
    return [
        {
            "artifact": artifact,
            "metric": metric,
            "value": value,
            "detail": detail,
        }
        for metric, value in metrics.items()
    ]


def smoothing_analysis(
    *,
    candidate_smoothing_path: Path,
    candidate_dataset: pd.DataFrame,
    reference_dataset: pd.DataFrame,
    current_spline_path: Path,
    candidate_acceptance: ArtifactView,
    action_grid: np.ndarray,
    eligible_population: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Analyze new sigmoid curves and the current portable spline cohort."""
    wrapper = _load_pickle(candidate_smoothing_path)
    policy_ids, parameters = extract_sigmoid_parameters(wrapper)
    covered_rows, missing_ids, duplicate_ids = _unique_rows_for_policy_ids(
        candidate_dataset, policy_ids
    )
    if missing_ids or duplicate_ids:
        raise ValueError(
            "Candidate smoothing IDs must resolve uniquely: "
            f"missing={missing_ids[:5]}, duplicate={duplicate_ids[:5]}"
        )
    candidate_smooth = sigmoid_acceptance_matrix(parameters, action_grid)

    expanded = covered_rows.iloc[
        np.repeat(np.arange(covered_rows.shape[0]), action_grid.size)
    ].reset_index(drop=True)
    expanded[_ACTION_COLUMN] = np.tile(action_grid, covered_rows.shape[0])
    embedded_raw = _embedded_raw_acceptance(wrapper, expanded).reshape(
        covered_rows.shape[0], action_grid.size
    )
    candidate_saved_raw = predict_artifact(
        candidate_acceptance,
        expanded,
    ).reshape(covered_rows.shape[0], action_grid.size)

    current_spline = load_xgb_logit_spline_acceptance(current_spline_path)
    current_rows = reference_dataset.iloc[
        current_spline.covered_row_indices()
    ].reset_index(drop=True)
    current_expanded = current_rows.iloc[
        np.repeat(np.arange(current_rows.shape[0]), action_grid.size)
    ].reset_index(drop=True)
    tiled_action = np.tile(action_grid, current_rows.shape[0])
    current_smooth = current_spline.predict_acceptance(
        current_expanded, tiled_action
    ).reshape(current_rows.shape[0], action_grid.size)

    current_ids = set(current_rows[_ID_COLUMN].astype(str))
    candidate_ids = set(policy_ids.tolist())
    smooth_minus_raw = candidate_smooth - embedded_raw
    embedded_minus_saved = embedded_raw - candidate_saved_raw
    legacy_float_breaks = False
    try:
        float(np.asarray(wrapper._curves[policy_ids[0]](np.asarray([0.08]))))
    except TypeError:
        legacy_float_breaks = True

    coverage_rows: list[dict[str, object]] = []
    coverage_rows.extend(
        _metric_rows(
            "candidate_xgb_smoothed",
            {
                "covered_policy_ids": policy_ids.size,
                "matching_dataset_rows": covered_rows.shape[0],
                "missing_policy_ids": len(missing_ids),
                "duplicate_policy_ids": len(duplicate_ids),
                "eligible_population": eligible_population,
                "coverage_fraction": policy_ids.size / eligible_population,
                "current_cohort_id_intersection": len(candidate_ids & current_ids),
                "mean_acceptance_u_0": float(candidate_smooth[:, 0].mean()),
                "mean_acceptance_u_max": float(candidate_smooth[:, -1].mean()),
                "mean_delta_u_max_minus_0": float(
                    np.mean(candidate_smooth[:, -1] - candidate_smooth[:, 0])
                ),
                "pct_curves_monotone_nonincreasing": float(
                    np.mean(np.all(np.diff(candidate_smooth, axis=1) <= 1e-12, axis=1))
                    * 100.0
                ),
                "smooth_vs_embedded_raw_mae": float(
                    np.mean(np.abs(smooth_minus_raw))
                ),
                "smooth_vs_embedded_raw_rmse": float(
                    np.sqrt(np.mean(np.square(smooth_minus_raw)))
                ),
                "smooth_vs_embedded_raw_max_abs": float(
                    np.max(np.abs(smooth_minus_raw))
                ),
                "embedded_raw_vs_saved_candidate_mae": float(
                    np.mean(np.abs(embedded_minus_saved))
                ),
                "embedded_raw_vs_saved_candidate_rmse": float(
                    np.sqrt(np.mean(np.square(embedded_minus_saved)))
                ),
                "embedded_raw_same_booster_as_saved_candidate": bool(
                    wrapper._model.get_booster().save_raw()
                    == candidate_acceptance.model.get_booster().save_raw()
                ),
                "legacy_numpy_scalar_conversion_breaks": legacy_float_breaks,
            },
            detail="Candidate cohort; direct portable evaluation of fitted sigmoid parameters.",
        )
    )
    for parameter_index, parameter_name in enumerate(("k", "m", "d")):
        for quantile in (0.0, 0.01, 0.25, 0.50, 0.75, 0.99, 1.0):
            coverage_rows.extend(
                _metric_rows(
                    "candidate_xgb_smoothed",
                    {
                        f"sigmoid_{parameter_name}_q{int(quantile * 100):02d}": float(
                            np.quantile(parameters[:, parameter_index], quantile)
                        )
                    },
                )
            )
    coverage_rows.extend(
        _metric_rows(
            "current_xgb_logit_spline",
            {
                "covered_policy_ids": current_rows.shape[0],
                "current_cohort_id_intersection": len(candidate_ids & current_ids),
                "mean_acceptance_u_0": float(current_smooth[:, 0].mean()),
                "mean_acceptance_u_max": float(current_smooth[:, -1].mean()),
                "mean_delta_u_max_minus_0": float(
                    np.mean(current_smooth[:, -1] - current_smooth[:, 0])
                ),
                "pct_curves_monotone_nonincreasing": float(
                    np.mean(np.all(np.diff(current_smooth, axis=1) <= 1e-12, axis=1))
                    * 100.0
                ),
            },
            detail="Current portable spline cohort; not customer-aligned with candidate.",
        )
    )

    action_summary = pd.concat(
        [
            summarize_action_predictions(
                model_name="candidate_xgb_smoothed",
                cohort="candidate_200_covered_ids",
                action_grid=action_grid,
                predictions=candidate_smooth,
            ),
            summarize_action_predictions(
                model_name="candidate_smoothing_embedded_raw_xgb",
                cohort="candidate_200_covered_ids",
                action_grid=action_grid,
                predictions=embedded_raw,
            ),
            summarize_action_predictions(
                model_name="current_xgb_logit_spline",
                cohort="current_200_covered_ids",
                action_grid=action_grid,
                predictions=current_smooth,
            ),
        ],
        ignore_index=True,
    )
    matrices = {
        "candidate_xgb_smoothed": candidate_smooth,
        "candidate_smoothing_embedded_raw_xgb": embedded_raw,
        "current_xgb_logit_spline": current_smooth,
        "candidate_smooth_minus_embedded_raw": smooth_minus_raw,
    }
    return pd.DataFrame(coverage_rows), action_summary, matrices


def _plot_acceptance_models(
    metrics: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    action_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    acceptance_names = metrics.loc[metrics["role"] == "acceptance", "artifact"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name in acceptance_names:
        axes[0].hist(
            predictions[name],
            bins=45,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=name,
        )
    axes[0].set_xlabel("Predicted acceptance at observed U")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    raw = action_summary[
        action_summary["cohort"] == "deterministic_common_sample"
    ]
    for name, group in raw.groupby("model", sort=False):
        axes[1].plot(
            group["u"],
            group["mean_acceptance"],
            marker="o",
            markersize=3,
            label=name,
        )
    axes[1].set_xlabel("Counterfactual U")
    axes[1].set_ylabel("Mean predicted acceptance")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_loss_models(
    metrics: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    output_path: Path,
) -> None:
    loss_names = metrics.loc[metrics["role"] == "loss", "artifact"].tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name in loss_names:
        axes[0].hist(
            predictions[name],
            bins=45,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=name,
        )
    axes[0].set_xlabel("Predicted financial loss")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    if "reference_xgb_loss" in predictions and "candidate_xgb_loss" in predictions:
        x = predictions["reference_xgb_loss"]
        y = predictions["candidate_xgb_loss"]
        axes[1].hexbin(x, y, gridsize=45, mincnt=1, cmap="Blues")
        low = float(min(x.min(), y.min()))
        high = float(max(x.max(), y.max()))
        axes[1].plot([low, high], [low, high], linestyle="--", color="black")
        axes[1].set_xlabel("Reference XGB loss")
        axes[1].set_ylabel("Candidate XGB loss")
        axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_smoothing_models(
    action_summary: pd.DataFrame,
    matrices: Mapping[str, np.ndarray],
    output_path: Path,
) -> None:
    smooth = action_summary[
        action_summary["cohort"] != "deterministic_common_sample"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for name, group in smooth.groupby("model", sort=False):
        axes[0].plot(
            group["u"],
            group["mean_acceptance"],
            marker="o",
            markersize=3,
            label=name,
        )
    axes[0].set_xlabel("Counterfactual U")
    axes[0].set_ylabel("Mean predicted acceptance")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    delta = np.asarray(
        matrices["candidate_smooth_minus_embedded_raw"], dtype=float
    ).reshape(-1)
    axes[1].hist(delta, bins=45, color="#5b8db8", alpha=0.8)
    axes[1].axvline(0.0, color="black", linestyle="--")
    axes[1].set_xlabel("Candidate smoothed − embedded raw acceptance")
    axes[1].set_ylabel("Policy/action cells")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], limit: int = 20) -> str:
    selected = frame.loc[:, list(columns)].head(limit).copy()
    selected = selected.fillna("")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| "
        + " | ".join(str(value).replace("|", "\\|") for value in row)
        + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def build_markdown_report(
    *,
    dataset_summary: pd.DataFrame,
    inventory: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    smoothing_coverage: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> str:
    """Build the human-readable report from generated tables."""
    hash_match = bool(
        dataset_summary.loc[
            (dataset_summary["section"] == "file")
            & (dataset_summary["metric"] == "sha256"),
            "match",
        ].iloc[0]
    )
    eligible_row = dataset_summary[
        (dataset_summary["section"] == "eligibility")
        & (dataset_summary["metric"] == "eligible_rows")
    ].iloc[0]
    candidate_loss = metrics[metrics["artifact"] == "candidate_xgb_loss"].iloc[0]
    reference_loss = metrics[metrics["artifact"] == "reference_xgb_loss"].iloc[0]
    coverage = smoothing_coverage[
        (smoothing_coverage["artifact"] == "candidate_xgb_smoothed")
        & (smoothing_coverage["metric"].isin(
            [
                "covered_policy_ids",
                "coverage_fraction",
                "current_cohort_id_intersection",
                "smooth_vs_embedded_raw_mae",
                "embedded_raw_same_booster_as_saved_candidate",
                "legacy_numpy_scalar_conversion_breaks",
            ]
        ))
    ]
    report = f"""# Real-Data Model and Dataset EDA

## Executive findings

- The candidate and canonical CSV files are {"byte-for-byte identical" if hash_match else "not byte-for-byte identical"}.
- The canonical objective schema has {json.loads(eligible_row["reference"]):,} complete eligible rows.
- Candidate XGB artifacts use selected-best-fold dictionaries; reference artifacts use CV dictionaries and the runtime selects their first folds.
- On the deterministic non-holdout sample, candidate XGB loss MAE is {candidate_loss["mae"]:.4f}, versus {reference_loss["mae"]:.4f} for the reference XGB loss model. Stored OOF metrics remain the primary performance evidence.
- Candidate smoothing is a customer-specific artifact, not a global smoother. It contains 200 fitted sigmoid curves and must not be described as smoothing the full eligible population.

## Dataset comparison

{_markdown_table(dataset_summary[dataset_summary["section"].isin(["file", "structure", "integrity", "eligibility"])], ["section", "metric", "reference", "candidate", "match"], 30)}

## Artifact inventory

{_markdown_table(inventory, ["artifact", "role", "family", "source_format", "model_class", "selected_fold_zero_based", "size_bytes"], 20)}

## Stored cross-validation metrics

These are artifact-recorded out-of-fold results and are the preferred evidence for model performance. The common-row prediction analysis below is descriptive because no independent holdout dataset was provided.

{_markdown_table(cv_metrics, [column for column in ["artifact", "fold", "roc_auc", "mae", "n_train", "n_val"] if column in cv_metrics], 40)}

## Fixed-sample prediction diagnostics

Sample size: {sample_size:,}; deterministic seed: {seed}. These rows may overlap model training data.

{_markdown_table(metrics, ["artifact", "role", "mean_prediction", "roc_auc", "log_loss", "brier_score", "mae", "rmse"], 20)}

### Pairwise agreement

{_markdown_table(pairwise, ["role", "left_artifact", "right_artifact", "mean_delta", "mae", "rmse", "correlation"], 20)}

## Smoothing coverage and behavior

{_markdown_table(coverage, ["artifact", "metric", "value", "detail"], 20)}

The current portable spline cohort and candidate sigmoid cohort contain no shared customer IDs, so their aggregate curves are not a customer-level matched comparison. The candidate wrapper also embeds a different raw XGB booster from `acceptance_model_xgb.pkl`. Its source `predict_proba` implementation converts a one-element NumPy array with `float(array)`, which fails under the current NumPy runtime; this report evaluates the stored sigmoid parameters directly.

## Integration handoff

No runtime integration was performed. A future change should:

1. Expose independent `acceptance_model_type` values `glm`, `xgb`, and `xgb_smoothed`, and `loss_model_type` values `glm` and `xgb`, while retaining existing presets as compatibility aliases.
2. Restrict smoothed runs to IDs covered by the artifact. Falling back over the full dataset would make virtually every row raw XGB rather than smoothed.
3. Convert the trusted legacy sigmoid parameters into a validated portable repo-native artifact with direct acceptance and derivative interfaces.
4. Place artifacts under the existing `src/data/models` hierarchy and keep loader/runtime code at the current model seam; do not add a separate `src/models` tree.

## Limitations

- No independent holdout dataset was supplied.
- The candidate and reference smoothing cohorts are disjoint and may differ in customer mix.
- Pickle metadata describes how artifacts were saved, but cannot establish complete training-data lineage.
- Candidate loss training sampled 10,000 rows before five-fold fitting; unless the source sampling seed is documented elsewhere, that training sample is not reproducible from the artifact alone.
"""
    return report


def write_outputs(
    *,
    output_dir: Path,
    dataset_summary: pd.DataFrame,
    inventory: pd.DataFrame,
    cv_metrics: pd.DataFrame,
    metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    action_grid: pd.DataFrame,
    smoothing_coverage: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    smoothing_matrices: Mapping[str, np.ndarray],
    sample_size: int,
    seed: int,
) -> None:
    """Write all tabular, plotted, and Markdown analysis outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "dataset_summary.csv": dataset_summary,
        "artifact_inventory.csv": inventory,
        "cv_metrics.csv": cv_metrics,
        "prediction_metrics.csv": metrics,
        "pairwise_prediction_deltas.csv": pairwise,
        "acceptance_action_grid.csv": action_grid,
        "smoothing_coverage.csv": smoothing_coverage,
    }
    for filename, frame in tables.items():
        frame.to_csv(output_dir / filename, index=False)
    _plot_acceptance_models(
        metrics, predictions, action_grid, output_dir / "acceptance_model_comparison.png"
    )
    _plot_loss_models(
        metrics, predictions, output_dir / "loss_model_comparison.png"
    )
    _plot_smoothing_models(
        action_grid,
        smoothing_matrices,
        output_dir / "smoothing_model_comparison.png",
    )
    report = build_markdown_report(
        dataset_summary=dataset_summary,
        inventory=inventory,
        cv_metrics=cv_metrics,
        metrics=metrics,
        pairwise=pairwise,
        smoothing_coverage=smoothing_coverage,
        sample_size=sample_size,
        seed=seed,
    )
    (output_dir / "eda_summary.md").write_text(report, encoding="utf-8")


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return results_root() / "real-data-model-eda" / timestamp


def _artifact_paths(
    candidate_artifact_dir: Path,
    reference_model_dir: Path,
) -> list[tuple[Path, str, str, str]]:
    return [
        (
            reference_model_dir
            / "linear"
            / "acceptance_model_linear_cv_20260527_142758.pkl",
            "reference_glm_acceptance",
            "glm",
            "acceptance",
        ),
        (
            reference_model_dir
            / "xgb"
            / "acceptance_model_xgb_cv_20260527_151725.pkl",
            "reference_xgb_acceptance",
            "xgb",
            "acceptance",
        ),
        (
            candidate_artifact_dir / "acceptance_model_xgb.pkl",
            "candidate_xgb_acceptance",
            "xgb",
            "acceptance",
        ),
        (
            reference_model_dir
            / "linear"
            / "financial_loss_model_linear_cv_20260527_142758.pkl",
            "reference_glm_loss",
            "glm",
            "loss",
        ),
        (
            reference_model_dir
            / "xgb"
            / "financial_loss_model_xgb_cv_20260527_151725.pkl",
            "reference_xgb_loss",
            "xgb",
            "loss",
        ),
        (
            candidate_artifact_dir / "financial_loss_model_xgb.pkl",
            "candidate_xgb_loss",
            "xgb",
            "loss",
        ),
    ]


def run_analysis(
    *,
    candidate_dataset: Path,
    candidate_artifact_dir: Path,
    reference_dataset: Path = DATASET_PATH,
    reference_model_dir: Path = _REFERENCE_MODEL_DIR,
    sample_size: int = 20_000,
    seed: int = 20_260_728,
    u_min: float = 0.0,
    u_max: float = 0.16,
    u_step: float = 0.01,
    output_dir: Path | None = None,
) -> Path:
    """Run the complete EDA and return the output directory."""
    if u_step <= 0.0 or u_max < u_min:
        raise ValueError("Require u_step > 0 and u_max >= u_min.")
    action_grid = np.arange(u_min, u_max + u_step * 0.5, u_step, dtype=float)
    dataset_summary, reference_frame, candidate_frame = compare_dataset_files(
        reference_dataset, candidate_dataset
    )
    artifacts = [
        load_artifact_view(path, name=name, family=family, role=role)
        for path, name, family, role in _artifact_paths(
            candidate_artifact_dir, reference_model_dir
        )
    ]
    inventory = artifact_inventory(artifacts)
    cv_metrics = artifact_cv_metrics(artifacts)
    sample, eligible_population = deterministic_common_sample(
        reference_frame,
        candidate_frame,
        artifacts,
        sample_size=sample_size,
        seed=seed,
    )
    metrics, predictions = prediction_metrics(artifacts, sample)
    pairwise = pairwise_prediction_metrics(artifacts, predictions)
    raw_action_summary, _ = raw_acceptance_action_grid(
        artifacts, sample, action_grid
    )
    candidate_acceptance = next(
        artifact
        for artifact in artifacts
        if artifact.name == "candidate_xgb_acceptance"
    )
    smoothing_coverage, smoothing_action_summary, smoothing_matrices = (
        smoothing_analysis(
            candidate_smoothing_path=(
                candidate_artifact_dir / "acceptance_smoothing_wrapper.pkl"
            ),
            candidate_dataset=candidate_frame,
            reference_dataset=reference_frame,
            current_spline_path=(
                reference_model_dir
                / "xgb_logit_spline"
                / "acceptance_xgb_logit_spline_20260706_112929.npz"
            ),
            candidate_acceptance=candidate_acceptance,
            action_grid=action_grid,
            eligible_population=eligible_population,
        )
    )
    combined_action_summary = pd.concat(
        [raw_action_summary, smoothing_action_summary], ignore_index=True
    )
    resolved_output = output_dir or _default_output_dir()
    write_outputs(
        output_dir=resolved_output,
        dataset_summary=dataset_summary,
        inventory=inventory,
        cv_metrics=cv_metrics,
        metrics=metrics,
        pairwise=pairwise,
        action_grid=combined_action_summary,
        smoothing_coverage=smoothing_coverage,
        predictions=predictions,
        smoothing_matrices=smoothing_matrices,
        sample_size=sample_size,
        seed=seed,
    )
    return resolved_output


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-dataset",
        type=Path,
        required=True,
        help="Candidate semicolon-delimited CSV to compare with the canonical dataset.",
    )
    parser.add_argument(
        "--candidate-artifact-dir",
        type=Path,
        required=True,
        help="Directory containing acceptance_model_xgb.pkl, "
        "acceptance_smoothing_wrapper.pkl, and financial_loss_model_xgb.pkl.",
    )
    parser.add_argument(
        "--reference-dataset",
        type=Path,
        default=DATASET_PATH,
        help="Reference CSV. Defaults to src/data/dataset.csv.",
    )
    parser.add_argument(
        "--reference-model-dir",
        type=Path,
        default=_REFERENCE_MODEL_DIR,
        help="Reference model root. Defaults to src/data/models.",
    )
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_728)
    parser.add_argument("--u-min", type=float, default=0.0)
    parser.add_argument("--u-max", type=float, default=0.16)
    parser.add_argument("--u-step", type=float, default=0.01)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to a timestamped results directory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> Path:
    args = _parse_args(argv)
    output_dir = run_analysis(
        candidate_dataset=args.candidate_dataset,
        candidate_artifact_dir=args.candidate_artifact_dir,
        reference_dataset=args.reference_dataset,
        reference_model_dir=args.reference_model_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        u_min=args.u_min,
        u_max=args.u_max,
        u_step=args.u_step,
        output_dir=args.output_dir,
    )
    print(f"Wrote real-data model EDA to {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
