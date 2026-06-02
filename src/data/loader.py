"""Data loading utilities for real insurance pricing datasets."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from data.feature_processor import FeatureProcessor
from data.dataset_metadata import (
    ACCEPTANCE_PROBABILITY_COL,
    ACCEPTANCE_STATE_COLS as _ACCEPTANCE_STATE_COLS,
    BASE_COLS as _BASE_COLS_TUPLE,
    DATA_DIR,
    DATASET_PATH,
    FEATURE_COLS as _FEATURE_COLS,
    FEATURE_COLS_GLM as _FEATURE_COLS_GLM,
    FEATURE_COLS_XGB as _FEATURE_COLS_XGB,
    LOSS_FEATURE_COLS as _LOSS_FEATURE_COLS,
    MODEL_ARTIFACTS,
    MODEL_FEATURE_COLS,
    OBSERVED_U_COL,
    PREMIUM_COL_INDEX,
)

# Directory containing model artifacts and CSV datasets
_DATA_DIR = DATA_DIR
_DATASET_CSV_PATH = DATASET_PATH

# Base 9 feature cols shared by both models (no premium, no U, no extra)
_BASE_COLS: list[str] = list(_BASE_COLS_TUPLE)

# GLM state features: 12 raw cols used by the policy state distribution.
FEATURE_COLS_GLM: list[str] = list(_FEATURE_COLS_GLM)

# XGB state features: 10 raw cols used by the policy state distribution.
FEATURE_COLS_XGB: list[str] = list(_FEATURE_COLS_XGB)

# Default alias — use when you don't need model-specific differences
FEATURE_COLS = list(_FEATURE_COLS)

# Columns consumed by the loss model before external preprocessing.
LOSS_FEATURE_COLS: list[str] = list(_LOSS_FEATURE_COLS)

# Columns consumed by acceptance model state part before external preprocessing.
ACCEPTANCE_STATE_COLS: list[str] = list(_ACCEPTANCE_STATE_COLS)

_PREMIUM_COL_INDEX: int = PREMIUM_COL_INDEX  # index of X_policy_premium in both FEATURE_COLS variants


@dataclass(frozen=True)
class ModelArtifactBundle:
    """A saved model together with its external feature processor."""

    model: Any
    preprocessor: FeatureProcessor | None
    u_cols: tuple[str, ...]
    x_feature_cols: tuple[str, ...]

    def model_frame(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        """Build the exact model-input frame from raw notebook-space columns."""
        missing = [
            col
            for col in (*self.u_cols, *self.x_feature_cols)
            if col not in raw_frame.columns
        ]
        if missing:
            raise ValueError(f"Missing required artifact columns: {missing}")

        if self.preprocessor is None:
            columns = [*self.u_cols, *self.x_feature_cols]
            model_frame = raw_frame.loc[:, columns].copy()
        else:
            transformed = self.preprocessor.transform(raw_frame.loc[:, list(self.x_feature_cols)].copy())
            if self.u_cols:
                u_frame = raw_frame.loc[:, list(self.u_cols)].reset_index(drop=True)
                model_frame = pd.concat([u_frame, transformed.reset_index(drop=True)], axis=1)
            else:
                model_frame = transformed

        if hasattr(self.model, "feature_names_in_"):
            model_columns = list(self.model.feature_names_in_)
            model_frame = model_frame.reindex(columns=model_columns)
        return model_frame

    def policy_feature_dim(self) -> int:
        """Return the state dimension seen by a policy over processed features."""
        if self.preprocessor is None:
            return len(self.x_feature_cols)
        return len(getattr(self.preprocessor, "output_feature_names_", ()))


class _ArtifactUnpickler(pickle.Unpickler):
    """Map notebook-local classes onto importable repo modules."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "FeatureProcessor":
            return FeatureProcessor
        return super().find_class(module, name)


_ARTIFACT_PATHS: dict[str, dict[str, Path]] = {
    "glm": {
        "acceptance": MODEL_ARTIFACTS["glm"]["acceptance"]["path"],
        "loss": MODEL_ARTIFACTS["glm"]["loss"]["path"],
    },
    "xgb": {
        "acceptance": MODEL_ARTIFACTS["xgb"]["acceptance"]["path"],
        "loss": MODEL_ARTIFACTS["xgb"]["loss"]["path"],
    },
}

_ACCEPTANCE_CSV_PATHS: dict[str, Path] = {
    "glm": _DATASET_CSV_PATH,
    "xgb": _DATASET_CSV_PATH,
}


def _acceptance_csv_path(model_type: Literal["glm", "xgb"]) -> Path:
    if model_type not in _ACCEPTANCE_CSV_PATHS:
        raise ValueError(f"model_type must be 'glm' or 'xgb', got '{model_type}'.")
    return _ACCEPTANCE_CSV_PATHS[model_type]


def dataset_csv_path() -> Path:
    """Return the canonical real-data source CSV path."""
    return _DATASET_CSV_PATH


def _csv_row_count(csv_path: Path) -> int:
    with open(csv_path, encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _validate_row_indices(row_indices: np.ndarray | Sequence[int], total_rows: int) -> np.ndarray:
    indices = np.asarray(row_indices)
    if indices.ndim != 1:
        raise ValueError("row_indices must be a 1D array of CSV row positions.")
    if indices.size == 0:
        raise ValueError("row_indices must contain at least one row position.")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("row_indices must contain integer row positions.")
    indices = indices.astype(int, copy=False)
    if np.any(indices < 0) or np.any(indices >= total_rows):
        raise ValueError(
            f"row_indices must be in [0, {total_rows - 1}] for the selected CSV."
        )
    if np.unique(indices).size != indices.size:
        raise ValueError("row_indices must not contain duplicates.")
    return indices


def sample_csv_row_indices(
    model_type: Literal["glm", "xgb"],
    n_rows: int,
    seed: int | None = None,
) -> np.ndarray:
    """Sample acceptance CSV row positions without replacement."""
    n_rows = int(n_rows)
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    csv_path = _acceptance_csv_path(model_type)
    total_rows = _csv_row_count(csv_path)
    if n_rows > total_rows:
        raise ValueError(
            f"Cannot sample {n_rows} rows from {csv_path.name}; CSV has {total_rows} rows."
        )
    rng = np.random.default_rng(seed)
    return rng.choice(total_rows, size=n_rows, replace=False).astype(int)


def _select_csv_rows(df: pd.DataFrame, row_indices: np.ndarray) -> pd.DataFrame:
    return df.iloc[row_indices].reset_index(drop=True)


def _encode_non_numeric_state_columns(df: pd.DataFrame, csv_path: Path, feature_cols: list[str]) -> pd.DataFrame:
    """Apply notebook-style label encoding to any string state columns."""
    df_encoded = df.copy()
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df_encoded[col]):
            continue
        full_col = pd.read_csv(csv_path, sep=";", usecols=[col])[col].fillna("__MISSING__").astype(str)
        classes = sorted(full_col.unique().tolist())
        mapping = {label: idx for idx, label in enumerate(classes)}
        df_encoded[col] = (
            df_encoded[col]
            .fillna("__MISSING__")
            .astype(str)
            .map(mapping)
            .astype(float)
        )
    return df_encoded


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return _ArtifactUnpickler(f).load()


def _normalize_artifact(raw_artifact: Any) -> ModelArtifactBundle:
    if isinstance(raw_artifact, ModelArtifactBundle):
        return raw_artifact
    if isinstance(raw_artifact, dict) and "model" in raw_artifact:
        return ModelArtifactBundle(
            model=raw_artifact["model"],
            preprocessor=raw_artifact.get("preprocessor"),
            u_cols=tuple(raw_artifact.get("u_cols", ())),
            x_feature_cols=tuple(raw_artifact.get("x_feature_cols", ())),
        )
    model = raw_artifact
    feature_names = tuple(getattr(model, "feature_names_in_", ()))
    return ModelArtifactBundle(
        model=model,
        preprocessor=None,
        u_cols=(),
        x_feature_cols=feature_names,
    )


def unwrap_model_artifact(artifact: Any) -> Any:
    """Return the estimator stored inside an artifact bundle."""
    if isinstance(artifact, ModelArtifactBundle):
        return artifact.model
    return artifact


def load_model_artifacts(model_type: Literal["glm", "xgb"]) -> tuple[ModelArtifactBundle, ModelArtifactBundle]:
    """Load and return (acceptance_artifact, loss_artifact) bundles."""
    if model_type not in _ARTIFACT_PATHS:
        raise ValueError(f"model_type must be 'glm' or 'xgb', got '{model_type}'.")
    paths = _ARTIFACT_PATHS[model_type]
    acceptance_model = _normalize_artifact(_load_pickle(paths["acceptance"]))
    loss_model = _normalize_artifact(_load_pickle(paths["loss"]))
    return acceptance_model, loss_model


def load_x_array(
    model_type: Literal["glm", "xgb"],
    n_rows: int = 5000,
    *,
    row_indices: np.ndarray | Sequence[int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Load a random sample of raw state features from the acceptance CSV."""
    csv_path = _acceptance_csv_path(model_type)
    feature_cols = list(MODEL_FEATURE_COLS[model_type])
    if row_indices is None:
        row_indices = sample_csv_row_indices(model_type, n_rows=n_rows, seed=seed)
    else:
        row_indices = _validate_row_indices(row_indices, _csv_row_count(csv_path))
    df = pd.read_csv(csv_path, sep=";")
    df = _select_csv_rows(df, np.asarray(row_indices, dtype=int))
    df = _encode_non_numeric_state_columns(df, csv_path, feature_cols)
    return df[feature_cols].to_numpy(dtype=float)


def load_observed_u_array(
    model_type: Literal["glm", "xgb"],
    n_rows: int | None = 5000,
    *,
    row_indices: np.ndarray | Sequence[int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Load observed pricing multipliers from sampled acceptance CSV rows."""
    csv_path = _acceptance_csv_path(model_type)
    if row_indices is None:
        if n_rows is None:
            n_rows = _csv_row_count(csv_path)
        row_indices = sample_csv_row_indices(model_type, n_rows=n_rows, seed=seed)
    else:
        row_indices = _validate_row_indices(row_indices, _csv_row_count(csv_path))
    df = pd.read_csv(csv_path, sep=";", usecols=[OBSERVED_U_COL])
    df = _select_csv_rows(df, np.asarray(row_indices, dtype=int))
    return df[OBSERVED_U_COL].to_numpy(dtype=float)


def _load_observed_u_array(
    model_type: Literal["glm", "xgb"],
    n_rows: int | None = 5000,
    *,
    row_indices: np.ndarray | Sequence[int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Backward-compatible alias for loading observed pricing multipliers."""
    return load_observed_u_array(model_type, n_rows=n_rows, row_indices=row_indices, seed=seed)


def load_mean_observed_acceptance(model_type: Literal["glm", "xgb"]) -> float:
    """Load mean acceptance probability from the exported acceptance CSV."""
    if model_type not in _ACCEPTANCE_CSV_PATHS:
        raise ValueError(f"model_type must be 'glm' or 'xgb', got '{model_type}'.")
    df = pd.read_csv(_ACCEPTANCE_CSV_PATHS[model_type], sep=";", usecols=[ACCEPTANCE_PROBABILITY_COL])
    return float(df[ACCEPTANCE_PROBABILITY_COL].mean())


def extract_glm_u_coef(glm_pipeline: Any) -> float:
    """Extract effective d_logit/dU = w_U / std_U from a fitted GLM Pipeline."""
    glm_pipeline = unwrap_model_artifact(glm_pipeline)
    preprocessor = glm_pipeline.named_steps["preprocessor"]
    classifier = glm_pipeline.named_steps["classifier"]

    transformed_names = list(preprocessor.get_feature_names_out())
    u_indices = [i for i, name in enumerate(transformed_names) if name.endswith("__U") or name == "U"]
    if len(u_indices) != 1:
        raise ValueError(
            f"Expected exactly 1 'U' column in preprocessor output, found at indices: {u_indices}. "
            f"Available names: {transformed_names}"
        )
    i_u_out = u_indices[0]
    w_u = float(classifier.coef_[0, i_u_out])

    for _, transformer, cols in preprocessor.transformers_:
        if not hasattr(transformer, "named_steps"):
            continue
        if "scaler" not in transformer.named_steps:
            continue
        col_list = list(cols)
        if "U" not in col_list:
            continue
        i_u_numeric = col_list.index("U")
        std_u = float(transformer.named_steps["scaler"].scale_[i_u_numeric])
        return w_u / std_u

    raise ValueError("Could not find a StandardScaler containing 'U' in the GLM pipeline preprocessor.")


def extract_glm_churn_coefficients(glm_pipeline: Any) -> dict[str, Any]:
    """Extract processed-space churn coefficients from a fitted GLM artifact."""
    glm_pipeline = unwrap_model_artifact(glm_pipeline)
    preprocessor = glm_pipeline.named_steps["preprocessor"]
    classifier = glm_pipeline.named_steps["classifier"]
    transformed_names = list(preprocessor.get_feature_names_out())

    numeric_transformer = None
    numeric_cols: list[str] | None = None
    for _, transformer, cols in preprocessor.transformers_:
        if not hasattr(transformer, "named_steps"):
            continue
        if "scaler" not in transformer.named_steps:
            continue
        numeric_transformer = transformer
        numeric_cols = list(cols)
        break

    if numeric_transformer is None or numeric_cols is None:
        raise ValueError("Could not find a scaled numeric transformer in the GLM pipeline preprocessor.")

    if len(transformed_names) != len(numeric_cols):
        raise ValueError(
            "GLM coefficient extraction only supports one-to-one preprocessing. "
            f"Found {len(transformed_names)} transformed features for {len(numeric_cols)} raw numeric columns."
        )

    for out_name, raw_name in zip(transformed_names, numeric_cols):
        if not (out_name.endswith(f"__{raw_name}") or out_name == raw_name):
            raise ValueError(
                "GLM coefficient extraction requires transformed feature names to align "
                f"with raw numeric columns. Got transformed name '{out_name}' for raw column '{raw_name}'."
            )

    scaler = numeric_transformer.named_steps["scaler"]
    coef_scaled = np.asarray(classifier.coef_[0], dtype=float)
    intercept_scaled = float(classifier.intercept_[0])
    mean = np.asarray(scaler.mean_, dtype=float)
    scale = np.asarray(scaler.scale_, dtype=float)
    coef_raw = coef_scaled / scale
    intercept_raw = intercept_scaled - float(np.dot(coef_scaled, mean / scale))

    if "U" not in numeric_cols:
        raise ValueError(f"Expected raw numeric columns to include 'U'. Got: {numeric_cols}")

    u_index = numeric_cols.index("U")
    x_feature_names = [name for name in numeric_cols if name != "U"]
    x_coef = np.delete(coef_raw, u_index)
    u_coef = float(coef_raw[u_index])

    return {
        "formula": "logit(p_churn(z, u)) = intercept + beta_z^T z + beta_u * u",
        "x_feature_names": x_feature_names,
        "x_coef": [float(val) for val in x_coef.tolist()],
        "u_coef": u_coef,
        "intercept": intercept_raw,
    }


def extract_linear_loss_coefficients(linear_model: Any) -> dict[str, Any]:
    """Extract coefficients from a fitted linear-regression loss artifact."""
    linear_model = unwrap_model_artifact(linear_model)
    if not hasattr(linear_model, "coef_") or not hasattr(linear_model, "intercept_"):
        raise ValueError("Expected a fitted linear model with coef_ and intercept_.")
    if not hasattr(linear_model, "feature_names_in_"):
        raise ValueError("Expected the fitted linear model to expose feature_names_in_.")

    coef = np.asarray(linear_model.coef_, dtype=float)
    feature_names = [str(name) for name in linear_model.feature_names_in_.tolist()]
    if coef.shape != (len(feature_names),):
        raise ValueError(
            "Expected one coefficient per model input feature. "
            f"Got coefficient shape {coef.shape} for {len(feature_names)} features."
        )

    return {
        "formula": "loss_hat(z) = intercept + gamma_z^T z",
        "x_feature_names": feature_names,
        "x_coef": [float(val) for val in coef.tolist()],
        "intercept": float(linear_model.intercept_),
    }


def extract_model_based_coefficients(acceptance_model: Any, loss_model: Any) -> dict[str, dict[str, Any]] | None:
    """Extract printable coefficient summaries for supported model-based artifacts."""
    try:
        churn = extract_glm_churn_coefficients(acceptance_model)
        loss = extract_linear_loss_coefficients(loss_model)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return {"churn": churn, "loss": loss}


__all__ = [
    "FEATURE_COLS",
    "FEATURE_COLS_GLM",
    "FEATURE_COLS_XGB",
    "LOSS_FEATURE_COLS",
    "ACCEPTANCE_STATE_COLS",
    "ModelArtifactBundle",
    "dataset_csv_path",
    "sample_csv_row_indices",
    "load_model_artifacts",
    "load_x_array",
    "load_observed_u_array",
    "load_mean_observed_acceptance",
    "unwrap_model_artifact",
    "extract_glm_u_coef",
    "extract_glm_churn_coefficients",
    "extract_linear_loss_coefficients",
    "extract_model_based_coefficients",
]
