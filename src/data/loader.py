"""Data loading utilities for the canonical Generali real-data dataset."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import pickle
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from data.feature_processor import FeatureProcessor
from data.monotone_spline_xgb import load_monotone_spline_xgb_acceptance
from data.dataset_metadata import (
    ACCEPTANCE_MODEL_ARTIFACTS,
    ACCEPTANCE_STATE_COLS as _ACCEPTANCE_STATE_COLS,
    AcceptanceModelType,
    DATA_DIR,
    DATASET_PATH,
    FEATURE_COLS as _FEATURE_COLS,
    FEATURE_COLS_GLM as _FEATURE_COLS_GLM,
    FEATURE_COLS_XGB as _FEATURE_COLS_XGB,
    LOOKAHEAD_X_COLS,
    LOSS_TARGET_COL,
    LOSS_FEATURE_COLS as _LOSS_FEATURE_COLS,
    LOSS_MODEL_ARTIFACTS,
    LossModelType,
    OBJECTIVE_EXCLUDED_COLS,
    OBSERVED_CHURN_COL,
    OBSERVED_U_COL,
    PREMIUM_COL,
    PREMIUM_COL_INDEX,
    REQUIRED_DATASET_COLUMNS,
    UNUSED_X_COLS,
    USED_X_COLS,
)

ModelType = Literal["linear", "xgb", "monotone_spline_xgb"]
AcceptanceSelection = ModelType | AcceptanceModelType
ProbabilityTarget = Literal["acceptance", "churn", "none"]

# Directory containing model artifacts and CSV datasets
_DATA_DIR = DATA_DIR
_DATASET_CSV_PATH = DATASET_PATH

# 052726 raw model covariates. These exclude lookahead columns such as
# X_upcoming_premium and exclude observed targets/actions.
FEATURE_COLS_GLM: list[str] = list(_FEATURE_COLS_GLM)
FEATURE_COLS_XGB: list[str] = list(_FEATURE_COLS_XGB)
FEATURE_COLS = list(_FEATURE_COLS)
LOSS_FEATURE_COLS: list[str] = list(_LOSS_FEATURE_COLS)
ACCEPTANCE_STATE_COLS: list[str] = list(_ACCEPTANCE_STATE_COLS)
_PREMIUM_COL_INDEX: int = PREMIUM_COL_INDEX


@dataclass(frozen=True)
class ModelArtifactBundle:
    """A fitted first-fold model together with its fitted feature processor."""

    model: Any
    preprocessor: FeatureProcessor | None
    u_cols: tuple[str, ...]
    x_feature_cols: tuple[str, ...]
    probability_target: ProbabilityTarget = "none"
    source_format: str = "single_model"
    model_type: str | None = None
    artifact_id: str | None = None
    role: str | None = None
    artifact_path: str | None = None

    def model_frame(self, raw_frame: pd.DataFrame) -> pd.DataFrame:
        """Build the exact model-input frame from raw source-space columns."""
        missing = [
            col
            for col in (*self.x_feature_cols, *self.u_cols)
            if col not in raw_frame.columns
        ]
        if missing:
            raise ValueError(f"Missing required artifact columns: {missing}")

        if self.preprocessor is None:
            columns = [*self.x_feature_cols, *self.u_cols]
            model_frame = raw_frame.loc[:, columns].copy()
        else:
            transformed = self.preprocessor.transform(
                raw_frame.loc[:, list(self.x_feature_cols)].copy()
            )
            if self.u_cols:
                u_frame = raw_frame.loc[:, list(self.u_cols)].reset_index(drop=True)
                model_frame = pd.concat(
                    [transformed.reset_index(drop=True), u_frame],
                    axis=1,
                )
            else:
                model_frame = transformed

        if hasattr(self.model, "feature_names_in_"):
            model_columns = list(self.model.feature_names_in_)
            model_frame = model_frame.reindex(columns=model_columns)
        return model_frame

    def policy_feature_dim(self) -> int:
        """Return the processed state dimension seen by an artifact-preprocessed policy."""
        if self.preprocessor is None:
            return len(self.x_feature_cols)
        return len(getattr(self.preprocessor, "output_feature_names_", ()))


class _ArtifactUnpickler(pickle.Unpickler):
    """Map notebook-local classes onto importable repo modules."""

    def find_class(self, module: str, name: str) -> Any:
        if name == "FeatureProcessor" and module in {"__main__", "preprocessing"}:
            return FeatureProcessor
        return super().find_class(module, name)


MODEL_PAIRS: dict[ModelType, tuple[AcceptanceModelType, LossModelType]] = {
    "linear": ("linear", "linear"),
    "xgb": ("xgb", "xgb"),
    "monotone_spline_xgb": ("monotone_spline_xgb", "xgb"),
}

_CURVE_ACCEPTANCE_TYPES = {"monotone_spline_xgb"}


def _acceptance_csv_path(model_type: AcceptanceSelection) -> Path:
    _validate_acceptance_selection(model_type)
    return _DATASET_CSV_PATH


def dataset_csv_path() -> Path:
    """Return the canonical 052726 real-data source CSV path."""
    return _DATASET_CSV_PATH


def dataset_column_roles() -> dict[str, tuple[str, ...]]:
    """Return column groups used by and excluded from objective construction."""
    return {
        "used_x_cols": tuple(USED_X_COLS),
        "acceptance_x_cols": tuple(ACCEPTANCE_STATE_COLS),
        "loss_x_cols": tuple(LOSS_FEATURE_COLS),
        "unused_x_cols": tuple(UNUSED_X_COLS),
        "lookahead_x_cols": tuple(LOOKAHEAD_X_COLS),
        "objective_excluded_cols": tuple(OBJECTIVE_EXCLUDED_COLS),
        "action_cols": (OBSERVED_U_COL,),
        "target_cols": (LOSS_TARGET_COL, OBSERVED_CHURN_COL),
    }


def _csv_row_count(csv_path: Path) -> int:
    with open(csv_path, encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def _validate_model_type(model_type: str) -> ModelType:
    if model_type == "glm":
        return "linear"
    if model_type not in MODEL_PAIRS:
        raise ValueError(
            "model_type must be 'linear', 'xgb', or 'monotone_spline_xgb', "
            f"got '{model_type}'."
        )
    return model_type  # type: ignore[return-value]


def _validate_acceptance_model_type(model_type: str) -> AcceptanceModelType:
    if model_type not in ACCEPTANCE_MODEL_ARTIFACTS:
        available = ", ".join(sorted(ACCEPTANCE_MODEL_ARTIFACTS))
        raise ValueError(f"Unknown acceptance_model_type '{model_type}'. Available: {available}.")
    return model_type  # type: ignore[return-value]


def _validate_loss_model_type(model_type: str) -> LossModelType:
    if model_type not in LOSS_MODEL_ARTIFACTS:
        available = ", ".join(sorted(LOSS_MODEL_ARTIFACTS))
        raise ValueError(f"Unknown loss_model_type '{model_type}'. Available: {available}.")
    return model_type  # type: ignore[return-value]


def _validate_acceptance_selection(model_type: str) -> AcceptanceSelection:
    if model_type == "glm":
        return "linear"
    if model_type in MODEL_PAIRS:
        return model_type  # type: ignore[return-value]
    return _validate_acceptance_model_type(model_type)


def resolve_model_artifact_ids(
    *,
    model_type: ModelType | None = None,
    acceptance_model_type: AcceptanceModelType | None = None,
    loss_model_type: LossModelType | None = None,
) -> tuple[AcceptanceModelType, LossModelType]:
    """Resolve a named family or explicit independent artifact IDs."""
    if model_type is not None:
        pair = MODEL_PAIRS[_validate_model_type(model_type)]
        if acceptance_model_type is not None and acceptance_model_type != pair[0]:
            raise ValueError("model_type conflicts with acceptance_model_type.")
        if loss_model_type is not None and loss_model_type != pair[1]:
            raise ValueError("model_type conflicts with loss_model_type.")
        return pair
    if acceptance_model_type is None or loss_model_type is None:
        raise ValueError(
            "Specify model_type or both acceptance_model_type and loss_model_type."
        )
    return (
        _validate_acceptance_model_type(acceptance_model_type),
        _validate_loss_model_type(loss_model_type),
    )


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


@lru_cache(maxsize=4)
def _eligible_row_indices(csv_path: Path) -> np.ndarray:
    """Return CSV row positions with complete objective X and observed diagnostics."""
    df = pd.read_csv(csv_path, sep=";", usecols=list(REQUIRED_DATASET_COLUMNS))
    complete = df.notna().all(axis=1).to_numpy(dtype=bool)
    return np.flatnonzero(complete).astype(int)


def _select_csv_rows(df: pd.DataFrame, row_indices: np.ndarray) -> pd.DataFrame:
    return df.iloc[row_indices].reset_index(drop=True)


def _validate_no_missing_required(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required dataset columns: {missing_columns}")
    missing_values = df.loc[:, list(columns)].isna().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        raise ValueError(
            "Selected rows contain missing required values: "
            f"{missing_values.astype(int).to_dict()}"
        )


def sample_csv_row_indices(
    model_type: AcceptanceSelection,
    n_rows: int,
    seed: int | None = None,
) -> np.ndarray:
    """Sample complete 052726 CSV row positions without replacement."""
    model_type = _validate_acceptance_selection(model_type)
    n_rows = int(n_rows)
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")
    csv_path = _acceptance_csv_path(model_type)
    eligible = eligible_csv_row_indices(model_type)
    if n_rows > eligible.size:
        raise ValueError(
            f"Cannot sample {n_rows} eligible rows from {csv_path.name}; "
            f"CSV has {eligible.size} complete eligible rows."
        )
    rng = np.random.default_rng(seed)
    return rng.choice(eligible, size=n_rows, replace=False).astype(int)


def eligible_csv_row_indices(model_type: AcceptanceSelection) -> np.ndarray:
    """Return complete canonical rows for every runtime model family."""
    model_type = _validate_acceptance_selection(model_type)
    return _eligible_row_indices(_acceptance_csv_path(model_type)).copy()


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return _ArtifactUnpickler(f).load()


def _artifact_probability_target(raw_artifact: Any, fallback: ProbabilityTarget) -> ProbabilityTarget:
    if isinstance(raw_artifact, dict):
        target = raw_artifact.get("target")
        if target == "acceptance":
            return "acceptance"
    return fallback


def _first_fold_artifact(
    raw_artifact: dict[str, Any],
    *,
    probability_target: ProbabilityTarget,
) -> ModelArtifactBundle:
    models = raw_artifact.get("trained_models")
    if not isinstance(models, Sequence) or len(models) == 0:
        raise ValueError("CV artifact must contain a non-empty trained_models sequence.")
    model = models[0]

    preprocessor = raw_artifact.get("preprocessor")
    x_feature_cols = raw_artifact.get("x_feature_cols")
    u_cols: Sequence[str] = ()

    trained_preprocessors = raw_artifact.get("trained_preprocessors")
    if preprocessor is None and isinstance(trained_preprocessors, Sequence) and trained_preprocessors:
        first_preprocessor = trained_preprocessors[0]
        if isinstance(first_preprocessor, dict):
            preprocessor = first_preprocessor.get("preprocessor")
            x_feature_cols = (
                first_preprocessor.get("x_feature_cols")
                or first_preprocessor.get("feature_cols")
                or x_feature_cols
            )
            u_cols = tuple(first_preprocessor.get("u_cols", ()))

    model_features = tuple(raw_artifact.get("model_features", ()))
    if not u_cols and "U" in model_features:
        u_cols = ("U",)
    if x_feature_cols is None:
        x_feature_cols = tuple(col for col in model_features if col not in set(u_cols))

    if x_feature_cols is None:
        raise ValueError("Could not resolve x_feature_cols from CV artifact.")

    return ModelArtifactBundle(
        model=model,
        preprocessor=preprocessor,
        u_cols=tuple(u_cols),
        x_feature_cols=tuple(x_feature_cols),
        probability_target=probability_target,
        source_format="cv_first_fold",
    )


def _normalize_artifact(
    raw_artifact: Any,
    *,
    probability_target: ProbabilityTarget = "none",
) -> ModelArtifactBundle:
    probability_target = _artifact_probability_target(raw_artifact, probability_target)
    if isinstance(raw_artifact, ModelArtifactBundle):
        return raw_artifact
    if isinstance(raw_artifact, dict) and "trained_models" in raw_artifact:
        return _first_fold_artifact(raw_artifact, probability_target=probability_target)
    if isinstance(raw_artifact, dict) and "model" in raw_artifact:
        preprocessor_payload = raw_artifact.get("preprocessor")
        preprocessor = preprocessor_payload
        u_cols: Sequence[str] = raw_artifact.get("u_cols", ())
        x_feature_cols: Sequence[str] = raw_artifact.get("x_feature_cols", ())
        if isinstance(preprocessor_payload, dict):
            preprocessor = preprocessor_payload.get("preprocessor")
            u_cols = preprocessor_payload.get("u_cols", u_cols)
            x_feature_cols = preprocessor_payload.get(
                "x_feature_cols",
                preprocessor_payload.get("feature_cols", x_feature_cols),
            )
        model_features = tuple(raw_artifact.get("model_features", ()))
        if not u_cols and "U" in model_features:
            u_cols = ("U",)
        if not x_feature_cols:
            x_feature_cols = tuple(
                column for column in model_features if column not in set(u_cols)
            )
        if not x_feature_cols:
            raise ValueError("Could not resolve x_feature_cols from single-model artifact.")
        return ModelArtifactBundle(
            model=raw_artifact["model"],
            preprocessor=preprocessor,
            u_cols=tuple(u_cols),
            x_feature_cols=tuple(x_feature_cols),
            probability_target=probability_target,
            source_format=str(
                raw_artifact.get(
                    "source_format",
                    "selected_best_fold" if "best_fold" in raw_artifact else "single_model",
                )
            ),
        )
    model = raw_artifact
    feature_names = tuple(getattr(model, "feature_names_in_", ()))
    return ModelArtifactBundle(
        model=model,
        preprocessor=None,
        u_cols=(),
        x_feature_cols=feature_names,
        probability_target=probability_target,
    )


def unwrap_model_artifact(artifact: Any) -> Any:
    """Return the estimator stored inside an artifact bundle."""
    if isinstance(artifact, ModelArtifactBundle):
        return artifact.model
    return artifact


def load_acceptance_artifact(model_type: AcceptanceModelType) -> Any:
    """Load one canonical acceptance artifact."""
    model_type = _validate_acceptance_model_type(model_type)
    spec = ACCEPTANCE_MODEL_ARTIFACTS[model_type]
    path = spec["path"]
    if model_type == "monotone_spline_xgb":
        xgb_raw = _normalize_artifact(
            _load_pickle(ACCEPTANCE_MODEL_ARTIFACTS["xgb"]["path"]),
            probability_target="acceptance",
        )
        xgb_raw = replace(
            xgb_raw,
            model_type="xgb",
            artifact_id="xgb",
            role="acceptance",
            artifact_path=str(ACCEPTANCE_MODEL_ARTIFACTS["xgb"]["path"]),
        )
        return load_monotone_spline_xgb_acceptance(path, xgb_raw)
    acceptance_model = _normalize_artifact(
        _load_pickle(path),
        probability_target=spec.get("probability_target", "acceptance"),
    )
    return replace(
        acceptance_model,
        model_type=model_type,
        artifact_id=model_type,
        role="acceptance",
        artifact_path=str(path),
    )


def load_loss_artifact(model_type: LossModelType) -> ModelArtifactBundle:
    """Load one canonical financial-loss artifact."""
    model_type = _validate_loss_model_type(model_type)
    spec = LOSS_MODEL_ARTIFACTS[model_type]
    path = spec["path"]
    loss_model = _normalize_artifact(
        _load_pickle(path),
        probability_target=spec.get("probability_target", "none"),
    )
    return replace(
        loss_model,
        model_type=model_type,
        artifact_id=model_type,
        role="loss",
        artifact_path=str(path),
    )


def load_model_artifact_pair(
    acceptance_model_type: AcceptanceModelType,
    loss_model_type: LossModelType,
) -> tuple[Any, ModelArtifactBundle]:
    """Load independently selected acceptance and financial-loss artifacts."""
    return (
        load_acceptance_artifact(acceptance_model_type),
        load_loss_artifact(loss_model_type),
    )


def load_model_artifacts(model_type: ModelType) -> tuple[Any, ModelArtifactBundle]:
    """Load the acceptance/loss pair for a canonical model family."""
    acceptance_model_type, loss_model_type = resolve_model_artifact_ids(
        model_type=model_type
    )
    return load_model_artifact_pair(acceptance_model_type, loss_model_type)


def load_x_frame(
    model_type: AcceptanceSelection,
    n_rows: int = 5000,
    *,
    row_indices: np.ndarray | Sequence[int] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Load raw 052726 X covariates for optimization, preserving categoricals."""
    model_type = _validate_acceptance_selection(model_type)
    csv_path = _acceptance_csv_path(model_type)
    feature_cols = list(_FEATURE_COLS)
    source_cols = ["id", *feature_cols] if model_type in _CURVE_ACCEPTANCE_TYPES else feature_cols
    if row_indices is None:
        row_indices = sample_csv_row_indices(model_type, n_rows=n_rows, seed=seed)
    else:
        row_indices = _validate_row_indices(row_indices, _csv_row_count(csv_path))
    dtype = {"id": "string"} if model_type in _CURVE_ACCEPTANCE_TYPES else None
    df = pd.read_csv(csv_path, sep=";", usecols=source_cols, dtype=dtype)
    df = _select_csv_rows(df, np.asarray(row_indices, dtype=int))
    _validate_no_missing_required(df, source_cols)
    return df.loc[:, source_cols].copy()


def load_x_array(
    model_type: AcceptanceSelection,
    n_rows: int = 5000,
    *,
    row_indices: np.ndarray | Sequence[int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Load raw X covariates as an object array; prefer load_x_frame for real data."""
    return load_x_frame(
        model_type,
        n_rows=n_rows,
        row_indices=row_indices,
        seed=seed,
    ).to_numpy(dtype=object)


def load_observed_u_array(
    model_type: AcceptanceSelection,
    n_rows: int | None = 5000,
    *,
    row_indices: np.ndarray | Sequence[int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Load historical pricing actions from sampled complete 052726 rows."""
    model_type = _validate_acceptance_selection(model_type)
    csv_path = _acceptance_csv_path(model_type)
    if row_indices is None:
        if n_rows is None:
            row_indices = _eligible_row_indices(csv_path)
        else:
            row_indices = sample_csv_row_indices(model_type, n_rows=n_rows, seed=seed)
    else:
        row_indices = _validate_row_indices(row_indices, _csv_row_count(csv_path))
    df = pd.read_csv(csv_path, sep=";", usecols=[OBSERVED_U_COL])
    df = _select_csv_rows(df, np.asarray(row_indices, dtype=int))
    _validate_no_missing_required(df, [OBSERVED_U_COL])
    return df[OBSERVED_U_COL].to_numpy(dtype=float)


def load_observed_loss_array(
    model_type: AcceptanceSelection,
    n_rows: int | None = 5000,
    *,
    row_indices: np.ndarray | Sequence[int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Load observed historical financial loss from sampled complete 052726 rows."""
    model_type = _validate_acceptance_selection(model_type)
    csv_path = _acceptance_csv_path(model_type)
    if row_indices is None:
        if n_rows is None:
            row_indices = _eligible_row_indices(csv_path)
        else:
            row_indices = sample_csv_row_indices(model_type, n_rows=n_rows, seed=seed)
    else:
        row_indices = _validate_row_indices(row_indices, _csv_row_count(csv_path))
    df = pd.read_csv(csv_path, sep=";", usecols=[LOSS_TARGET_COL])
    df = _select_csv_rows(df, np.asarray(row_indices, dtype=int))
    _validate_no_missing_required(df, [LOSS_TARGET_COL])
    return df[LOSS_TARGET_COL].to_numpy(dtype=float)


def _load_observed_u_array(
    model_type: AcceptanceSelection,
    n_rows: int | None = 5000,
    *,
    row_indices: np.ndarray | Sequence[int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Backward-compatible alias for loading observed pricing multipliers."""
    return load_observed_u_array(model_type, n_rows=n_rows, row_indices=row_indices, seed=seed)


def load_mean_observed_acceptance(model_type: AcceptanceSelection) -> float:
    """Load mean observed acceptance (1 - is_churn) on complete eligible rows."""
    model_type = _validate_acceptance_selection(model_type)
    csv_path = _acceptance_csv_path(model_type)
    row_indices = eligible_csv_row_indices(model_type)
    df = pd.read_csv(csv_path, sep=";", usecols=[OBSERVED_CHURN_COL])
    df = _select_csv_rows(df, row_indices)
    return float(1.0 - df[OBSERVED_CHURN_COL].to_numpy(dtype=float).mean())


def extract_glm_u_coef(glm_pipeline: Any) -> float:
    """Extract d logit(P(accept)) / dU from the first-fold GLM artifact."""
    model = unwrap_model_artifact(glm_pipeline)
    if hasattr(model, "coef_") and hasattr(model, "feature_names_in_"):
        feature_names = [str(name) for name in model.feature_names_in_.tolist()]
        if "U" not in feature_names:
            raise ValueError(f"Expected fitted GLM feature names to include 'U'. Got: {feature_names}")
        return float(np.asarray(model.coef_, dtype=float)[0, feature_names.index("U")])

    # Legacy sklearn Pipeline path retained for older artifacts.
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
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
        if not hasattr(transformer, "named_steps") or "scaler" not in transformer.named_steps:
            continue
        col_list = list(cols)
        if "U" not in col_list:
            continue
        std_u = float(transformer.named_steps["scaler"].scale_[col_list.index("U")])
        return w_u / std_u
    raise ValueError("Could not find a StandardScaler containing 'U' in the GLM pipeline preprocessor.")


def extract_glm_acceptance_coefficients(glm_pipeline: Any) -> dict[str, Any]:
    """Extract processed-space acceptance coefficients from a fitted GLM artifact."""
    model = unwrap_model_artifact(glm_pipeline)
    if hasattr(model, "coef_") and hasattr(model, "intercept_") and hasattr(model, "feature_names_in_"):
        feature_names = [str(name) for name in model.feature_names_in_.tolist()]
        if "U" not in feature_names:
            raise ValueError(f"Expected fitted GLM feature names to include 'U'. Got: {feature_names}")
        coef = np.asarray(model.coef_[0], dtype=float)
        u_index = feature_names.index("U")
        return {
            "formula": "logit(p_accept(z, u)) = intercept + beta_z^T z + beta_u * u",
            "x_feature_names": [name for name in feature_names if name != "U"],
            "x_coef": [float(val) for val in np.delete(coef, u_index).tolist()],
            "u_coef": float(coef[u_index]),
            "intercept": float(np.asarray(model.intercept_, dtype=float).reshape(-1)[0]),
            "probability_target": "acceptance",
        }

    # Legacy pipeline path retained for older artifacts.
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed_names = list(preprocessor.get_feature_names_out())
    if "U" not in transformed_names:
        raise ValueError(f"Expected transformed feature names to include 'U'. Got: {transformed_names}")
    coef = np.asarray(classifier.coef_[0], dtype=float)
    u_index = transformed_names.index("U")
    return {
        "formula": "logit(p_churn(z, u)) = intercept + beta_z^T z + beta_u * u",
        "x_feature_names": [name for name in transformed_names if name != "U"],
        "x_coef": [float(val) for val in np.delete(coef, u_index).tolist()],
        "u_coef": float(coef[u_index]),
        "intercept": float(classifier.intercept_[0]),
        "probability_target": "churn",
    }


def extract_linear_loss_coefficients(linear_model: Any) -> dict[str, Any]:
    """Extract coefficients from a fitted linear/ridge loss artifact."""
    model = unwrap_model_artifact(linear_model)
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise ValueError("Expected a fitted linear model with coef_ and intercept_.")
    if not hasattr(model, "feature_names_in_"):
        raise ValueError("Expected the fitted linear model to expose feature_names_in_.")

    coef = np.asarray(model.coef_, dtype=float)
    feature_names = [str(name) for name in model.feature_names_in_.tolist()]
    if coef.shape != (len(feature_names),):
        raise ValueError(
            "Expected one coefficient per model input feature. "
            f"Got coefficient shape {coef.shape} for {len(feature_names)} features."
        )

    return {
        "formula": "loss_hat(z) = intercept + gamma_z^T z",
        "x_feature_names": feature_names,
        "x_coef": [float(val) for val in coef.tolist()],
        "intercept": float(np.asarray(model.intercept_, dtype=float).reshape(-1)[0]),
    }


def extract_model_based_coefficients(acceptance_model: Any, loss_model: Any) -> dict[str, dict[str, Any]] | None:
    """Extract printable coefficient summaries for supported model-based artifacts."""
    try:
        acceptance = extract_glm_acceptance_coefficients(acceptance_model)
        loss = extract_linear_loss_coefficients(loss_model)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return {"acceptance": acceptance, "loss": loss}


__all__ = [
    "AcceptanceModelType",
    "AcceptanceSelection",
    "FEATURE_COLS",
    "FEATURE_COLS_GLM",
    "FEATURE_COLS_XGB",
    "LOSS_FEATURE_COLS",
    "ACCEPTANCE_STATE_COLS",
    "ModelArtifactBundle",
    "ModelType",
    "LossModelType",
    "dataset_column_roles",
    "dataset_csv_path",
    "eligible_csv_row_indices",
    "sample_csv_row_indices",
    "load_acceptance_artifact",
    "load_loss_artifact",
    "load_model_artifact_pair",
    "load_model_artifacts",
    "load_x_frame",
    "load_x_array",
    "load_observed_u_array",
    "load_observed_loss_array",
    "load_mean_observed_acceptance",
    "resolve_model_artifact_ids",
    "unwrap_model_artifact",
    "extract_glm_u_coef",
    "extract_glm_acceptance_coefficients",
    "extract_linear_loss_coefficients",
    "extract_model_based_coefficients",
]
