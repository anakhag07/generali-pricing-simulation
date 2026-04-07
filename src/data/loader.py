"""Data loading utilities for real insurance pricing datasets."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

# Directory containing model artifacts and CSV datasets
_DATA_DIR = Path(__file__).parent

# Base 9 feature cols shared by both models (no premium, no U, no extra)
_BASE_COLS: list[str] = [
    "X_age",
    "X_bonus_malus_rating",
    "X_distr_channel",
    "X_vehicle_type",
    "X_ttm_claims",
    "X_policy_count",
    "X_risk_code",
    "X_vehicle_age",
    "X_policy_tenure",
]

# GLM state features: 12 cols (9 base + premium + prev_renewal_perc + year).
# The GLM acceptance CSV has all 12; extra cols are for the policy only.
FEATURE_COLS_GLM: list[str] = _BASE_COLS + [
    "X_policy_premium",   # index 9; used in revenue term u * premium(x)
    "X_prev_renewal_perc",
    "X_year",
]

# XGB state features: 10 cols (9 base + premium).
# The XGB acceptance CSV doesn't carry X_prev_renewal_perc / X_year.
FEATURE_COLS_XGB: list[str] = _BASE_COLS + [
    "X_policy_premium",   # index 9; used in revenue term u * premium(x)
]

# Default alias — use when you don't need model-specific differences
FEATURE_COLS = FEATURE_COLS_XGB

# Columns consumed by the loss model (no premium, no U)
LOSS_FEATURE_COLS: list[str] = _BASE_COLS

# Columns consumed by acceptance model state part: base + premium (no U)
ACCEPTANCE_STATE_COLS: list[str] = _BASE_COLS + ["X_policy_premium"]

_PREMIUM_COL_INDEX: int = 9  # index of X_policy_premium in both FEATURE_COLS variants

_ARTIFACT_PATHS: dict[str, dict[str, Path]] = {
    "glm": {
        "acceptance": _DATA_DIR / "model_artifacts" / "glm_logistic_prob_acceptance.pkl",
        "loss": _DATA_DIR / "model_artifacts" / "linear_regression_expected_fin_loss.pkl",
    },
    "xgb": {
        "acceptance": _DATA_DIR / "model_artifacts" / "xgb_classifier_prob_acceptance.pkl",
        "loss": _DATA_DIR / "model_artifacts" / "xgb_regressor_expected_fin_loss.pkl",
    },
}

_ACCEPTANCE_CSV_PATHS: dict[str, Path] = {
    "glm": _DATA_DIR
    / "dataset_bbox_optim_linear_models"
    / "df_acceptance_linear_model_black_box.csv",
    "xgb": _DATA_DIR
    / "dataset_bbox_optim_xgb_models"
    / "df_acceptance_xgb_black_box.csv",
}


def load_model_artifacts(model_type: Literal["glm", "xgb"]) -> tuple[Any, Any]:
    """Load and return (acceptance_model, loss_model) from pickle files.

    acceptance_model: GLM logistic Pipeline or XGBClassifier.
    loss_model: Linear Regression or XGBRegressor.
    """
    if model_type not in _ARTIFACT_PATHS:
        raise ValueError(f"model_type must be 'glm' or 'xgb', got '{model_type}'.")
    paths = _ARTIFACT_PATHS[model_type]
    with open(paths["acceptance"], "rb") as f:
        acceptance_model = pickle.load(f)
    with open(paths["loss"], "rb") as f:
        loss_model = pickle.load(f)
    return acceptance_model, loss_model


def load_x_array(model_type: Literal["glm", "xgb"], n_rows: int = 5000) -> np.ndarray:
    """Load first n_rows of real X features from the acceptance CSV.

    GLM: returns shape (n_rows, 12) using FEATURE_COLS_GLM (includes X_prev_renewal_perc, X_year).
    XGB: returns shape (n_rows, 10) using FEATURE_COLS_XGB (base + premium only).
    Metadata columns (1-Z, Z, Y, Y_hat, U, prob_acceptance, etc.) are excluded.
    """
    if model_type not in _ACCEPTANCE_CSV_PATHS:
        raise ValueError(f"model_type must be 'glm' or 'xgb', got '{model_type}'.")
    feature_cols = FEATURE_COLS_GLM if model_type == "glm" else FEATURE_COLS_XGB
    df = pd.read_csv(_ACCEPTANCE_CSV_PATHS[model_type], sep=";", nrows=n_rows)
    return df[feature_cols].to_numpy(dtype=float)


def extract_glm_u_coef(glm_pipeline: Any) -> float:
    """Extract effective d_logit/dU = w_U / std_U from a fitted GLM Pipeline.

    Uses the ColumnTransformer's feature names and the StandardScaler scale
    to compute the unscaled coefficient for U in the logistic regression.
    """
    preprocessor = glm_pipeline.named_steps["preprocessor"]
    classifier = glm_pipeline.named_steps["classifier"]

    # Get full output feature names from ColumnTransformer (sklearn >= 1.0)
    transformed_names = list(preprocessor.get_feature_names_out())
    # Names are like "pipeline-1__U", "pipeline-2__X_distr_channel_0", etc.
    u_indices = [i for i, name in enumerate(transformed_names) if name.endswith("__U") or name == "U"]
    if len(u_indices) != 1:
        raise ValueError(
            f"Expected exactly 1 'U' column in preprocessor output, found at indices: {u_indices}. "
            f"Available names: {transformed_names}"
        )
    i_U_out = u_indices[0]
    w_U = float(classifier.coef_[0, i_U_out])

    # Find std_U from the StandardScaler in the numeric sub-pipeline
    for _, transformer, cols in preprocessor.transformers_:
        if not hasattr(transformer, "named_steps"):
            continue
        if "scaler" not in transformer.named_steps:
            continue
        col_list = list(cols)
        if "U" not in col_list:
            continue
        i_U_numeric = col_list.index("U")
        std_U = float(transformer.named_steps["scaler"].scale_[i_U_numeric])
        return w_U / std_U

    raise ValueError("Could not find a StandardScaler containing 'U' in the GLM pipeline preprocessor.")


def extract_glm_churn_coefficients(glm_pipeline: Any) -> dict[str, Any]:
    """Extract raw-space churn coefficients from a fitted GLM Pipeline.

    Returns coefficients for the churn logit in the original feature space:
    ``logit(p_churn(x, u)) = intercept + beta_x^T x + beta_u * u``.
    """
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
        "formula": "logit(p_churn(x, u)) = intercept + beta_x^T x + beta_u * u",
        "x_feature_names": x_feature_names,
        "x_coef": [float(val) for val in x_coef.tolist()],
        "u_coef": u_coef,
        "intercept": intercept_raw,
    }


def extract_linear_loss_coefficients(linear_model: Any) -> dict[str, Any]:
    """Extract coefficients from a fitted linear-regression loss model.

    Returns coefficients for ``loss_hat(x) = intercept + gamma_x^T x``.
    """
    if not hasattr(linear_model, "coef_") or not hasattr(linear_model, "intercept_"):
        raise ValueError("Expected a fitted linear model with coef_ and intercept_.")
    if not hasattr(linear_model, "feature_names_in_"):
        raise ValueError("Expected the fitted linear model to expose feature_names_in_.")

    coef = np.asarray(linear_model.coef_, dtype=float)
    feature_names = [str(name) for name in linear_model.feature_names_in_.tolist()]
    if coef.shape != (len(feature_names),):
        raise ValueError(
            "Expected one coefficient per raw loss feature. "
            f"Got coefficient shape {coef.shape} for {len(feature_names)} features."
        )

    return {
        "formula": "loss_hat(x) = intercept + gamma_x^T x",
        "x_feature_names": feature_names,
        "x_coef": [float(val) for val in coef.tolist()],
        "intercept": float(linear_model.intercept_),
    }


def extract_model_based_coefficients(acceptance_model: Any, loss_model: Any) -> dict[str, dict[str, Any]] | None:
    """Extract printable coefficient summaries for supported model-based artifacts.

    Returns ``None`` for unsupported artifact types such as XGBoost.
    """
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
    "load_model_artifacts",
    "load_x_array",
    "extract_glm_u_coef",
    "extract_glm_churn_coefficients",
    "extract_linear_loss_coefficients",
    "extract_model_based_coefficients",
]
