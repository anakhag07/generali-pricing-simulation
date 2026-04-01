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

_CSV_PATHS: dict[str, dict[str, Path]] = {
    "glm": {
        "acceptance": _DATA_DIR
        / "dataset_bbox_optim_linear_models"
        / "df_acceptance_linear_model_black_box.csv",
        "loss": _DATA_DIR
        / "dataset_bbox_optim_linear_models"
        / "df_exp_financial_loss_linear_black_box.csv",
    },
    "xgb": {
        "acceptance": _DATA_DIR
        / "dataset_bbox_optim_xgb_models"
        / "df_acceptance_xgb_black_box.csv",
        "loss": _DATA_DIR
        / "dataset_bbox_optim_xgb_models"
        / "df_exp_financial_loss_xgb_black_box.csv",
    },
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
    if model_type not in _CSV_PATHS:
        raise ValueError(f"model_type must be 'glm' or 'xgb', got '{model_type}'.")
    feature_cols = FEATURE_COLS_GLM if model_type == "glm" else FEATURE_COLS_XGB
    df = pd.read_csv(_CSV_PATHS[model_type]["acceptance"], sep=";", nrows=n_rows)
    return df[feature_cols].to_numpy(dtype=float)


def load_csv_dataset(model_type: Literal["glm", "xgb"]) -> pd.DataFrame:
    """Load and join acceptance + loss CSVs for pre-computed CSV-based evaluation.

    Joins on 'id'. The GLM loss CSV stores U as a percentage change
    (range ~[-0.002, 0.418]); this is normalized to uplift-factor scale (+1.0)
    before joining so all U values are on the same [0.998, 1.418] scale.

    Returns DataFrame with columns: FEATURE_COLS + [U, prob_acceptance, Y_hat].
    """
    if model_type not in _CSV_PATHS:
        raise ValueError(f"model_type must be 'glm' or 'xgb', got '{model_type}'.")

    acc_df = pd.read_csv(_CSV_PATHS[model_type]["acceptance"], sep=";")
    loss_df = pd.read_csv(_CSV_PATHS[model_type]["loss"], sep=";")

    # Normalize GLM loss CSV U column from percentage-change to uplift-factor scale
    if model_type == "glm":
        loss_df = loss_df.copy()
        loss_df["U"] = loss_df["U"] + 1.0

    loss_sub = loss_df[["id", "Y_hat"]].rename(columns={"Y_hat": "Y_hat"})
    merged = acc_df.merge(loss_sub, on="id", how="inner")

    feature_cols = FEATURE_COLS_GLM if model_type == "glm" else FEATURE_COLS_XGB
    keep_cols = feature_cols + ["U", "prob_acceptance", "Y_hat"]
    return merged[keep_cols].reset_index(drop=True)


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


__all__ = [
    "FEATURE_COLS",
    "FEATURE_COLS_GLM",
    "FEATURE_COLS_XGB",
    "LOSS_FEATURE_COLS",
    "ACCEPTANCE_STATE_COLS",
    "load_model_artifacts",
    "load_x_array",
    "load_csv_dataset",
    "extract_glm_u_coef",
]
