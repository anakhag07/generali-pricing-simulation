"""Metadata for the canonical real-data optimization dataset and artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict


DATA_DIR = Path(__file__).parent
DATASET_FILENAME = "dataset.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
DATASET_SCHEMA_VERSION = "glm-acceptance-v1"
DATASET_SOURCE = "Current GLM acceptance CSV export."
DATASET_DESCRIPTION = (
    "Canonical real-data source rows for pricing optimization. The current file "
    "is the GLM acceptance export and is shared by GLM and XGB runtime loaders."
)

BASE_COLS: tuple[str, ...] = (
    "X_age",
    "X_bonus_malus_rating",
    "X_distr_channel",
    "X_vehicle_type",
    "X_ttm_claims",
    "X_policy_count",
    "X_risk_code",
    "X_vehicle_age",
    "X_policy_tenure",
)
FEATURE_COLS_GLM: tuple[str, ...] = BASE_COLS + (
    "X_policy_premium",
    "X_prev_renewal_perc",
    "X_year",
)
FEATURE_COLS_XGB: tuple[str, ...] = BASE_COLS + ("X_policy_premium",)
FEATURE_COLS: tuple[str, ...] = FEATURE_COLS_XGB
LOSS_FEATURE_COLS: tuple[str, ...] = BASE_COLS
ACCEPTANCE_STATE_COLS: tuple[str, ...] = BASE_COLS + ("X_policy_premium",)

OBSERVED_U_COL = "U"
ACCEPTANCE_PROBABILITY_COL = "prob_acceptance"
PREMIUM_COL = "X_policy_premium"
PREMIUM_COL_INDEX = FEATURE_COLS_GLM.index(PREMIUM_COL)

REQUIRED_DATASET_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *FEATURE_COLS_GLM,
            OBSERVED_U_COL,
            ACCEPTANCE_PROBABILITY_COL,
            "churn_prediction",
        )
    )
)


class ArtifactSpec(TypedDict):
    """Paths and preprocessing notes for a model artifact bundle."""

    path: Path
    contains_feature_processor: bool
    description: str


class ModelArtifactSpec(TypedDict):
    """Acceptance/loss artifact paths for one model family."""

    acceptance: ArtifactSpec
    loss: ArtifactSpec


MODEL_ARTIFACTS: dict[Literal["glm", "xgb"], ModelArtifactSpec] = {
    "glm": {
        "acceptance": {
            "path": DATA_DIR / "models" / "linear" / "glm_logistic_churn_feat_preproc.pkl",
            "contains_feature_processor": True,
            "description": "GLM churn classifier with bundled feature preprocessor.",
        },
        "loss": {
            "path": DATA_DIR / "models" / "linear" / "linear_regression_policy_premium_feat_preproc.pkl",
            "contains_feature_processor": True,
            "description": "Linear expected-loss regressor with bundled feature preprocessor.",
        },
    },
    "xgb": {
        "acceptance": {
            "path": DATA_DIR / "models" / "xgb" / "xgb_classifier_churn_feat_preproc.pkl",
            "contains_feature_processor": True,
            "description": "XGBoost churn classifier with bundled feature preprocessor.",
        },
        "loss": {
            "path": DATA_DIR / "models" / "xgb" / "xgb_regressor_policy_premium_feat_preproc.pkl",
            "contains_feature_processor": True,
            "description": "XGBoost expected-loss regressor with bundled feature preprocessor.",
        },
    },
}

MODEL_FEATURE_COLS: dict[Literal["glm", "xgb"], tuple[str, ...]] = {
    "glm": FEATURE_COLS_GLM,
    "xgb": FEATURE_COLS_XGB,
}

DATASET_COLUMN_DESCRIPTIONS: dict[str, str] = {
    "X_age": "Customer age feature from the source export.",
    "X_bonus_malus_rating": "Bonus-malus rating feature.",
    "X_distr_channel": "Distribution channel feature.",
    "X_vehicle_type": "Vehicle type feature; encoded by loaders when non-numeric.",
    "X_ttm_claims": "Trailing claims count feature.",
    "X_policy_count": "Policy count feature.",
    "X_risk_code": "Risk code feature; encoded by loaders when non-numeric.",
    "X_vehicle_age": "Vehicle age feature.",
    "X_policy_tenure": "Policy tenure feature.",
    "X_policy_premium": "Baseline policy premium used in the revenue term.",
    "X_prev_renewal_perc": "Previous renewal percentage feature used by GLM-state configs.",
    "X_year": "Policy year feature used by GLM-state configs.",
    OBSERVED_U_COL: "Observed historical centered pricing multiplier.",
    ACCEPTANCE_PROBABILITY_COL: "Exported observed/model acceptance probability.",
    "churn_prediction": "Exported churn prediction diagnostic from the GLM acceptance CSV.",
}


__all__ = [
    "ACCEPTANCE_PROBABILITY_COL",
    "ACCEPTANCE_STATE_COLS",
    "BASE_COLS",
    "DATA_DIR",
    "DATASET_COLUMN_DESCRIPTIONS",
    "DATASET_DESCRIPTION",
    "DATASET_FILENAME",
    "DATASET_PATH",
    "DATASET_SCHEMA_VERSION",
    "DATASET_SOURCE",
    "FEATURE_COLS",
    "FEATURE_COLS_GLM",
    "FEATURE_COLS_XGB",
    "LOSS_FEATURE_COLS",
    "MODEL_ARTIFACTS",
    "MODEL_FEATURE_COLS",
    "OBSERVED_U_COL",
    "PREMIUM_COL",
    "PREMIUM_COL_INDEX",
    "REQUIRED_DATASET_COLUMNS",
]
