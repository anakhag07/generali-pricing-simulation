"""Metadata for the canonical 052726 real-data optimization dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypedDict


DATA_DIR = Path(__file__).parent
DATASET_FILENAME = "dataset.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
DATASET_SCHEMA_VERSION = "generali-052726-v1"
DATASET_SOURCE = "Generali 052726 raw single-year export with separate acceptance/loss artifacts."
DATASET_DESCRIPTION = (
    "Canonical 052726 real-data source rows for pricing optimization. The objective "
    "uses only the model artifact X covariates and policy-generated U; observed "
    "targets/actions are retained only for diagnostics."
)

# X covariates consumed by the acceptance black-box before artifact preprocessing.
ACCEPTANCE_STATE_COLS: tuple[str, ...] = (
    "X_vehicle_value_at_new",
    "X_driving_license_years",
    "X_vehicle_weight",
    "X_vehicle_power",
    "X_fuel_type_vehicle",
    "X_policy_tenure",
    "X_policy_count",
    "X_gender",
    "X_ttm_claims",
    "X_customer_segment",
    "X_installment",
    "X_bonus_malus_rating",
    "X_vehicle_type",
    "X_district",
    "X_age",
    "X_distr_channel",
    "X_claim_tot_value",
    "X_vehicle_age",
    "X_policy_premium",
)

# X covariates consumed by the financial-loss black-box before artifact preprocessing.
LOSS_FEATURE_COLS: tuple[str, ...] = (
    "X_vehicle_value_at_new",
    "X_driving_license_years",
    "X_vehicle_weight",
    "X_vehicle_power",
    "X_fuel_type_vehicle",
    "X_policy_tenure",
    "X_policy_count",
    "X_gender",
    "X_ttm_claims",
    "X_customer_segment",
    "X_installment",
    "X_bonus_malus_rating",
    "X_vehicle_type",
    "X_district",
    "X_age",
    "X_distr_channel",
    "X_claim_tot_value",
    "X_vehicle_age",
)

PREMIUM_COL = "X_policy_premium"
OBSERVED_U_COL = "U"
OBSERVED_CHURN_COL = "is_churn"
LOSS_TARGET_COL = "Y_G_Loss"

# Historical compatibility name. The 052726 CSV has no exported acceptance
# probability column; observed acceptance is computed as 1 - is_churn.
ACCEPTANCE_PROBABILITY_COL = OBSERVED_CHURN_COL

FEATURE_COLS: tuple[str, ...] = ACCEPTANCE_STATE_COLS
FEATURE_COLS_GLM: tuple[str, ...] = FEATURE_COLS
FEATURE_COLS_XGB: tuple[str, ...] = FEATURE_COLS
MODEL_FEATURE_COLS: dict[Literal["glm", "xgb"], tuple[str, ...]] = {
    "glm": FEATURE_COLS_GLM,
    "xgb": FEATURE_COLS_XGB,
}
PREMIUM_COL_INDEX = FEATURE_COLS.index(PREMIUM_COL)

USED_X_COLS: tuple[str, ...] = tuple(
    dict.fromkeys((*ACCEPTANCE_STATE_COLS, *LOSS_FEATURE_COLS))
)
LOOKAHEAD_X_COLS: tuple[str, ...] = ("X_upcoming_premium",)
UNUSED_X_COLS: tuple[str, ...] = LOOKAHEAD_X_COLS
ACTION_COLS: tuple[str, ...] = (OBSERVED_U_COL,)
TARGET_COLS: tuple[str, ...] = (LOSS_TARGET_COL, OBSERVED_CHURN_COL)
ID_COLS: tuple[str, ...] = ("dummy_id", "id")
DATE_COLS: tuple[str, ...] = ("start_date", "end_date", "check_date")
OBJECTIVE_EXCLUDED_COLS: tuple[str, ...] = tuple(
    dict.fromkeys((*UNUSED_X_COLS, *ACTION_COLS, *TARGET_COLS, *ID_COLS, *DATE_COLS))
)

REQUIRED_DATASET_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((*USED_X_COLS, OBSERVED_U_COL, LOSS_TARGET_COL, OBSERVED_CHURN_COL))
)
OBJECTIVE_REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((*USED_X_COLS, PREMIUM_COL))
)
OBSERVED_DIAGNOSTIC_REQUIRED_COLUMNS: tuple[str, ...] = (
    OBSERVED_U_COL,
    OBSERVED_CHURN_COL,
)


class ArtifactSpec(TypedDict):
    """Paths and preprocessing notes for a model artifact bundle."""

    path: Path
    contains_feature_processor: bool
    probability_target: Literal["acceptance", "churn", "none"]
    description: str


class ModelArtifactSpec(TypedDict):
    """Acceptance/loss artifact paths for one model family."""

    acceptance: ArtifactSpec
    loss: ArtifactSpec


MODEL_ARTIFACTS: dict[Literal["glm", "xgb"], ModelArtifactSpec] = {
    "glm": {
        "acceptance": {
            "path": DATA_DIR / "models" / "linear" / "acceptance_model_linear_cv_20260527_142758.pkl",
            "contains_feature_processor": True,
            "probability_target": "acceptance",
            "description": "First-fold LogisticRegression acceptance model with fitted FeatureProcessor.",
        },
        "loss": {
            "path": DATA_DIR / "models" / "linear" / "financial_loss_model_linear_cv_20260527_142758.pkl",
            "contains_feature_processor": True,
            "probability_target": "none",
            "description": "First-fold Ridge financial-loss model with fitted FeatureProcessor.",
        },
    },
    "xgb": {
        "acceptance": {
            "path": DATA_DIR / "models" / "xgb" / "acceptance_model_xgb_cv_20260527_151725.pkl",
            "contains_feature_processor": True,
            "probability_target": "acceptance",
            "description": "First-fold XGBoost acceptance model with fitted FeatureProcessor.",
        },
        "loss": {
            "path": DATA_DIR / "models" / "xgb" / "financial_loss_model_xgb_cv_20260527_151725.pkl",
            "contains_feature_processor": True,
            "probability_target": "none",
            "description": "First-fold XGBoost financial-loss model with fitted FeatureProcessor.",
        },
    },
}

DATASET_COLUMN_DESCRIPTIONS: dict[str, str] = {
    "X_vehicle_value_at_new": "Vehicle value at new; model X covariate.",
    "X_driving_license_years": "Driving-license tenure; model X covariate.",
    "X_vehicle_weight": "Vehicle weight category; model X covariate.",
    "X_vehicle_power": "Vehicle power category; model X covariate.",
    "X_fuel_type_vehicle": "Fuel type category; model X covariate.",
    "X_policy_tenure": "Policy tenure; model X covariate.",
    "X_policy_count": "Policy count; model X covariate.",
    "X_gender": "Customer gender category; model X covariate.",
    "X_ttm_claims": "Trailing claims count; model X covariate.",
    "X_customer_segment": "Customer segment category; model X covariate.",
    "X_installment": "Installment category; model X covariate.",
    "X_bonus_malus_rating": "Bonus-malus rating; model X covariate.",
    "X_vehicle_type": "Vehicle type category; model X covariate.",
    "X_district": "District code; model X covariate.",
    "X_age": "Customer age; model X covariate.",
    "X_distr_channel": "Distribution channel category; model X covariate.",
    "X_claim_tot_value": "Total claim value feature; model X covariate.",
    "X_vehicle_age": "Vehicle age; model X covariate.",
    PREMIUM_COL: "Baseline policy premium used for deterministic revenue.",
    "X_upcoming_premium": "Lookahead premium column retained in CSV but excluded from objective X.",
    OBSERVED_U_COL: "Historical observed pricing action; diagnostics only.",
    OBSERVED_CHURN_COL: "Observed churn target; diagnostics only.",
    LOSS_TARGET_COL: "Observed financial-loss target; diagnostics only.",
    "dummy_id": "Source row identifier; excluded from objective X.",
    "id": "Policy/customer identifier; excluded from objective X.",
    "start_date": "Policy start date; excluded from objective X.",
    "end_date": "Policy end date; excluded from objective X.",
    "check_date": "Source check date; excluded from objective X.",
}


__all__ = [
    "ACCEPTANCE_PROBABILITY_COL",
    "ACCEPTANCE_STATE_COLS",
    "ACTION_COLS",
    "DATA_DIR",
    "DATASET_COLUMN_DESCRIPTIONS",
    "DATASET_DESCRIPTION",
    "DATASET_FILENAME",
    "DATASET_PATH",
    "DATASET_SCHEMA_VERSION",
    "DATASET_SOURCE",
    "DATE_COLS",
    "FEATURE_COLS",
    "FEATURE_COLS_GLM",
    "FEATURE_COLS_XGB",
    "ID_COLS",
    "LOOKAHEAD_X_COLS",
    "LOSS_FEATURE_COLS",
    "LOSS_TARGET_COL",
    "MODEL_ARTIFACTS",
    "MODEL_FEATURE_COLS",
    "OBJECTIVE_EXCLUDED_COLS",
    "OBJECTIVE_REQUIRED_COLUMNS",
    "OBSERVED_CHURN_COL",
    "OBSERVED_DIAGNOSTIC_REQUIRED_COLUMNS",
    "OBSERVED_U_COL",
    "PREMIUM_COL",
    "PREMIUM_COL_INDEX",
    "REQUIRED_DATASET_COLUMNS",
    "TARGET_COLS",
    "UNUSED_X_COLS",
    "USED_X_COLS",
]
