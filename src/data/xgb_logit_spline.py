"""Portable per-policy logit-spline acceptance artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from data.feature_processor import FeatureProcessor


ARTIFACT_SCHEMA_VERSION = 1
SMOOTHER_NAME = "logit_smoothing_spline_isotonic"


@dataclass(frozen=True)
class XGBLogitSplineArtifactData:
    """Portable arrays defining one churn logit spline per insurance policy."""

    policy_ids: np.ndarray
    row_indices: np.ndarray
    action_grid: np.ndarray
    knots: np.ndarray
    coefficients: np.ndarray
    degrees: np.ndarray
    churn_min: np.ndarray
    churn_max: np.ndarray
    upper_slopes: np.ndarray
    source_sha256: str = ""

    def __post_init__(self) -> None:
        policy_ids = np.asarray(self.policy_ids, dtype=str)
        row_indices = np.asarray(self.row_indices, dtype=int)
        action_grid = np.asarray(self.action_grid, dtype=float)
        knots = np.asarray(self.knots, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        degrees = np.asarray(self.degrees, dtype=int)
        churn_min = np.asarray(self.churn_min, dtype=float)
        churn_max = np.asarray(self.churn_max, dtype=float)
        upper_slopes = np.asarray(self.upper_slopes, dtype=float)

        n_policies = policy_ids.size
        if policy_ids.ndim != 1 or n_policies == 0:
            raise ValueError("policy_ids must be a non-empty 1D array.")
        if np.unique(policy_ids).size != n_policies:
            raise ValueError("policy_ids must be unique.")
        if row_indices.shape != (n_policies,) or np.unique(row_indices).size != n_policies:
            raise ValueError("row_indices must contain one unique row per policy.")
        if action_grid.ndim != 1 or action_grid.size < 2 or np.any(np.diff(action_grid) <= 0):
            raise ValueError("action_grid must be a strictly increasing 1D array.")
        if knots.ndim != 2 or knots.shape[0] != n_policies:
            raise ValueError("knots must contain one row per policy.")
        if coefficients.ndim != 2 or coefficients.shape[0] != n_policies:
            raise ValueError("coefficients must contain one row per policy.")
        for name, values in {
            "degrees": degrees,
            "churn_min": churn_min,
            "churn_max": churn_max,
            "upper_slopes": upper_slopes,
        }.items():
            if values.shape != (n_policies,):
                raise ValueError(f"{name} must contain one value per policy.")
        numeric_arrays = (action_grid, knots, coefficients, churn_min, churn_max, upper_slopes)
        if not all(np.isfinite(values).all() for values in numeric_arrays):
            raise ValueError("Spline artifact arrays must be finite.")
        if np.any((churn_min < 0.0) | (churn_min > 1.0)) or np.any(
            (churn_max < 0.0) | (churn_max > 1.0)
        ):
            raise ValueError("Boundary churn probabilities must lie in [0, 1].")

        object.__setattr__(self, "policy_ids", policy_ids)
        object.__setattr__(self, "row_indices", row_indices)
        object.__setattr__(self, "action_grid", action_grid)
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "degrees", degrees)
        object.__setattr__(self, "churn_min", churn_min)
        object.__setattr__(self, "churn_max", churn_max)
        object.__setattr__(self, "upper_slopes", upper_slopes)
        object.__setattr__(self, "source_sha256", str(self.source_sha256))

    @property
    def u_min(self) -> float:
        return float(self.action_grid[0])

    @property
    def u_max(self) -> float:
        return float(self.action_grid[-1])


class _LegacyXGBoostEnsembleWrapper:
    """Compatibility target for the source artifact's notebook-local wrapper."""

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        frame = frame.reset_index(drop=True)
        fold_acceptance = []
        for model, bundle in zip(self._models, self._prep):
            x_cols = [col for col in bundle.get("x_feature_cols", ()) if col in frame.columns]
            processed = bundle["preprocessor"].transform(frame.loc[:, x_cols]).reset_index(drop=True)
            if self._u_col in frame.columns:
                processed = pd.concat(
                    [processed, frame.loc[:, [self._u_col]].reset_index(drop=True)],
                    axis=1,
                )
            if hasattr(model, "feature_names_in_"):
                processed = processed.reindex(columns=list(model.feature_names_in_))
            fold_acceptance.append(np.asarray(model.predict_proba(processed)[:, 1], dtype=float))
        mean_acceptance = np.mean(fold_acceptance, axis=0)
        # The legacy wrapper intentionally exposes [acceptance, churn].
        return np.column_stack([mean_acceptance, 1.0 - mean_acceptance])


class _LegacyArtifactUnpickler(pickle.Unpickler):
    """Map source-repository classes onto local compatibility implementations."""

    def find_class(self, module: str, name: str) -> Any:
        if name == "FeatureProcessor" and module in {"__main__", "preprocessing"}:
            return FeatureProcessor
        if name == "XGBoostEnsembleWrapper" and module in {
            "__main__",
            "black_box_objective",
            "smoothing_wrapper",
        }:
            return _LegacyXGBoostEnsembleWrapper
        return super().find_class(module, name)


def load_legacy_smoothing_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate the trusted source XGBoost smoothing bundle."""
    artifact_path = Path(path)
    with artifact_path.open("rb") as handle:
        artifact = _LegacyArtifactUnpickler(handle).load()
    if not isinstance(artifact, dict):
        raise ValueError("Legacy smoothing artifact must contain a dictionary.")
    required = {
        "smoothing_wrapper",
        "model_features",
        "profiles",
        "sigma",
        "MAX_PI_FIT",
    }
    missing = sorted(required.difference(artifact))
    if missing:
        raise ValueError(f"Legacy smoothing artifact is missing keys: {missing}")
    return artifact


def source_churn_grid(
    artifact: Mapping[str, Any],
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """Batch-score source XGBoost churn on the artifact's covered profiles."""
    profiles = artifact["profiles"]
    if not isinstance(profiles, Mapping) or not profiles:
        raise ValueError("Legacy artifact profiles must be a non-empty mapping.")
    policy_ids = np.asarray([str(policy_id) for policy_id in profiles], dtype=str)
    profile_frame = pd.concat(
        [profiles[policy_id].iloc[[0]] for policy_id in profiles],
        ignore_index=True,
    )
    profile_frame["id"] = policy_ids

    max_pi_fit = int(artifact["MAX_PI_FIT"])
    action_grid = np.arange(max_pi_fit + 1, dtype=float) / 100.0
    n_actions = action_grid.size
    expanded = profile_frame.iloc[
        np.repeat(np.arange(profile_frame.shape[0]), n_actions)
    ].reset_index(drop=True)
    expanded["U"] = np.tile(action_grid, profile_frame.shape[0])

    model_features = [str(col) for col in artifact["model_features"]]
    missing_features = [col for col in model_features if col not in expanded.columns]
    if missing_features:
        raise ValueError(f"Covered profiles are missing model features: {missing_features}")
    probabilities = np.asarray(
        artifact["smoothing_wrapper"].predict_proba(expanded.loc[:, model_features]),
        dtype=float,
    )
    if probabilities.shape != (expanded.shape[0], 2):
        raise ValueError("Legacy wrapper must return two probabilities per row.")
    churn = probabilities[:, 1].reshape(profile_frame.shape[0], n_actions)
    if not np.isfinite(churn).all() or np.any((churn < 0.0) | (churn > 1.0)):
        raise ValueError("Source churn probabilities must be finite and lie in [0, 1].")
    return policy_ids, profile_frame, action_grid, churn


def fit_logit_spline_artifact(
    *,
    policy_ids: Sequence[str],
    row_indices: Sequence[int],
    action_grid: np.ndarray,
    churn_grid: np.ndarray,
    weights: np.ndarray | None,
    source_sha256: str = "",
) -> XGBLogitSplineArtifactData:
    """Fit deterministic isotonic logit splines to a policy-by-action churn grid."""
    policy_ids_arr = np.asarray(policy_ids, dtype=str)
    row_indices_arr = np.asarray(row_indices, dtype=int)
    action_arr = np.asarray(action_grid, dtype=float)
    churn_arr = np.asarray(churn_grid, dtype=float)
    if churn_arr.shape != (policy_ids_arr.size, action_arr.size):
        raise ValueError("churn_grid must have shape (n_policies, n_actions).")
    if weights is None:
        weights_arr = np.ones(action_arr.size, dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.shape != action_arr.shape:
            raise ValueError("weights must align with action_grid.")
        if not np.isfinite(weights_arr).all() or np.any(weights_arr < 0.0):
            raise ValueError("weights must be finite and non-negative.")
        if not np.any(weights_arr > 0.0):
            raise ValueError("weights must contain at least one positive value.")

    splines = []
    churn_min = []
    churn_max = []
    upper_slopes = []
    eps = 1e-6
    for raw_churn in churn_arr:
        isotonic_churn = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
        ).fit_transform(action_arr, raw_churn, sample_weight=weights_arr)
        spline = make_smoothing_spline(
            action_arr,
            logit(np.clip(isotonic_churn, eps, 1.0 - eps)),
            w=None,
            lam=None,
        )
        splines.append(spline)
        q_min = float(expit(spline(action_arr[0])))
        q_max = float(expit(spline(action_arr[-1])))
        h = max(float(action_arr[-1] - action_arr[0]) * 1e-4, 1e-6)
        d_logit_du = float((spline(action_arr[-1]) - spline(action_arr[-1] - h)) / h)
        churn_min.append(q_min)
        churn_max.append(q_max)
        upper_slopes.append(q_max * (1.0 - q_max) * d_logit_du)

    knot_lengths = {np.asarray(spline.t).size for spline in splines}
    coefficient_lengths = {np.asarray(spline.c).size for spline in splines}
    if len(knot_lengths) != 1 or len(coefficient_lengths) != 1:
        raise ValueError("Fitted splines do not share a portable array shape.")
    return XGBLogitSplineArtifactData(
        policy_ids=policy_ids_arr,
        row_indices=row_indices_arr,
        action_grid=action_arr,
        knots=np.stack([np.asarray(spline.t, dtype=float) for spline in splines]),
        coefficients=np.stack([np.asarray(spline.c, dtype=float) for spline in splines]),
        degrees=np.asarray([int(spline.k) for spline in splines], dtype=int),
        churn_min=np.asarray(churn_min, dtype=float),
        churn_max=np.asarray(churn_max, dtype=float),
        upper_slopes=np.asarray(upper_slopes, dtype=float),
        source_sha256=source_sha256,
    )


def canonical_row_indices_for_policy_ids(
    dataset_path: str | Path,
    policy_ids: Sequence[str],
    *,
    id_col: str = "id",
) -> np.ndarray:
    """Resolve policy IDs to unique zero-based canonical CSV row positions."""
    requested = np.asarray(policy_ids, dtype=str)
    id_frame = pd.read_csv(
        dataset_path,
        sep=";",
        usecols=[id_col],
        dtype={id_col: "string"},
    )
    positions: dict[str, list[int]] = {}
    requested_set = set(requested.tolist())
    for position, value in enumerate(id_frame[id_col]):
        if pd.isna(value):
            continue
        policy_id = str(value)
        if policy_id in requested_set:
            positions.setdefault(policy_id, []).append(position)
    missing = [policy_id for policy_id in requested if policy_id not in positions]
    duplicate = [policy_id for policy_id in requested if len(positions.get(policy_id, ())) != 1]
    if missing:
        raise ValueError(f"Covered policy IDs are missing from the canonical dataset: {missing[:5]}")
    if duplicate:
        raise ValueError(f"Covered policy IDs must map to one canonical row: {duplicate[:5]}")
    return np.asarray([positions[policy_id][0] for policy_id in requested], dtype=int)


def prepare_xgb_logit_spline_artifact(
    source_path: str | Path,
    dataset_path: str | Path,
) -> XGBLogitSplineArtifactData:
    """Convert a legacy smoothing bundle into portable logit-spline arrays."""
    source = Path(source_path)
    artifact = load_legacy_smoothing_artifact(source)
    policy_ids, _, action_grid, churn_grid = source_churn_grid(artifact)
    row_indices = canonical_row_indices_for_policy_ids(dataset_path, policy_ids)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    return fit_logit_spline_artifact(
        policy_ids=policy_ids,
        row_indices=row_indices,
        action_grid=action_grid,
        churn_grid=churn_grid,
        weights=np.asarray(artifact["sigma"], dtype=float),
        source_sha256=source_sha256,
    )


def save_xgb_logit_spline_artifact(
    artifact: XGBLogitSplineArtifactData,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a portable spline artifact without pickle-backed object arrays."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=int),
        smoother_name=np.asarray(SMOOTHER_NAME),
        probability_target=np.asarray("acceptance"),
        source_sha256=np.asarray(artifact.source_sha256),
        policy_ids=artifact.policy_ids,
        row_indices=artifact.row_indices,
        action_grid=artifact.action_grid,
        knots=artifact.knots,
        coefficients=artifact.coefficients,
        degrees=artifact.degrees,
        churn_min=artifact.churn_min,
        churn_max=artifact.churn_max,
        upper_slopes=artifact.upper_slopes,
    )
    return output_path


def load_xgb_logit_spline_artifact(path: str | Path) -> XGBLogitSplineArtifactData:
    """Load and validate a portable XGBoost logit-spline artifact."""
    with np.load(path, allow_pickle=False) as loaded:
        if int(loaded["schema_version"]) != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Unsupported XGB logit-spline artifact schema version.")
        if str(loaded["smoother_name"]) != SMOOTHER_NAME:
            raise ValueError("Artifact does not contain XGB logit-spline curves.")
        return XGBLogitSplineArtifactData(
            policy_ids=loaded["policy_ids"].copy(),
            row_indices=loaded["row_indices"].copy(),
            action_grid=loaded["action_grid"].copy(),
            knots=loaded["knots"].copy(),
            coefficients=loaded["coefficients"].copy(),
            degrees=loaded["degrees"].copy(),
            churn_min=loaded["churn_min"].copy(),
            churn_max=loaded["churn_max"].copy(),
            upper_slopes=loaded["upper_slopes"].copy(),
            source_sha256=str(loaded["source_sha256"]),
        )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "SMOOTHER_NAME",
    "XGBLogitSplineArtifactData",
    "canonical_row_indices_for_policy_ids",
    "fit_logit_spline_artifact",
    "load_xgb_logit_spline_artifact",
    "prepare_xgb_logit_spline_artifact",
    "save_xgb_logit_spline_artifact",
    "source_churn_grid",
]
