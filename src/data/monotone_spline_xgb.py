"""Monotone-spline acceptance wrapper derived from the canonical XGB model."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import PPoly, PchipInterpolator, make_smoothing_spline
from sklearn.isotonic import IsotonicRegression


ARTIFACT_SCHEMA_VERSION = 2
SMOOTHER_NAME = "monotone_smoothing_spline"
MODEL_TYPE = "monotone_spline_xgb"
_MONOTONICITY_TOLERANCE = 1e-10
_PROBABILITY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class MonotoneSplineArtifactData:
    """Portable PCHIP coefficients for a deterministic cache of policy curves."""

    policy_ids: np.ndarray
    row_indices: np.ndarray
    action_grid: np.ndarray
    coefficients: np.ndarray
    churn_min: np.ndarray
    churn_max: np.ndarray
    upper_slopes: np.ndarray
    base_artifact_sha256: str = ""
    source_fold: int = 0

    def __post_init__(self) -> None:
        policy_ids = np.asarray(self.policy_ids, dtype=str)
        row_indices = np.asarray(self.row_indices, dtype=int)
        action_grid = np.asarray(self.action_grid, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
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
        if (
            action_grid.ndim != 1
            or action_grid.size < 2
            or not np.isfinite(action_grid).all()
            or np.any(np.diff(action_grid) <= 0.0)
        ):
            raise ValueError("action_grid must be finite, 1D, and strictly increasing.")
        expected = (n_policies, 4, action_grid.size - 1)
        if coefficients.shape != expected or not np.isfinite(coefficients).all():
            raise ValueError(f"coefficients must be finite with shape {expected}.")
        for name, values in {
            "churn_min": churn_min,
            "churn_max": churn_max,
            "upper_slopes": upper_slopes,
        }.items():
            if values.shape != (n_policies,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain one finite value per policy.")
        if np.any((churn_min < 0.0) | (churn_min > 1.0)) or np.any(
            (churn_max < 0.0) | (churn_max > 1.0)
        ):
            raise ValueError("Boundary churn probabilities must lie in [0, 1].")
        if np.any(upper_slopes < -_MONOTONICITY_TOLERANCE):
            raise ValueError("upper_slopes must be non-negative.")

        _validate_curves(action_grid, coefficients, churn_min, churn_max)
        object.__setattr__(self, "policy_ids", policy_ids)
        object.__setattr__(self, "row_indices", row_indices)
        object.__setattr__(self, "action_grid", action_grid)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "churn_min", churn_min)
        object.__setattr__(self, "churn_max", churn_max)
        object.__setattr__(self, "upper_slopes", np.maximum(upper_slopes, 0.0))
        object.__setattr__(self, "base_artifact_sha256", str(self.base_artifact_sha256))
        object.__setattr__(self, "source_fold", int(self.source_fold))

    @property
    def u_min(self) -> float:
        return float(self.action_grid[0])

    @property
    def u_max(self) -> float:
        return float(self.action_grid[-1])


def _validate_curves(
    action_grid: np.ndarray,
    coefficients: np.ndarray,
    churn_min: np.ndarray,
    churn_max: np.ndarray,
) -> None:
    left = action_grid[:-1, None]
    interior = (left + np.diff(action_grid)[:, None] * np.asarray([0.25, 0.5, 0.75])).ravel()
    validation_grid = np.sort(np.concatenate([action_grid, interior]))
    for index, values in enumerate(coefficients):
        evaluated = np.asarray(PPoly(values, action_grid)(validation_grid), dtype=float)
        if not np.isfinite(evaluated).all():
            raise ValueError(f"Policy curve {index} produces non-finite values.")
        if np.any(evaluated < -_PROBABILITY_TOLERANCE) or np.any(
            evaluated > 1.0 + _PROBABILITY_TOLERANCE
        ):
            raise ValueError(f"Policy curve {index} leaves the probability range.")
        if np.any(np.diff(evaluated) < -_MONOTONICITY_TOLERANCE):
            raise ValueError(f"Policy curve {index} is not monotone.")
        if not np.isclose(evaluated[0], churn_min[index], atol=1e-10, rtol=0.0):
            raise ValueError(f"Policy curve {index} does not match churn_min.")
        if not np.isclose(evaluated[-1], churn_max[index], atol=1e-10, rtol=0.0):
            raise ValueError(f"Policy curve {index} does not match churn_max.")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_monotone_spline_artifact(
    base_acceptance: Any,
    dataset: pd.DataFrame,
    row_indices: Sequence[int],
    *,
    base_artifact_path: str | Path,
    source_fold: int = 0,
    id_col: str = "id",
    u_col: str = "U",
    max_price_increase: int = 16,
    dense_grid_size: int = 500,
) -> MonotoneSplineArtifactData:
    """Fit the original probability-space monotone smoother over selected profiles."""
    selected = np.asarray(row_indices, dtype=int)
    profiles = dataset.iloc[selected].copy()
    policy_ids = profiles[id_col].astype("string").to_numpy(dtype=str)
    if np.unique(policy_ids).size != policy_ids.size:
        raise ValueError("Selected curve-cache rows must contain unique policy IDs.")

    action_fit = np.linspace(0.0, max_price_increase / 100.0, max_price_increase + 1)
    observed = profiles[u_col].round(2).value_counts(normalize=True)
    weights = observed.reindex(action_fit.round(2), fill_value=0.0).to_numpy(dtype=float)
    weights = np.where(weights > 0.0, weights, 1e-9)
    dense_grid = np.linspace(action_fit[0], action_fit[-1], dense_grid_size)
    coefficients: list[np.ndarray] = []
    churn_min: list[float] = []
    churn_max: list[float] = []
    upper_slopes: list[float] = []

    for (_, profile), policy_id in zip(profiles.iterrows(), policy_ids, strict=True):
        raw_grid = pd.DataFrame([profile.to_dict()] * action_fit.size)
        raw_grid[id_col] = policy_id
        raw_grid[u_col] = action_fit
        acceptance = np.asarray(
            base_acceptance.model.predict_proba(base_acceptance.model_frame(raw_grid))[:, 1],
            dtype=float,
        )
        churn = 1.0 - acceptance
        smooth = make_smoothing_spline(action_fit, churn, w=weights)
        dense_churn = np.clip(smooth(dense_grid), 0.0, 1.0)
        monotone_churn = IsotonicRegression(
            increasing=True, out_of_bounds="clip"
        ).fit_transform(dense_grid, dense_churn)
        curve = PchipInterpolator(dense_grid, np.clip(monotone_churn, 0.0, 1.0))
        h = max((dense_grid[-1] - dense_grid[0]) * 1e-4, 1e-6)
        slope = float((curve(dense_grid[-1]) - curve(dense_grid[-1] - h)) / h)
        coefficients.append(curve.c)
        churn_min.append(float(curve(dense_grid[0])))
        churn_max.append(float(curve(dense_grid[-1])))
        upper_slopes.append(max(slope, 0.0))

    return MonotoneSplineArtifactData(
        policy_ids=policy_ids,
        row_indices=selected,
        action_grid=dense_grid,
        coefficients=np.stack(coefficients),
        churn_min=np.asarray(churn_min),
        churn_max=np.asarray(churn_max),
        upper_slopes=np.asarray(upper_slopes),
        base_artifact_sha256=_sha256_file(base_artifact_path),
        source_fold=source_fold,
    )


class MonotoneSplineXGBAcceptance:
    """Use cached monotone curves and raw XGB fallback, matching the source wrapper."""

    model_type = MODEL_TYPE
    artifact_id = MODEL_TYPE
    role = "acceptance"
    probability_target = "acceptance"
    source_format = "monotone_spline_xgb_npz"
    u_cols = ("U",)

    def __init__(
        self,
        artifact: MonotoneSplineArtifactData,
        base_acceptance: Any,
        *,
        artifact_path: str | Path | None = None,
        id_col: str = "id",
    ) -> None:
        self.artifact = artifact
        self.base_acceptance = base_acceptance
        self.artifact_path = str(artifact_path) if artifact_path is not None else None
        self.id_col = id_col
        self.auxiliary_state_cols = (id_col,)
        self.x_feature_cols = tuple(base_acceptance.x_feature_cols)
        self.preprocessor = base_acceptance.preprocessor
        base_path = getattr(base_acceptance, "artifact_path", None)
        if artifact.base_artifact_sha256 and base_path is not None:
            actual_sha256 = _sha256_file(base_path)
            if actual_sha256 != artifact.base_artifact_sha256:
                raise ValueError(
                    "Monotone curve cache was not derived from the configured XGB "
                    "acceptance artifact."
                )
        self._policy_index = dict(zip(artifact.policy_ids.tolist(), range(artifact.policy_ids.size)))
        self._curves = tuple(PPoly(c, artifact.action_grid) for c in artifact.coefficients)
        self._derivatives = tuple(curve.derivative() for curve in self._curves)

    def covered_policy_ids(self) -> tuple[str, ...]:
        return tuple(self.artifact.policy_ids.tolist())

    def covered_row_indices(self) -> np.ndarray:
        return self.artifact.row_indices.copy()

    def policy_feature_dim(self) -> int:
        return self.base_acceptance.policy_feature_dim()

    def predict_acceptance(self, raw_frame: pd.DataFrame, u: np.ndarray) -> np.ndarray:
        frame, actions, policy_indices = self._inputs(raw_frame, u)
        acceptance = np.empty(actions.size, dtype=float)
        cached = policy_indices >= 0
        if np.any(cached):
            churn, _ = self._cached_churn(policy_indices[cached], actions[cached], derivative=False)
            acceptance[cached] = 1.0 - churn
        if np.any(~cached):
            acceptance[~cached] = self._raw_acceptance(frame.loc[~cached], actions[~cached])
        return np.clip(acceptance, 0.0, 1.0)

    def d_acceptance_du(self, raw_frame: pd.DataFrame, u: np.ndarray) -> np.ndarray:
        frame, actions, policy_indices = self._inputs(raw_frame, u)
        derivative = np.empty(actions.size, dtype=float)
        cached = policy_indices >= 0
        if np.any(cached):
            _, d_churn = self._cached_churn(
                policy_indices[cached], actions[cached], derivative=True
            )
            derivative[cached] = -d_churn
        if np.any(~cached):
            epsilon = 1e-3
            derivative[~cached] = (
                self._raw_acceptance(frame.loc[~cached], actions[~cached] + epsilon)
                - self._raw_acceptance(frame.loc[~cached], actions[~cached] - epsilon)
            ) / (2.0 * epsilon)
        return derivative

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if "U" not in frame.columns:
            raise KeyError("Monotone-spline XGB prediction requires column 'U'.")
        acceptance = self.predict_acceptance(frame, frame["U"].to_numpy(dtype=float))
        return np.column_stack([1.0 - acceptance, acceptance])

    def _inputs(
        self, raw_frame: pd.DataFrame, u: np.ndarray
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        if not isinstance(raw_frame, pd.DataFrame):
            raise TypeError("Monotone-spline XGB acceptance requires a DataFrame batch.")
        if self.id_col not in raw_frame.columns:
            raise KeyError(f"Missing policy-ID column '{self.id_col}'.")
        actions = np.asarray(u, dtype=float)
        if actions.shape != (raw_frame.shape[0],) or not np.isfinite(actions).all():
            raise ValueError("u must contain one finite action per input row.")
        ids = raw_frame[self.id_col].astype("string").to_numpy(dtype=str)
        indices = np.asarray([self._policy_index.get(value, -1) for value in ids], dtype=int)
        return raw_frame.reset_index(drop=True), actions, indices

    def _raw_acceptance(self, frame: pd.DataFrame, actions: np.ndarray) -> np.ndarray:
        raw = frame.copy().reset_index(drop=True)
        raw["U"] = actions
        return np.asarray(
            self.base_acceptance.model.predict_proba(
                self.base_acceptance.model_frame(raw)
            )[:, 1],
            dtype=float,
        )

    def _cached_churn(
        self, indices: np.ndarray, actions: np.ndarray, *, derivative: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        churn = np.empty(actions.size, dtype=float)
        d_churn = np.empty(actions.size, dtype=float) if derivative else None
        for index in np.unique(indices):
            rows = indices == index
            values = actions[rows]
            below = values < self.artifact.u_min
            above = values > self.artifact.u_max
            inside = ~(below | above)
            result = np.empty(values.size, dtype=float)
            result[below] = self.artifact.churn_min[index]
            result[inside] = np.clip(self._curves[index](values[inside]), 0.0, 1.0)
            raw_upper = self.artifact.churn_max[index] + self.artifact.upper_slopes[index] * (
                values[above] - self.artifact.u_max
            )
            result[above] = np.clip(raw_upper, 0.0, 1.0)
            churn[rows] = result
            if d_churn is not None:
                d_result = np.zeros(values.size, dtype=float)
                raw_inside = self._curves[index](values[inside])
                d_result[inside] = np.where(
                    (raw_inside > 0.0) & (raw_inside < 1.0),
                    self._derivatives[index](values[inside]),
                    0.0,
                )
                d_result[above] = np.where(
                    (raw_upper > 0.0) & (raw_upper < 1.0),
                    self.artifact.upper_slopes[index],
                    0.0,
                )
                d_churn[rows] = d_result
        return churn, d_churn


def save_monotone_spline_artifact(
    artifact: MonotoneSplineArtifactData,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(ARTIFACT_SCHEMA_VERSION),
        smoother_name=np.asarray(SMOOTHER_NAME),
        model_type=np.asarray(MODEL_TYPE),
        probability_target=np.asarray("acceptance"),
        policy_ids=artifact.policy_ids,
        row_indices=artifact.row_indices,
        action_grid=artifact.action_grid,
        coefficients=artifact.coefficients,
        churn_min=artifact.churn_min,
        churn_max=artifact.churn_max,
        upper_slopes=artifact.upper_slopes,
        base_artifact_sha256=np.asarray(artifact.base_artifact_sha256),
        source_fold=np.asarray(artifact.source_fold),
    )
    return output


def load_monotone_spline_artifact(path: str | Path) -> MonotoneSplineArtifactData:
    with np.load(path, allow_pickle=False) as loaded:
        if int(loaded["schema_version"]) != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Unsupported monotone-spline artifact schema version.")
        if str(loaded["smoother_name"]) != SMOOTHER_NAME:
            raise ValueError("Artifact does not contain monotone smoothing splines.")
        if str(loaded["model_type"]) != MODEL_TYPE:
            raise ValueError("Artifact has an unexpected model type.")
        if str(loaded["probability_target"]) != "acceptance":
            raise ValueError("Monotone-spline artifact must target acceptance.")
        return MonotoneSplineArtifactData(
            policy_ids=loaded["policy_ids"].copy(),
            row_indices=loaded["row_indices"].copy(),
            action_grid=loaded["action_grid"].copy(),
            coefficients=loaded["coefficients"].copy(),
            churn_min=loaded["churn_min"].copy(),
            churn_max=loaded["churn_max"].copy(),
            upper_slopes=loaded["upper_slopes"].copy(),
            base_artifact_sha256=str(loaded["base_artifact_sha256"]),
            source_fold=int(loaded["source_fold"]),
        )


def load_monotone_spline_xgb_acceptance(
    path: str | Path, base_acceptance: Any
) -> MonotoneSplineXGBAcceptance:
    return MonotoneSplineXGBAcceptance(
        load_monotone_spline_artifact(path),
        base_acceptance,
        artifact_path=path,
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "MODEL_TYPE",
    "SMOOTHER_NAME",
    "MonotoneSplineArtifactData",
    "MonotoneSplineXGBAcceptance",
    "fit_monotone_spline_artifact",
    "load_monotone_spline_artifact",
    "load_monotone_spline_xgb_acceptance",
    "save_monotone_spline_artifact",
]
