"""Portable per-policy monotone PCHIP acceptance artifacts.

The runtime seam in this module is array-only.  Pickle compatibility exists
solely for the explicit, trusted source-conversion path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import PPoly

from data.feature_processor import FeatureProcessor


ARTIFACT_SCHEMA_VERSION = 1
SMOOTHER_NAME = "monotone_smoothing_spline"
MODEL_TYPE = "xgb_monotone_spline_20260728"
_MONOTONICITY_TOLERANCE = 1e-10
_PROBABILITY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class XGBMonotoneSplineArtifactData:
    """Portable coefficients defining one monotone churn curve per policy."""

    policy_ids: np.ndarray
    row_indices: np.ndarray
    action_grid: np.ndarray
    coefficients: np.ndarray
    churn_min: np.ndarray
    churn_max: np.ndarray
    upper_slopes: np.ndarray
    source_sha256: str = ""
    embedded_booster_sha256: str = ""
    base_artifact_sha256: str = ""
    base_preprocessor_sha256: str = ""

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
        expected_coefficients = (n_policies, 4, action_grid.size - 1)
        if coefficients.shape != expected_coefficients:
            raise ValueError(
                "coefficients must contain cubic PCHIP coefficients with shape "
                f"{expected_coefficients}."
            )
        if not np.isfinite(coefficients).all():
            raise ValueError("coefficients must be finite.")
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
            raise ValueError("upper_slopes must be non-negative for monotone churn.")

        _validate_curve_values(
            action_grid,
            coefficients,
            churn_min,
            churn_max,
        )

        object.__setattr__(self, "policy_ids", policy_ids)
        object.__setattr__(self, "row_indices", row_indices)
        object.__setattr__(self, "action_grid", action_grid)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "churn_min", churn_min)
        object.__setattr__(self, "churn_max", churn_max)
        object.__setattr__(self, "upper_slopes", np.maximum(upper_slopes, 0.0))
        for name in (
            "source_sha256",
            "embedded_booster_sha256",
            "base_artifact_sha256",
            "base_preprocessor_sha256",
        ):
            object.__setattr__(self, name, str(getattr(self, name)))

    @property
    def u_min(self) -> float:
        return float(self.action_grid[0])

    @property
    def u_max(self) -> float:
        return float(self.action_grid[-1])


def _validation_grid(action_grid: np.ndarray) -> np.ndarray:
    """Sample knots and interval interiors for persisted-curve validation."""
    fractions = np.asarray([0.25, 0.5, 0.75])
    left = action_grid[:-1, None]
    widths = np.diff(action_grid)[:, None]
    interior = (left + widths * fractions).reshape(-1)
    return np.sort(np.concatenate([action_grid, interior]))


def _validate_curve_values(
    action_grid: np.ndarray,
    coefficients: np.ndarray,
    churn_min: np.ndarray,
    churn_max: np.ndarray,
) -> None:
    validation_grid = _validation_grid(action_grid)
    for index, policy_coefficients in enumerate(coefficients):
        curve = PPoly(policy_coefficients, action_grid, extrapolate=False)
        values = np.asarray(curve(validation_grid), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Policy curve {index} produces non-finite churn values.")
        if np.any(values < -_PROBABILITY_TOLERANCE) or np.any(
            values > 1.0 + _PROBABILITY_TOLERANCE
        ):
            raise ValueError(f"Policy curve {index} leaves the [0, 1] probability range.")
        if np.any(np.diff(values) < -_MONOTONICITY_TOLERANCE):
            raise ValueError(f"Policy curve {index} is not monotone non-decreasing.")
        if not np.isclose(values[0], churn_min[index], rtol=0.0, atol=1e-10):
            raise ValueError(f"Policy curve {index} does not match churn_min.")
        if not np.isclose(values[-1], churn_max[index], rtol=0.0, atol=1e-10):
            raise ValueError(f"Policy curve {index} does not match churn_max.")


class XGBMonotoneSplineAcceptance:
    """Evaluate portable policy-specific monotone acceptance curves."""

    model_type = MODEL_TYPE
    artifact_id = MODEL_TYPE
    role = "acceptance"
    probability_target = "acceptance"
    source_format = "xgb_monotone_spline_npz"
    u_cols = ("U",)

    def __init__(
        self,
        artifact: XGBMonotoneSplineArtifactData,
        *,
        artifact_path: str | Path | None = None,
        id_col: str = "id",
        x_feature_cols: Sequence[str] = (),
        preprocessor: FeatureProcessor | None = None,
    ) -> None:
        self.artifact = artifact
        self.artifact_path = str(artifact_path) if artifact_path is not None else None
        self.id_col = str(id_col)
        self.auxiliary_state_cols = (self.id_col,)
        self.x_feature_cols = tuple(str(col) for col in x_feature_cols)
        self.preprocessor = preprocessor
        self._policy_index = {
            policy_id: index for index, policy_id in enumerate(artifact.policy_ids.tolist())
        }
        self._curves = tuple(
            PPoly(coefficients, artifact.action_grid, extrapolate=False)
            for coefficients in artifact.coefficients
        )
        self._derivative_curves = tuple(curve.derivative() for curve in self._curves)

    def covered_policy_ids(self) -> tuple[str, ...]:
        return tuple(self.artifact.policy_ids.tolist())

    def covered_row_indices(self) -> np.ndarray:
        return self.artifact.row_indices.copy()

    def policy_feature_dim(self) -> int:
        if self.preprocessor is None:
            return len(self.x_feature_cols)
        return len(getattr(self.preprocessor, "output_feature_names_", ()))

    def predict_acceptance(self, raw_frame: pd.DataFrame, u: np.ndarray) -> np.ndarray:
        policy_indices, actions = self._inputs(raw_frame, u)
        churn, _ = self._churn_and_derivative(policy_indices, actions, derivative=False)
        return 1.0 - churn

    def d_acceptance_du(self, raw_frame: pd.DataFrame, u: np.ndarray) -> np.ndarray:
        policy_indices, actions = self._inputs(raw_frame, u)
        _, derivative = self._churn_and_derivative(
            policy_indices,
            actions,
            derivative=True,
        )
        if derivative is None:  # pragma: no cover - derivative=True guarantees it
            raise RuntimeError("Monotone spline derivative was not computed.")
        return -derivative

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if "U" not in frame.columns:
            raise KeyError("XGB monotone-spline prediction requires column 'U'.")
        acceptance = self.predict_acceptance(frame, frame["U"].to_numpy(dtype=float))
        return np.column_stack([1.0 - acceptance, acceptance])

    def _inputs(
        self,
        raw_frame: pd.DataFrame,
        u: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(raw_frame, pd.DataFrame):
            raise TypeError("XGB monotone-spline acceptance requires a DataFrame batch.")
        if self.id_col not in raw_frame.columns:
            raise KeyError(f"Missing monotone-spline policy-ID column '{self.id_col}'.")
        policy_ids = raw_frame[self.id_col].astype("string").to_numpy(dtype=str)
        unknown = sorted(
            {policy_id for policy_id in policy_ids if policy_id not in self._policy_index}
        )
        if unknown:
            raise ValueError(f"No fitted monotone spline for policy IDs: {unknown[:5]}")
        actions = np.asarray(u, dtype=float)
        if actions.shape != (raw_frame.shape[0],):
            raise ValueError("u must contain one action per monotone-spline input row.")
        if not np.isfinite(actions).all():
            raise ValueError("Monotone-spline actions must be finite.")
        return (
            np.asarray([self._policy_index[policy_id] for policy_id in policy_ids], dtype=int),
            actions,
        )

    def _churn_and_derivative(
        self,
        policy_indices: np.ndarray,
        actions: np.ndarray,
        *,
        derivative: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        churn = np.empty_like(actions)
        d_churn_du = np.empty_like(actions) if derivative else None
        for policy_index in np.unique(policy_indices):
            rows = policy_indices == policy_index
            policy_actions = actions[rows]
            below = policy_actions < self.artifact.u_min
            above = policy_actions > self.artifact.u_max
            inside = ~(below | above)

            policy_churn = np.empty_like(policy_actions)
            policy_churn[below] = self.artifact.churn_min[policy_index]
            if np.any(inside):
                policy_churn[inside] = np.clip(
                    self._curves[policy_index](policy_actions[inside]),
                    0.0,
                    1.0,
                )
            raw_upper = (
                self.artifact.churn_max[policy_index]
                + self.artifact.upper_slopes[policy_index]
                * (policy_actions[above] - self.artifact.u_max)
            )
            if np.any(above):
                policy_churn[above] = np.clip(raw_upper, 0.0, 1.0)
            churn[rows] = policy_churn

            if d_churn_du is None:
                continue
            policy_derivative = np.zeros_like(policy_actions)
            if np.any(inside):
                raw_inside = self._curves[policy_index](policy_actions[inside])
                derivative_inside = self._derivative_curves[policy_index](
                    policy_actions[inside]
                )
                policy_derivative[inside] = np.where(
                    (raw_inside > 0.0) & (raw_inside < 1.0),
                    derivative_inside,
                    0.0,
                )
            if np.any(above):
                policy_derivative[above] = np.where(
                    (raw_upper > 0.0) & (raw_upper < 1.0),
                    self.artifact.upper_slopes[policy_index],
                    0.0,
                )
            d_churn_du[rows] = policy_derivative
        return churn, d_churn_du


class _LegacyMonotoneSplineCurve:
    """Compatibility target for the trusted source artifact's curve class."""

    def __call__(self, values: np.ndarray) -> np.ndarray:
        actions = np.asarray(values, dtype=float)
        churn = np.empty_like(actions)
        below = actions < self.x_min
        above = actions > self.x_max
        inside = ~(below | above)
        churn[below] = self.p_min
        # SciPy 1.17 renamed PPoly's serialized ``x``/``c`` slots to
        # ``_x``/``_c`` without preserving the old object's helper state.
        # Reconstructing a current PPoly keeps conversion independent of that
        # private compatibility break.
        interpolator = PPoly(
            np.asarray(object.__getattribute__(self.interp, "_c"), dtype=float),
            np.asarray(object.__getattribute__(self.interp, "_x"), dtype=float),
            extrapolate=False,
        )
        churn[inside] = interpolator(actions[inside])
        churn[above] = self.p_max + self.slope_p * (actions[above] - self.x_max)
        return np.clip(churn, 0.0, 1.0)


class _LegacySmoothedXGBoostWrapper:
    """Compatibility target exposing conversion-only wrapper state."""


class _LegacyArtifactUnpickler(pickle.Unpickler):
    """Map trusted source-repository classes onto local compatibility classes."""

    def find_class(self, module: str, name: str) -> Any:
        if name == "FeatureProcessor" and module in {"__main__", "preprocessing"}:
            return FeatureProcessor
        if name == "SmoothedXGBoostWrapper" and module in {
            "__main__",
            "black_box_objective",
            "smoothing_wrapper",
        }:
            return _LegacySmoothedXGBoostWrapper
        if name == "_MonotoneSplineCurve" and module.endswith("curve_specifications"):
            return _LegacyMonotoneSplineCurve
        return super().find_class(module, name)


def _load_trusted_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as handle:
        return _LegacyArtifactUnpickler(handle).load()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pickle_sha256(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def _booster_sha256(model: Any) -> str:
    if model is None or not hasattr(model, "get_booster"):
        raise ValueError("Expected an embedded XGBoost classifier.")
    return hashlib.sha256(bytes(model.get_booster().save_raw())).hexdigest()


def load_legacy_monotone_wrapper(path: str | Path) -> Any:
    """Load a trusted monotone wrapper for explicit conversion only."""
    wrapper = _load_trusted_pickle(path)
    if getattr(wrapper, "_function_name", None) != SMOOTHER_NAME:
        raise ValueError(f"Expected source smoother '{SMOOTHER_NAME}'.")
    curves = getattr(wrapper, "_curves", None)
    if not isinstance(curves, Mapping) or not curves:
        raise ValueError("Monotone source wrapper must contain fitted curves.")
    return wrapper


def canonical_row_indices_for_policy_ids(
    dataset_path: str | Path,
    policy_ids: Sequence[str],
    *,
    id_col: str = "id",
) -> np.ndarray:
    """Resolve each policy ID to one canonical CSV row, preserving order."""
    requested = np.asarray(policy_ids, dtype=str)
    frame = pd.read_csv(
        dataset_path,
        sep=";",
        usecols=[id_col],
        dtype={id_col: "string"},
    )
    positions: dict[str, list[int]] = {}
    requested_set = set(requested.tolist())
    for position, value in enumerate(frame[id_col]):
        if pd.isna(value):
            continue
        policy_id = str(value)
        if policy_id in requested_set:
            positions.setdefault(policy_id, []).append(position)
    missing = [policy_id for policy_id in requested if policy_id not in positions]
    duplicates = [
        policy_id for policy_id in requested if len(positions.get(policy_id, ())) != 1
    ]
    if missing or duplicates:
        raise ValueError(
            "Monotone policy IDs must resolve uniquely in the canonical dataset: "
            f"missing={missing[:5]}, duplicates={duplicates[:5]}"
        )
    return np.asarray([positions[policy_id][0] for policy_id in requested], dtype=int)


def prepare_xgb_monotone_spline_artifact(
    source_path: str | Path,
    dataset_path: str | Path,
    base_acceptance_path: str | Path,
) -> XGBMonotoneSplineArtifactData:
    """Convert the trusted wrapper into validated, portable PCHIP arrays."""
    wrapper = load_legacy_monotone_wrapper(source_path)
    base_artifact = _load_trusted_pickle(base_acceptance_path)
    if not isinstance(base_artifact, Mapping) or "model" not in base_artifact:
        raise ValueError("Base acceptance artifact must contain a selected model.")

    embedded_booster_sha256 = _booster_sha256(getattr(wrapper, "_model", None))
    base_booster_sha256 = _booster_sha256(base_artifact["model"])
    if embedded_booster_sha256 != base_booster_sha256:
        raise ValueError("Embedded XGBoost booster does not match the canonical base model.")
    source_preprocessor = getattr(wrapper, "_prep", None)
    base_preprocessor = base_artifact.get("preprocessor")
    source_preprocessor_sha256 = _pickle_sha256(source_preprocessor)
    base_preprocessor_sha256 = _pickle_sha256(base_preprocessor)
    if source_preprocessor_sha256 != base_preprocessor_sha256:
        raise ValueError("Embedded preprocessor does not match the canonical base model.")

    normalized = {str(policy_id): curve for policy_id, curve in wrapper._curves.items()}
    if len(normalized) != len(wrapper._curves):
        raise ValueError("Policy IDs must remain unique after string normalization.")
    policy_ids = np.asarray(sorted(normalized), dtype=str)
    row_indices = canonical_row_indices_for_policy_ids(dataset_path, policy_ids)

    first_curve = normalized[policy_ids[0]]
    action_grid = np.asarray(
        object.__getattribute__(first_curve.interp, "_x"),
        dtype=float,
    )
    coefficients = []
    churn_min = []
    churn_max = []
    upper_slopes = []
    for policy_id in policy_ids:
        curve = normalized[policy_id]
        curve_grid = np.asarray(
            object.__getattribute__(curve.interp, "_x"),
            dtype=float,
        )
        if not np.array_equal(curve_grid, action_grid):
            raise ValueError("All monotone curves must share one exact action grid.")
        coefficients.append(
            np.asarray(object.__getattribute__(curve.interp, "_c"), dtype=float)
        )
        churn_min.append(float(curve.p_min))
        churn_max.append(float(curve.p_max))
        upper_slopes.append(float(curve.slope_p))

    return XGBMonotoneSplineArtifactData(
        policy_ids=policy_ids,
        row_indices=row_indices,
        action_grid=action_grid,
        coefficients=np.stack(coefficients),
        churn_min=np.asarray(churn_min),
        churn_max=np.asarray(churn_max),
        upper_slopes=np.asarray(upper_slopes),
        source_sha256=_sha256_file(source_path),
        embedded_booster_sha256=embedded_booster_sha256,
        base_artifact_sha256=_sha256_file(base_acceptance_path),
        base_preprocessor_sha256=base_preprocessor_sha256,
    )


def save_xgb_monotone_spline_artifact(
    artifact: XGBMonotoneSplineArtifactData,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Persist portable arrays without pickle-backed object arrays."""
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=int),
        smoother_name=np.asarray(SMOOTHER_NAME),
        probability_target=np.asarray("acceptance"),
        model_type=np.asarray(MODEL_TYPE),
        policy_ids=artifact.policy_ids,
        row_indices=artifact.row_indices,
        action_grid=artifact.action_grid,
        coefficients=artifact.coefficients,
        churn_min=artifact.churn_min,
        churn_max=artifact.churn_max,
        upper_slopes=artifact.upper_slopes,
        source_sha256=np.asarray(artifact.source_sha256),
        embedded_booster_sha256=np.asarray(artifact.embedded_booster_sha256),
        base_artifact_sha256=np.asarray(artifact.base_artifact_sha256),
        base_preprocessor_sha256=np.asarray(artifact.base_preprocessor_sha256),
    )
    return output


def load_xgb_monotone_spline_artifact(
    path: str | Path,
) -> XGBMonotoneSplineArtifactData:
    """Load and validate a portable monotone-spline artifact."""
    with np.load(path, allow_pickle=False) as loaded:
        if int(loaded["schema_version"]) != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Unsupported XGB monotone-spline artifact schema version.")
        if str(loaded["smoother_name"]) != SMOOTHER_NAME:
            raise ValueError("Artifact does not contain monotone smoothing splines.")
        if str(loaded["probability_target"]) != "acceptance":
            raise ValueError("Monotone-spline artifact must target acceptance.")
        if str(loaded["model_type"]) != MODEL_TYPE:
            raise ValueError("Monotone-spline artifact has an unexpected model type.")
        return XGBMonotoneSplineArtifactData(
            policy_ids=loaded["policy_ids"].copy(),
            row_indices=loaded["row_indices"].copy(),
            action_grid=loaded["action_grid"].copy(),
            coefficients=loaded["coefficients"].copy(),
            churn_min=loaded["churn_min"].copy(),
            churn_max=loaded["churn_max"].copy(),
            upper_slopes=loaded["upper_slopes"].copy(),
            source_sha256=str(loaded["source_sha256"]),
            embedded_booster_sha256=str(loaded["embedded_booster_sha256"]),
            base_artifact_sha256=str(loaded["base_artifact_sha256"]),
            base_preprocessor_sha256=str(loaded["base_preprocessor_sha256"]),
        )


def load_xgb_monotone_spline_acceptance(
    path: str | Path,
    **kwargs: Any,
) -> XGBMonotoneSplineAcceptance:
    """Load the portable artifact and construct its runtime acceptance model."""
    return XGBMonotoneSplineAcceptance(
        load_xgb_monotone_spline_artifact(path),
        artifact_path=path,
        **kwargs,
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "MODEL_TYPE",
    "SMOOTHER_NAME",
    "XGBMonotoneSplineAcceptance",
    "XGBMonotoneSplineArtifactData",
    "canonical_row_indices_for_policy_ids",
    "load_legacy_monotone_wrapper",
    "load_xgb_monotone_spline_acceptance",
    "load_xgb_monotone_spline_artifact",
    "prepare_xgb_monotone_spline_artifact",
    "save_xgb_monotone_spline_artifact",
]
