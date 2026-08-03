"""Portable per-policy shifted-sigmoid acceptance artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.special import expit

from data.feature_processor import FeatureProcessor


ARTIFACT_SCHEMA_VERSION = 1
SMOOTHER_NAME = "shifted_sigmoid"


@dataclass(frozen=True)
class XGBSigmoidArtifactData:
    """Portable parameters defining one shifted-sigmoid churn curve per policy."""

    policy_ids: np.ndarray
    row_indices: np.ndarray
    parameters: np.ndarray
    action_min: float = 0.0
    action_max: float = 0.16
    source_sha256: str = ""
    embedded_booster_sha256: str = ""

    def __post_init__(self) -> None:
        policy_ids = np.asarray(self.policy_ids, dtype=str)
        row_indices = np.asarray(self.row_indices, dtype=int)
        parameters = np.asarray(self.parameters, dtype=float)
        n_policies = policy_ids.size
        if policy_ids.ndim != 1 or n_policies == 0:
            raise ValueError("policy_ids must be a non-empty 1D array.")
        if np.unique(policy_ids).size != n_policies:
            raise ValueError("policy_ids must be unique.")
        if row_indices.shape != (n_policies,) or np.unique(row_indices).size != n_policies:
            raise ValueError("row_indices must contain one unique row per policy.")
        if parameters.shape != (n_policies, 3) or not np.isfinite(parameters).all():
            raise ValueError("parameters must be finite with shape (n_policies, 3).")
        if not np.isfinite([self.action_min, self.action_max]).all():
            raise ValueError("Action support must be finite.")
        if float(self.action_min) >= float(self.action_max):
            raise ValueError("action_min must be smaller than action_max.")
        object.__setattr__(self, "policy_ids", policy_ids)
        object.__setattr__(self, "row_indices", row_indices)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "action_min", float(self.action_min))
        object.__setattr__(self, "action_max", float(self.action_max))
        object.__setattr__(self, "source_sha256", str(self.source_sha256))
        object.__setattr__(
            self, "embedded_booster_sha256", str(self.embedded_booster_sha256)
        )


class XGBSigmoidAcceptance:
    """Evaluate portable per-policy shifted-sigmoid acceptance and derivatives."""

    model_type = "xgb_sigmoid_20260728"
    artifact_id = "xgb_sigmoid_20260728"
    role = "acceptance"
    probability_target = "acceptance"
    source_format = "xgb_sigmoid_npz"
    u_cols = ("U",)

    def __init__(
        self,
        artifact: XGBSigmoidArtifactData,
        *,
        artifact_path: str | Path | None = None,
        id_col: str = "id",
        x_feature_cols: tuple[str, ...] = (),
        preprocessor: FeatureProcessor | None = None,
    ) -> None:
        self.artifact = artifact
        self.artifact_path = str(artifact_path) if artifact_path is not None else None
        self.id_col = str(id_col)
        self.auxiliary_state_cols = (self.id_col,)
        self.x_feature_cols = tuple(x_feature_cols)
        self.preprocessor = preprocessor
        self._id_to_index = {
            policy_id: index
            for index, policy_id in enumerate(self.artifact.policy_ids.tolist())
        }

    def covered_policy_ids(self) -> tuple[str, ...]:
        """Return the policy IDs that have fitted curves."""
        return tuple(self.artifact.policy_ids.tolist())

    def covered_row_indices(self) -> np.ndarray:
        """Return canonical CSV row positions covered by the artifact."""
        return self.artifact.row_indices.copy()

    def policy_feature_dim(self) -> int:
        """Return the acceptance-preprocessed state dimension."""
        if self.preprocessor is None:
            return len(self.x_feature_cols)
        return len(getattr(self.preprocessor, "output_feature_names_", ()))

    def predict_acceptance(
        self, raw_frame: pd.DataFrame, u: np.ndarray
    ) -> np.ndarray:
        """Return one acceptance probability per policy/action row."""
        indices = self._curve_indices(raw_frame)
        u_arr = self._validate_u(u, indices.size)
        k, m, d = self.artifact.parameters[indices].T
        churn = np.clip(d + expit(k * (u_arr - m)), 0.0, 1.0)
        return 1.0 - churn

    def d_acceptance_du(
        self, raw_frame: pd.DataFrame, u: np.ndarray
    ) -> np.ndarray:
        """Return the analytical action derivative of acceptance."""
        indices = self._curve_indices(raw_frame)
        u_arr = self._validate_u(u, indices.size)
        k, m, d = self.artifact.parameters[indices].T
        sigmoid = expit(k * (u_arr - m))
        raw_churn = d + sigmoid
        derivative = -k * sigmoid * (1.0 - sigmoid)
        return np.where((raw_churn > 0.0) & (raw_churn < 1.0), derivative, 0.0)

    def _curve_indices(self, raw_frame: pd.DataFrame) -> np.ndarray:
        if not isinstance(raw_frame, pd.DataFrame):
            raise ValueError("Shifted-sigmoid acceptance requires a pandas DataFrame.")
        if self.id_col not in raw_frame.columns:
            raise ValueError(
                f"Shifted-sigmoid acceptance requires policy ID column '{self.id_col}'."
            )
        policy_ids = raw_frame[self.id_col].astype("string").astype(str).to_numpy()
        missing = sorted(
            {policy_id for policy_id in policy_ids if policy_id not in self._id_to_index}
        )
        if missing:
            raise ValueError(
                "No fitted acceptance sigmoid for policy IDs: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )
        return np.asarray([self._id_to_index[policy_id] for policy_id in policy_ids], dtype=int)

    @staticmethod
    def _validate_u(u: np.ndarray, n_rows: int) -> np.ndarray:
        u_arr = np.asarray(u, dtype=float)
        if u_arr.shape != (n_rows,):
            raise ValueError("u must contain one finite action per row.")
        if not np.isfinite(u_arr).all():
            raise ValueError("u must contain only finite actions.")
        return u_arr


class _LegacyParametricCurve:
    """Compatibility target for trusted notebook-local parametric curves."""

    def __call__(self, values: np.ndarray) -> np.ndarray:
        return self.func(np.asarray(values, dtype=float), *self.params)


class _LegacySmoothedXGBoostWrapper:
    """Compatibility target exposing state needed for deterministic conversion."""


def _legacy_sigmoid_with_shift(
    x: np.ndarray, k: float, m: float, d: float
) -> np.ndarray:
    return np.clip(d + expit(k * (np.asarray(x, dtype=float) - m)), 0.0, 1.0)


class _LegacyArtifactUnpickler(pickle.Unpickler):
    """Map source-repository classes onto conversion-only compatibility targets."""

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_legacy_sigmoid_wrapper(path: str | Path) -> Any:
    """Load and validate the trusted legacy sigmoid wrapper for conversion only."""
    source = Path(path)
    with source.open("rb") as handle:
        wrapper = _LegacyArtifactUnpickler(handle).load()
    if getattr(wrapper, "_function_name", None) != "sigmoid_with_shift":
        raise ValueError("Only sigmoid_with_shift smoothing artifacts are supported.")
    curves = getattr(wrapper, "_curves", None)
    if not isinstance(curves, Mapping) or not curves:
        raise ValueError("Smoothing artifact must contain a non-empty _curves mapping.")
    return wrapper


def extract_sigmoid_parameters(wrapper: Any) -> tuple[np.ndarray, np.ndarray]:
    """Extract sorted policy IDs and finite ``(k, m, d)`` curve parameters."""
    if getattr(wrapper, "_function_name", None) != "sigmoid_with_shift":
        raise ValueError("Only sigmoid_with_shift smoothing artifacts are supported.")
    curves = getattr(wrapper, "_curves", None)
    if not isinstance(curves, Mapping) or not curves:
        raise ValueError("Smoothing artifact must contain a non-empty _curves mapping.")
    normalized = {str(policy_id): curve for policy_id, curve in curves.items()}
    if len(normalized) != len(curves):
        raise ValueError("Policy IDs must remain unique after string normalization.")
    policy_ids = np.asarray(sorted(normalized), dtype=str)
    parameters: list[np.ndarray] = []
    for policy_id in policy_ids:
        values = np.asarray(getattr(normalized[policy_id], "params", ()), dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"Policy {policy_id} has invalid sigmoid parameters.")
        parameters.append(values)
    return policy_ids, np.stack(parameters)


def canonical_row_indices_for_policy_ids(
    dataset_path: str | Path, policy_ids: np.ndarray, *, id_col: str = "id"
) -> np.ndarray:
    """Resolve every requested policy ID to exactly one canonical CSV row."""
    requested = np.asarray(policy_ids, dtype=str)
    frame = pd.read_csv(dataset_path, sep=";", usecols=[id_col], dtype={id_col: "string"})
    positions: dict[str, list[int]] = {}
    requested_set = set(requested.tolist())
    for position, policy_id in enumerate(frame[id_col].astype(str)):
        if policy_id in requested_set:
            positions.setdefault(policy_id, []).append(position)
    missing = [policy_id for policy_id in requested if policy_id not in positions]
    duplicates = [
        policy_id for policy_id in requested if len(positions.get(policy_id, ())) != 1
    ]
    if missing or duplicates:
        raise ValueError(
            "Sigmoid policy IDs must resolve uniquely in the canonical dataset: "
            f"missing={missing[:5]}, duplicates={duplicates[:5]}"
        )
    return np.asarray([positions[policy_id][0] for policy_id in requested], dtype=int)


def prepare_xgb_sigmoid_artifact(
    source_path: str | Path,
    dataset_path: str | Path,
    *,
    action_min: float = 0.0,
    action_max: float = 0.16,
) -> XGBSigmoidArtifactData:
    """Convert the trusted legacy wrapper into validated portable arrays."""
    source = Path(source_path)
    wrapper = load_legacy_sigmoid_wrapper(source)
    policy_ids, parameters = extract_sigmoid_parameters(wrapper)
    row_indices = canonical_row_indices_for_policy_ids(dataset_path, policy_ids)
    model = getattr(wrapper, "_model", None)
    if model is None or not hasattr(model, "get_booster"):
        raise ValueError("Legacy smoothing artifact must embed an XGBoost model.")
    booster_bytes = bytes(model.get_booster().save_raw())
    return XGBSigmoidArtifactData(
        policy_ids=policy_ids,
        row_indices=row_indices,
        parameters=parameters,
        action_min=action_min,
        action_max=action_max,
        source_sha256=_sha256_file(source),
        embedded_booster_sha256=hashlib.sha256(booster_bytes).hexdigest(),
    )


def save_xgb_sigmoid_artifact(
    artifact: XGBSigmoidArtifactData,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Save a portable shifted-sigmoid artifact without pickle object arrays."""
    output = Path(path)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        schema_version=np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=int),
        smoother_name=np.asarray(SMOOTHER_NAME),
        policy_ids=artifact.policy_ids,
        row_indices=artifact.row_indices,
        parameters=artifact.parameters,
        action_min=np.asarray(artifact.action_min, dtype=float),
        action_max=np.asarray(artifact.action_max, dtype=float),
        source_sha256=np.asarray(artifact.source_sha256),
        embedded_booster_sha256=np.asarray(artifact.embedded_booster_sha256),
    )
    return output


def load_xgb_sigmoid_artifact(path: str | Path) -> XGBSigmoidArtifactData:
    """Load and validate a portable shifted-sigmoid artifact."""
    with np.load(path, allow_pickle=False) as loaded:
        if int(loaded["schema_version"]) != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Unsupported XGB sigmoid artifact schema version.")
        if str(loaded["smoother_name"]) != SMOOTHER_NAME:
            raise ValueError("Unsupported XGB sigmoid smoother.")
        return XGBSigmoidArtifactData(
            policy_ids=loaded["policy_ids"],
            row_indices=loaded["row_indices"],
            parameters=loaded["parameters"],
            action_min=float(loaded["action_min"]),
            action_max=float(loaded["action_max"]),
            source_sha256=str(loaded["source_sha256"]),
            embedded_booster_sha256=str(loaded["embedded_booster_sha256"]),
        )


def load_xgb_sigmoid_acceptance(
    path: str | Path,
    **kwargs: Any,
) -> XGBSigmoidAcceptance:
    """Build the runtime acceptance adapter from a portable NPZ."""
    return XGBSigmoidAcceptance(
        load_xgb_sigmoid_artifact(path),
        artifact_path=path,
        **kwargs,
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "XGBSigmoidAcceptance",
    "XGBSigmoidArtifactData",
    "canonical_row_indices_for_policy_ids",
    "extract_sigmoid_parameters",
    "load_legacy_sigmoid_wrapper",
    "load_xgb_sigmoid_acceptance",
    "load_xgb_sigmoid_artifact",
    "prepare_xgb_sigmoid_artifact",
    "save_xgb_sigmoid_artifact",
]
