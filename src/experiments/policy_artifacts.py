"""Saved policy artifacts for optimizer-independent replay and validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from data.dataset_metadata import DATASET_SCHEMA_VERSION
from data.loader import ModelType, load_model_artifacts, load_observed_loss_array, load_x_frame
from experiments.policy_validation import evaluate_policy, policy_u_values
from experiments.results import ExperimentResult, PolicyEvaluation
from objective.noise import NoisyObjective
from objective.objectives import ModelBasedObjective
from objective.policy import (
    ConstantPolicy,
    CubicFeatureMap,
    FeatureMap,
    IdentityFeatureMap,
    LinearPolicy,
    MLPPolicy,
    QuadraticFeatureMap,
    QuarticFeatureMap,
    SoftmaxPolicy,
)
from objective.policy_preprocessing import PolicyFeaturePreprocessor

SplitName = Literal["train", "test", "all"]
_ARTIFACT_SCHEMA_VERSION = 1
_ARRAY_FILE = "arrays.npz"
_PREPROCESSOR_PREFIX = "policy_preprocessor__"


@dataclass(frozen=True)
class PolicyFeatureMapSpec:
    """Serializable policy feature map $$\varphi(z)$$ applied after preprocessing."""

    type: str
    kind: str | None = None
    include_interactions: bool | None = None
    feature_dim: int | None = None
    name: str | None = None

    @classmethod
    def from_policy(cls, policy: object) -> "PolicyFeatureMapSpec | None":
        feature_map = getattr(policy, "feature_map", None)
        if feature_map is None:
            return None
        return cls(
            type=type(feature_map).__name__,
            kind=getattr(feature_map, "kind", None),
            include_interactions=getattr(feature_map, "include_interactions", None),
            feature_dim=getattr(feature_map, "feature_dim", None),
            name=getattr(feature_map, "name", None),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object] | None) -> "PolicyFeatureMapSpec | None":
        if payload is None:
            return None
        return cls(
            type=str(payload["type"]),
            kind=_optional_str(payload.get("kind")),
            include_interactions=_optional_bool(payload.get("include_interactions")),
            feature_dim=_optional_int(payload.get("feature_dim")),
            name=_optional_str(payload.get("name")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "kind": self.kind,
            "include_interactions": self.include_interactions,
            "feature_dim": self.feature_dim,
            "name": self.name,
        }

    def build(self) -> FeatureMap:
        if self.type == "IdentityFeatureMap":
            return IdentityFeatureMap()
        if self.type == "QuadraticFeatureMap":
            return QuadraticFeatureMap(include_interactions=self.include_interactions is not False)
        if self.type == "CubicFeatureMap":
            return CubicFeatureMap(include_interactions=self.include_interactions is not False)
        if self.type == "QuarticFeatureMap":
            return QuarticFeatureMap(include_interactions=self.include_interactions is not False)
        raise ValueError(f"Unsupported saved policy feature map '{self.type}'.")


@dataclass(frozen=True)
class PolicyHeadSpec:
    """Serializable policy head mapping features and theta to action $$u$$."""

    type: str
    action_low: float | None = None
    action_high: float | None = None
    hidden: int | None = None

    @classmethod
    def from_policy(cls, policy: object) -> "PolicyHeadSpec":
        if isinstance(policy, ConstantPolicy):
            return cls(type="ConstantPolicy")
        if isinstance(policy, LinearPolicy):
            return cls(type="LinearPolicy")
        if isinstance(policy, SoftmaxPolicy):
            return cls(
                type="SoftmaxPolicy",
                action_low=float(policy.action_low),
                action_high=float(policy.action_high),
            )
        if isinstance(policy, MLPPolicy):
            return cls(type="MLPPolicy", hidden=int(policy.hidden))
        raise ValueError(f"Unsupported policy type '{type(policy).__name__}'.")

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PolicyHeadSpec":
        return cls(
            type=str(payload["type"]),
            action_low=_optional_float(payload.get("action_low")),
            action_high=_optional_float(payload.get("action_high")),
            hidden=_optional_int(payload.get("hidden")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "action_low": self.action_low,
            "action_high": self.action_high,
            "hidden": self.hidden,
        }

    def build(self, feature_map: FeatureMap | None) -> object:
        if self.type == "ConstantPolicy":
            return ConstantPolicy()
        resolved_feature_map = feature_map if feature_map is not None else IdentityFeatureMap()
        if self.type == "LinearPolicy":
            return LinearPolicy(feature_map=resolved_feature_map)
        if self.type == "SoftmaxPolicy":
            return SoftmaxPolicy(
                feature_map=resolved_feature_map,
                action_low=float(self.action_low),
                action_high=float(self.action_high),
            )
        if self.type == "MLPPolicy":
            return MLPPolicy(feature_map=resolved_feature_map, hidden=int(self.hidden or 16))
        raise ValueError(f"Unsupported saved policy head '{self.type}'.")


@dataclass(frozen=True)
class PolicyInputPreprocessingSpec:
    r"""Serializable policy-input preprocessing $$x_{raw} \mapsto z$$, before $$\varphi(z)$$."""

    artifact_preprocessing: bool
    policy_side_preprocessing: bool
    policy_feature_cols: tuple[str, ...] | None = None
    policy_preprocessor: PolicyFeaturePreprocessor | None = None

    @classmethod
    def from_objective(cls, objective: ModelBasedObjective) -> "PolicyInputPreprocessingSpec":
        policy_preprocessor = objective.policy_preprocessor
        policy_feature_cols = (
            tuple(objective.policy_feature_cols)
            if objective.policy_feature_cols is not None
            else None
        )
        artifact_preprocessing = not (
            policy_preprocessor is not None and policy_feature_cols is not None
        )
        return cls(
            artifact_preprocessing=artifact_preprocessing,
            policy_side_preprocessing=policy_preprocessor is not None,
            policy_feature_cols=policy_feature_cols,
            policy_preprocessor=policy_preprocessor,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        arrays: Mapping[str, np.ndarray],
    ) -> "PolicyInputPreprocessingSpec":
        artifact_payload = _mapping(payload.get("artifact_preprocessing"))
        policy_side_payload = _mapping(payload.get("policy_side_preprocessing"))
        policy_side_enabled = bool(policy_side_payload.get("enabled", False))
        policy_preprocessor = None
        if policy_side_enabled:
            metadata = _mapping(policy_side_payload.get("state_metadata"))
            state_arrays = _prefixed_arrays(arrays, _PREPROCESSOR_PREFIX)
            policy_preprocessor = PolicyFeaturePreprocessor.from_state(metadata, state_arrays)
        feature_cols = payload.get("policy_feature_cols")
        return cls(
            artifact_preprocessing=bool(artifact_payload.get("enabled", False)),
            policy_side_preprocessing=policy_side_enabled,
            policy_feature_cols=tuple(str(col) for col in feature_cols) if feature_cols is not None else None,
            policy_preprocessor=policy_preprocessor,
        )

    def to_dict(self) -> dict[str, object]:
        transform_order: list[str] = []
        if self.artifact_preprocessing:
            transform_order.append("artifact_preprocessing")
        elif self.policy_feature_cols is not None:
            transform_order.append("raw_policy_feature_cols")
        if self.policy_side_preprocessing:
            transform_order.append("policy_side_preprocessing")
        return {
            "description": "raw x -> policy input preprocessing -> z; feature maps phi/varphi are stored separately",
            "transform_order": transform_order,
            "artifact_preprocessing": {
                "enabled": bool(self.artifact_preprocessing),
                "source": "acceptance_model" if self.artifact_preprocessing else None,
            },
            "policy_side_preprocessing": {
                "enabled": bool(self.policy_side_preprocessing),
                "state_metadata": self.policy_preprocessor.to_dict()
                if self.policy_preprocessor is not None
                else None,
            },
            "policy_feature_cols": list(self.policy_feature_cols)
            if self.policy_feature_cols is not None
            else None,
        }

    def add_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        if self.policy_preprocessor is None:
            return
        state = self.policy_preprocessor.to_state()
        for name, value in _mapping(state["arrays"]).items():
            arrays[f"{_PREPROCESSOR_PREFIX}{name}"] = np.asarray(value, dtype=float)


@dataclass(frozen=True)
class ObjectiveReplaySpec:
    """Serializable metadata needed to rebuild a model-based objective."""

    model_type: ModelType
    acceptance_state_cols: tuple[str, ...]
    loss_cols: tuple[str, ...]
    premium_col: str | int
    loss_source: Literal["predicted", "observed"] = "predicted"
    observed_loss_col: str = "Y_G_Loss"
    u_coef: float | None = None
    u_bounds: tuple[float, float] | None = None
    acceptance_floor: float | None = None
    acceptance_penalty_weight: float | None = None
    acceptance_penalty_temperature: float = 0.01
    lagrangian_lambda: float | None = None
    acceptance_artifact_path: str | None = None
    loss_artifact_path: str | None = None

    @classmethod
    def from_objective(cls, objective: ModelBasedObjective) -> "ObjectiveReplaySpec":
        return cls(
            model_type=_infer_model_type(objective),
            acceptance_state_cols=tuple(objective.acceptance_state_cols),
            loss_cols=tuple(objective.loss_cols),
            premium_col=objective.premium_col,
            loss_source=objective.loss_source,
            observed_loss_col=objective.observed_loss_col,
            u_coef=float(objective.u_coef) if objective.u_coef is not None else None,
            u_bounds=tuple(float(value) for value in objective.u_bounds)
            if objective.u_bounds is not None
            else None,
            acceptance_floor=float(objective.acceptance_floor)
            if objective.acceptance_floor is not None
            else None,
            acceptance_penalty_weight=float(objective.acceptance_penalty_weight)
            if objective.acceptance_penalty_weight is not None
            else None,
            acceptance_penalty_temperature=float(objective.acceptance_penalty_temperature),
            lagrangian_lambda=float(objective.lagrangian_lambda)
            if objective.lagrangian_lambda is not None
            else None,
            acceptance_artifact_path=getattr(objective.acceptance_model, "artifact_path", None),
            loss_artifact_path=getattr(objective.loss_model, "artifact_path", None),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ObjectiveReplaySpec":
        u_bounds = payload.get("u_bounds")
        return cls(
            model_type=str(payload["model_type"]),  # type: ignore[arg-type]
            acceptance_state_cols=tuple(str(col) for col in payload["acceptance_state_cols"]),
            loss_cols=tuple(str(col) for col in payload["loss_cols"]),
            premium_col=payload["premium_col"],
            loss_source=str(payload.get("loss_source", "predicted")),  # type: ignore[arg-type]
            observed_loss_col=str(payload.get("observed_loss_col", "Y_G_Loss")),
            u_coef=_optional_float(payload.get("u_coef")),
            u_bounds=tuple(float(value) for value in u_bounds) if u_bounds is not None else None,
            acceptance_floor=_optional_float(payload.get("acceptance_floor")),
            acceptance_penalty_weight=_optional_float(payload.get("acceptance_penalty_weight")),
            acceptance_penalty_temperature=float(payload.get("acceptance_penalty_temperature", 0.01)),
            lagrangian_lambda=_optional_float(payload.get("lagrangian_lambda")),
            acceptance_artifact_path=_optional_str(payload.get("acceptance_artifact_path")),
            loss_artifact_path=_optional_str(payload.get("loss_artifact_path")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "type": "ModelBasedObjective",
            "model_type": self.model_type,
            "acceptance_state_cols": list(self.acceptance_state_cols),
            "loss_cols": list(self.loss_cols),
            "premium_col": self.premium_col,
            "loss_source": self.loss_source,
            "observed_loss_col": self.observed_loss_col,
            "u_coef": self.u_coef,
            "u_bounds": list(self.u_bounds) if self.u_bounds is not None else None,
            "acceptance_floor": self.acceptance_floor,
            "acceptance_penalty_weight": self.acceptance_penalty_weight,
            "acceptance_penalty_temperature": self.acceptance_penalty_temperature,
            "lagrangian_lambda": self.lagrangian_lambda,
            "acceptance_artifact_path": self.acceptance_artifact_path,
            "loss_artifact_path": self.loss_artifact_path,
        }


@dataclass(frozen=True)
class PolicyDataBinding:
    """Source-row binding for replaying a saved policy on train/test/all splits."""

    model_type: ModelType
    train_row_indices: np.ndarray
    test_row_indices: np.ndarray
    selected_row_indices: np.ndarray
    dataset_schema_version: str = DATASET_SCHEMA_VERSION
    kind: str = "real_data_rows"

    @classmethod
    def from_result(cls, result: ExperimentResult, model_type: ModelType) -> "PolicyDataBinding":
        if result.train_row_indices is None:
            raise ValueError("Policy artifacts for real-data objectives require train_row_indices.")
        train_rows = np.asarray(result.train_row_indices, dtype=int)
        test_rows = (
            np.asarray(result.test_row_indices, dtype=int)
            if result.test_row_indices is not None
            else np.asarray([], dtype=int)
        )
        selected_rows = (
            np.asarray(result.config.x_fixed_row_indices, dtype=int)
            if result.config.x_fixed_row_indices is not None
            else np.concatenate([train_rows, test_rows]).astype(int)
        )
        return cls(
            model_type=model_type,
            train_row_indices=train_rows,
            test_row_indices=test_rows,
            selected_row_indices=selected_rows,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> "PolicyDataBinding":
        return cls(
            model_type=str(payload["model_type"]),  # type: ignore[arg-type]
            train_row_indices=np.asarray(arrays["train_row_indices"], dtype=int),
            test_row_indices=np.asarray(arrays["test_row_indices"], dtype=int),
            selected_row_indices=np.asarray(arrays["selected_row_indices"], dtype=int),
            dataset_schema_version=str(payload.get("dataset_schema_version", DATASET_SCHEMA_VERSION)),
            kind=str(payload.get("kind", "real_data_rows")),
        )

    def add_arrays(self, arrays: dict[str, np.ndarray]) -> None:
        arrays["train_row_indices"] = np.asarray(self.train_row_indices, dtype=int)
        arrays["test_row_indices"] = np.asarray(self.test_row_indices, dtype=int)
        arrays["selected_row_indices"] = np.asarray(self.selected_row_indices, dtype=int)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "model_type": self.model_type,
            "dataset_schema_version": self.dataset_schema_version,
            "train_n_samples": int(self.train_row_indices.size),
            "test_n_samples": int(self.test_row_indices.size),
            "selected_n_samples": int(self.selected_row_indices.size),
        }

    def row_indices(self, split: SplitName) -> np.ndarray:
        if split == "train":
            return self.train_row_indices.copy()
        if split == "test":
            if self.test_row_indices.size == 0:
                raise ValueError("Saved policy artifact has no test split rows.")
            return self.test_row_indices.copy()
        if split == "all":
            return self.selected_row_indices.copy()
        raise ValueError("split must be 'train', 'test', or 'all'.")


@dataclass(frozen=True)
class PolicyArtifact:
    """Reloadable trained policy with exact preprocessing and row binding."""

    estimator: str
    theta: np.ndarray
    policy_input: PolicyInputPreprocessingSpec
    feature_map: PolicyFeatureMapSpec | None
    policy_head: PolicyHeadSpec
    objective: ObjectiveReplaySpec
    data_binding: PolicyDataBinding
    train_metrics: PolicyEvaluation | None = None
    test_metrics: PolicyEvaluation | None = None

    def build_policy(self) -> object:
        """Rebuild the concrete policy head and feature map from saved specs."""
        feature_map = self.feature_map.build() if self.feature_map is not None else None
        return self.policy_head.build(feature_map)

    def build_objective(self) -> ModelBasedObjective:
        """Rebuild the model-based objective used to replay this policy."""
        acceptance_model, loss_model = load_model_artifacts(self.objective.model_type)
        return ModelBasedObjective(
            policy=self.build_policy(),
            acceptance_model=acceptance_model,
            loss_model=loss_model,
            acceptance_state_cols=self.objective.acceptance_state_cols,
            loss_cols=self.objective.loss_cols,
            premium_col=self.objective.premium_col,
            loss_source=self.objective.loss_source,
            observed_loss_col=self.objective.observed_loss_col,
            u_coef=self.objective.u_coef,
            u_bounds=self.objective.u_bounds,
            acceptance_floor=self.objective.acceptance_floor,
            acceptance_penalty_weight=self.objective.acceptance_penalty_weight,
            acceptance_penalty_temperature=self.objective.acceptance_penalty_temperature,
            lagrangian_lambda=self.objective.lagrangian_lambda,
            policy_preprocessor=self.policy_input.policy_preprocessor,
            policy_feature_cols=self.policy_input.policy_feature_cols,
        )

    def row_indices(self, split: SplitName = "train") -> np.ndarray:
        """Return saved canonical CSV row positions for a split."""
        return self.data_binding.row_indices(split)

    def load_x(self, split: SplitName = "train") -> object:
        """Load raw source-space rows bound to the saved split."""
        row_indices = self.row_indices(split)
        x_frame = load_x_frame(self.objective.model_type, row_indices=row_indices)
        if self.objective.loss_source == "observed":
            x_frame = x_frame.copy()
            x_frame[self.objective.observed_loss_col] = load_observed_loss_array(
                self.objective.model_type,
                row_indices=row_indices,
            )
        return x_frame

    def policy_input_features(self, x_batch: object | None = None, *, split: SplitName = "train") -> np.ndarray:
        """Return preprocessed policy inputs ``z`` before applying ``varphi`` or ``phi``."""
        x_eval = self.load_x(split) if x_batch is None else x_batch
        objective = self.build_objective()
        return np.asarray(objective._policy_features(x_eval), dtype=float)

    def mapped_features(self, x_batch: object | None = None, *, split: SplitName = "train") -> np.ndarray:
        """Return ``varphi(z)`` after policy input preprocessing and before intercept handling."""
        z = self.policy_input_features(x_batch, split=split)
        if self.feature_map is None:
            return np.empty((z.shape[0], 0), dtype=float)
        return np.asarray(self.feature_map.build().transform(z), dtype=float)

    def policy_design_matrix(self, x_batch: object | None = None, *, split: SplitName = "train") -> np.ndarray:
        """Return ``phi(z)`` for linear/softmax policies, distinct from preprocessing ``z``."""
        mapped = self.mapped_features(x_batch, split=split)
        if self.policy_head.type in {"LinearPolicy", "SoftmaxPolicy"}:
            intercept = np.ones((mapped.shape[0], 1), dtype=float)
            return np.concatenate([intercept, mapped], axis=1)
        return mapped

    def predict_u(
        self,
        x_batch: object | None = None,
        *,
        split: SplitName = "train",
        clip: bool = True,
    ) -> np.ndarray:
        """Replay the saved theta and return one policy action per row."""
        x_eval = self.load_x(split) if x_batch is None else x_batch
        objective = self.build_objective()
        return policy_u_values(objective, self.theta, x_eval, clip=clip)

    def evaluate(self, x_batch: object | None = None, *, split: SplitName = "train") -> PolicyEvaluation:
        """Evaluate the saved policy on a split or supplied batch without retraining."""
        x_eval = self.load_x(split) if x_batch is None else x_batch
        return evaluate_policy(self.build_objective(), self.theta, x_eval)

    def to_dict(self, *, array_file: str = _ARRAY_FILE) -> dict[str, object]:
        return {
            "schema_version": _ARTIFACT_SCHEMA_VERSION,
            "array_file": array_file,
            "estimator": self.estimator,
            "theta_shape": list(self.theta.shape),
            "policy_input_preprocessing": self.policy_input.to_dict(),
            "feature_map": self.feature_map.to_dict() if self.feature_map is not None else None,
            "policy_head": self.policy_head.to_dict(),
            "objective": self.objective.to_dict(),
            "data_binding": self.data_binding.to_dict(),
            "metrics": {
                "train": _evaluation_to_dict(self.train_metrics),
                "test": _evaluation_to_dict(self.test_metrics),
            },
        }

    def save(self, path: str | Path) -> Path:
        """Save ``policy.json`` plus sidecar arrays and return the JSON path."""
        json_path = Path(path)
        if json_path.suffix != ".json":
            json_path = json_path / "policy.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        arrays_path = json_path.with_name(_ARRAY_FILE)
        arrays: dict[str, np.ndarray] = {"theta": np.asarray(self.theta, dtype=float)}
        self.data_binding.add_arrays(arrays)
        self.policy_input.add_arrays(arrays)
        np.savez(arrays_path, **arrays)
        payload = self.to_dict(array_file=arrays_path.name)
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        return json_path

    @classmethod
    def load(cls, path: str | Path) -> "PolicyArtifact":
        """Load a saved policy from its JSON entry point."""
        json_path = Path(path)
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("schema_version", -1)) != _ARTIFACT_SCHEMA_VERSION:
            raise ValueError("Unsupported policy artifact schema_version.")
        arrays_path = json_path.with_name(str(payload.get("array_file", _ARRAY_FILE)))
        with np.load(arrays_path, allow_pickle=False) as loaded:
            arrays = {name: np.asarray(loaded[name]).copy() for name in loaded.files}
        policy_input = PolicyInputPreprocessingSpec.from_dict(
            _mapping(payload["policy_input_preprocessing"]),
            arrays,
        )
        return cls(
            estimator=str(payload["estimator"]),
            theta=np.asarray(arrays["theta"], dtype=float),
            policy_input=policy_input,
            feature_map=PolicyFeatureMapSpec.from_dict(payload.get("feature_map")),
            policy_head=PolicyHeadSpec.from_dict(_mapping(payload["policy_head"])),
            objective=ObjectiveReplaySpec.from_dict(_mapping(payload["objective"])),
            data_binding=PolicyDataBinding.from_dict(_mapping(payload["data_binding"]), arrays),
            train_metrics=_evaluation_from_dict(_mapping(payload.get("metrics", {})).get("train")),
            test_metrics=_evaluation_from_dict(_mapping(payload.get("metrics", {})).get("test")),
        )


def build_policy_artifact(result: ExperimentResult, estimator: str) -> PolicyArtifact:
    """Build a replayable artifact for one estimator in an experiment result."""
    if estimator not in result.results:
        raise ValueError(f"Estimator '{estimator}' is not present in the result.")
    objective = _unwrap_model_based_objective(result.config.objective)
    if not isinstance(objective, ModelBasedObjective):
        raise ValueError("PolicyArtifact currently supports ModelBasedObjective results.")
    policy = ConstantPolicy() if estimator == "constant" else objective.policy
    objective_spec = ObjectiveReplaySpec.from_objective(objective)
    return PolicyArtifact(
        estimator=estimator,
        theta=np.asarray(result.results[estimator].theta, dtype=float),
        policy_input=PolicyInputPreprocessingSpec.from_objective(objective),
        feature_map=PolicyFeatureMapSpec.from_policy(policy),
        policy_head=PolicyHeadSpec.from_policy(policy),
        objective=objective_spec,
        data_binding=PolicyDataBinding.from_result(result, objective_spec.model_type),
        train_metrics=result.train_metrics.get(estimator),
        test_metrics=result.test_metrics.get(estimator),
    )


def _unwrap_model_based_objective(objective: object) -> object:
    if isinstance(objective, NoisyObjective):
        return objective.base_objective
    return objective


def load_policy_artifact(path: str | Path) -> PolicyArtifact:
    """Load a saved policy artifact from ``policy.json``."""
    return PolicyArtifact.load(path)


def save_policy_artifacts(result: ExperimentResult, output_dir: str | Path) -> dict[str, Path]:
    """Save one policy artifact per estimator and return JSON paths by estimator."""
    root = Path(output_dir)
    paths: dict[str, Path] = {}
    for estimator in result.results:
        artifact = build_policy_artifact(result, estimator)
        paths[estimator] = artifact.save(root / estimator / "policy.json")
    return paths


def _infer_model_type(objective: ModelBasedObjective) -> ModelType:
    model_type = getattr(objective.acceptance_model, "model_type", None)
    if model_type in {"glm", "xgb", "xgb_logit_spline"}:
        return model_type
    model = getattr(objective.acceptance_model, "model", objective.acceptance_model)
    module = type(model).__module__.lower()
    name = type(model).__name__.lower()
    if "xgboost" in module or "xgb" in name:
        return "xgb"
    if "sklearn" in module or "logistic" in name:
        return "glm"
    raise ValueError("Could not infer real-data model_type from objective artifacts.")


def _evaluation_to_dict(evaluation: PolicyEvaluation | None) -> dict[str, object] | None:
    if evaluation is None:
        return None
    return {
        "n_samples": int(evaluation.n_samples),
        "objective_value": float(evaluation.objective_value),
        "objective_sum": float(evaluation.objective_sum),
        "mean_u": float(evaluation.mean_u),
        "u_q25": float(evaluation.u_q25),
        "u_q75": float(evaluation.u_q75),
        "mean_acceptance": _optional_float(evaluation.mean_acceptance),
        "projected_loss": _optional_float(evaluation.projected_loss),
        "projected_revenue": _optional_float(evaluation.projected_revenue),
    }


def _evaluation_from_dict(payload: object) -> PolicyEvaluation | None:
    if payload is None:
        return None
    values = _mapping(payload)
    return PolicyEvaluation(
        n_samples=int(values["n_samples"]),
        objective_value=float(values["objective_value"]),
        objective_sum=float(values["objective_sum"]),
        mean_u=float(values["mean_u"]),
        u_q25=float(values["u_q25"]),
        u_q75=float(values["u_q75"]),
        mean_acceptance=_optional_float(values.get("mean_acceptance")),
        projected_loss=_optional_float(values.get("projected_loss")),
        projected_revenue=_optional_float(values.get("projected_revenue")),
    )


def _prefixed_arrays(arrays: Mapping[str, np.ndarray], prefix: str) -> dict[str, np.ndarray]:
    return {
        name[len(prefix):]: np.asarray(value)
        for name, value in arrays.items()
        if name.startswith(prefix)
    }


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Expected a mapping payload.")
    return dict(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


__all__ = [
    "ObjectiveReplaySpec",
    "PolicyArtifact",
    "PolicyDataBinding",
    "PolicyFeatureMapSpec",
    "PolicyHeadSpec",
    "PolicyInputPreprocessingSpec",
    "build_policy_artifact",
    "load_policy_artifact",
    "save_policy_artifacts",
]
