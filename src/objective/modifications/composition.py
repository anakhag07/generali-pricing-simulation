"""Typed objective modification specs and composition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from objective.base import Objective
from objective.modifications.acceptance import (
    AcceptanceLagrangianObjective,
    AcceptancePenaltyObjective,
)
from objective.modifications.bias import (
    ActionBias,
    BiasedObjective,
    LinearActionBias,
    UpperSupportHingeBias,
)
from objective.modifications.noise import (
    HeteroskedasticGaussianNoise,
    HomoskedasticGaussianNoise,
    NoisyObjective,
    NoNoise,
    ObjectiveNoise,
)
from objective.modifications.regularization import (
    RegularizedObjective,
    ThetaRegularizer,
    regularizer_from_dict,
    regularizer_to_dict,
)


class ObjectiveModificationSpec:
    """Configuration object that wraps an objective with one modification."""

    def apply(self, objective: Objective) -> Objective:
        """Return ``objective`` wrapped with this modification."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serialize the modification spec for run summaries."""
        raise NotImplementedError


@dataclass(frozen=True)
class BiasModification(ObjectiveModificationSpec):
    """Apply an action-bias wrapper."""

    lambda_bias: float | None = None
    bias: ActionBias | Mapping[str, Any] | None = None

    def apply(self, objective: Objective) -> Objective:
        bias = None if self.bias is None else _coerce_action_bias(self.bias)
        return BiasedObjective(base_objective=objective, lambda_bias=self.lambda_bias, bias=bias)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "BiasModification",
            "lambda_bias": None if self.lambda_bias is None else float(self.lambda_bias),
            "bias": None if self.bias is None else action_bias_to_dict(_coerce_action_bias(self.bias)),
        }


@dataclass(frozen=True)
class NoiseModification(ObjectiveModificationSpec):
    """Apply an additive-noise wrapper."""

    noise: ObjectiveNoise | Mapping[str, Any]

    def apply(self, objective: Objective) -> Objective:
        return NoisyObjective(base_objective=objective, noise=_coerce_noise(self.noise))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "NoiseModification", "noise": noise_to_dict(_coerce_noise(self.noise))}


@dataclass(frozen=True)
class RegularizationModification(ObjectiveModificationSpec):
    """Apply one or more theta-space regularizers."""

    regularizers: Sequence[ThetaRegularizer | Mapping[str, Any]]

    def apply(self, objective: Objective) -> Objective:
        return RegularizedObjective(
            base_objective=objective,
            regularizers=tuple(_coerce_regularizer(regularizer) for regularizer in self.regularizers),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "RegularizationModification",
            "regularizers": [
                regularizer_to_dict(_coerce_regularizer(regularizer))
                for regularizer in self.regularizers
            ],
        }


@dataclass(frozen=True)
class AcceptancePenaltyModification(ObjectiveModificationSpec):
    """Apply a smooth scalar penalty for violating a mean-acceptance floor."""

    acceptance_floor: float
    acceptance_penalty_weight: float
    acceptance_penalty_temperature: float = 0.01

    def apply(self, objective: Objective) -> Objective:
        return AcceptancePenaltyObjective(
            base_objective=objective,
            acceptance_floor=float(self.acceptance_floor),
            acceptance_penalty_weight=float(self.acceptance_penalty_weight),
            acceptance_penalty_temperature=float(self.acceptance_penalty_temperature),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "AcceptancePenaltyModification",
            "acceptance_floor": float(self.acceptance_floor),
            "acceptance_penalty_weight": float(self.acceptance_penalty_weight),
            "acceptance_penalty_temperature": float(self.acceptance_penalty_temperature),
        }


@dataclass(frozen=True)
class AcceptanceLagrangianModification(ObjectiveModificationSpec):
    """Apply a scalar Lagrangian term for a mean-acceptance floor."""

    acceptance_floor: float
    lagrangian_lambda: float

    def apply(self, objective: Objective) -> Objective:
        return AcceptanceLagrangianObjective(
            base_objective=objective,
            acceptance_floor=float(self.acceptance_floor),
            lagrangian_lambda=float(self.lagrangian_lambda),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "AcceptanceLagrangianModification",
            "acceptance_floor": float(self.acceptance_floor),
            "lagrangian_lambda": float(self.lagrangian_lambda),
        }


def compose_objective(
    objective: Objective,
    modifications: Sequence[ObjectiveModificationSpec | Mapping[str, Any]] = (),
) -> Objective:
    """Apply modifications to an objective in explicit order."""
    composed = objective
    for modification in modifications:
        composed = coerce_objective_modification(modification).apply(composed)
    return composed


def coerce_objective_modification(
    modification: ObjectiveModificationSpec | Mapping[str, Any],
) -> ObjectiveModificationSpec:
    """Return a typed objective modification spec."""
    if isinstance(modification, ObjectiveModificationSpec):
        return modification
    if not isinstance(modification, Mapping):
        raise TypeError(f"Unsupported objective modification: {type(modification).__name__}.")
    kind = str(modification.get("type") or modification.get("kind") or "")
    if kind in {"BiasModification", "bias"}:
        return BiasModification(
            lambda_bias=(
                None
                if modification.get("lambda_bias") is None
                else float(modification["lambda_bias"])
            ),
            bias=modification.get("bias"),
        )
    if kind in {"NoiseModification", "noise"}:
        return NoiseModification(noise=_mapping_value(modification, "noise"))
    if kind in {"RegularizationModification", "regularization"}:
        return RegularizationModification(
            regularizers=tuple(_mapping_value(modification, "regularizers"))
        )
    if kind in {"AcceptancePenaltyModification", "acceptance_penalty"}:
        return AcceptancePenaltyModification(
            acceptance_floor=float(_mapping_value(modification, "acceptance_floor")),
            acceptance_penalty_weight=float(_mapping_value(modification, "acceptance_penalty_weight")),
            acceptance_penalty_temperature=float(
                modification.get("acceptance_penalty_temperature", 0.01)
            ),
        )
    if kind in {"AcceptanceLagrangianModification", "acceptance_lagrangian"}:
        return AcceptanceLagrangianModification(
            acceptance_floor=float(_mapping_value(modification, "acceptance_floor")),
            lagrangian_lambda=float(_mapping_value(modification, "lagrangian_lambda")),
        )
    raise ValueError(f"Unknown objective modification type: {kind!r}.")


def modification_to_dict(
    modification: ObjectiveModificationSpec | Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize an objective modification spec."""
    return coerce_objective_modification(modification).to_dict()


def noise_to_dict(noise: ObjectiveNoise) -> dict[str, Any]:
    """Serialize an objective noise adapter."""
    if isinstance(noise, NoNoise):
        return {"type": "NoNoise"}
    if isinstance(noise, HomoskedasticGaussianNoise):
        return {
            "type": "HomoskedasticGaussianNoise",
            "std": float(noise.std),
            "seed": int(noise.seed) if noise.seed is not None else None,
        }
    if isinstance(noise, HeteroskedasticGaussianNoise):
        return {
            "type": "HeteroskedasticGaussianNoise",
            "base_std": float(noise.base_std),
            "growth": float(noise.growth),
            "u_center": float(noise.u_center),
            "seed": int(noise.seed) if noise.seed is not None else None,
        }
    return {"type": type(noise).__name__}


def noise_from_dict(payload: Mapping[str, Any]) -> ObjectiveNoise:
    """Build a noise adapter from a config/summary payload."""
    kind = str(payload.get("type") or payload.get("kind") or "")
    if kind == "NoNoise":
        return NoNoise()
    if kind == "HomoskedasticGaussianNoise":
        return HomoskedasticGaussianNoise(
            std=float(payload.get("std", 1.0)),
            seed=None if payload.get("seed") is None else int(payload["seed"]),
        )
    if kind == "HeteroskedasticGaussianNoise":
        return HeteroskedasticGaussianNoise(
            base_std=float(payload.get("base_std", 0.0)),
            growth=float(payload.get("growth", 1.0)),
            u_center=float(payload.get("u_center", 0.0)),
            seed=None if payload.get("seed") is None else int(payload["seed"]),
        )
    raise ValueError(f"Unknown objective noise type: {kind!r}.")


def action_bias_to_dict(bias: ActionBias) -> dict[str, Any]:
    """Serialize an action-bias term."""
    if isinstance(bias, LinearActionBias):
        return {"type": "LinearActionBias", "lambda_bias": float(bias.lambda_bias)}
    if isinstance(bias, UpperSupportHingeBias):
        return {
            "type": "UpperSupportHingeBias",
            "lambda_bias": float(bias.lambda_bias),
            "support_center": float(bias.support_center),
            "support_radius": float(bias.support_radius),
            "support_upper": float(bias.support_upper),
            "smooth_tau": float(bias.smooth_tau) if bias.smooth_tau is not None else None,
        }
    return {"type": type(bias).__name__, "lambda_bias": float(bias.lambda_bias)}


def action_bias_from_dict(payload: Mapping[str, Any]) -> ActionBias:
    """Build an action-bias term from a config/summary payload."""
    kind = str(payload.get("type") or payload.get("kind") or "")
    if kind == "LinearActionBias":
        return LinearActionBias(lambda_bias=float(payload["lambda_bias"]))
    if kind == "UpperSupportHingeBias":
        return UpperSupportHingeBias(
            lambda_bias=float(payload["lambda_bias"]),
            support_center=float(payload["support_center"]),
            support_radius=float(payload["support_radius"]),
            smooth_tau=None if payload.get("smooth_tau") is None else float(payload["smooth_tau"]),
        )
    raise ValueError(f"Unknown action-bias type: {kind!r}.")


def _coerce_noise(noise: ObjectiveNoise | Mapping[str, Any]) -> ObjectiveNoise:
    if isinstance(noise, ObjectiveNoise):
        return noise
    if isinstance(noise, Mapping):
        return noise_from_dict(noise)
    raise TypeError(f"Unsupported objective noise: {type(noise).__name__}.")


def _coerce_action_bias(bias: ActionBias | Mapping[str, Any]) -> ActionBias:
    if isinstance(bias, ActionBias):
        return bias
    if isinstance(bias, Mapping):
        return action_bias_from_dict(bias)
    raise TypeError(f"Unsupported action-bias term: {type(bias).__name__}.")


def _coerce_regularizer(regularizer: ThetaRegularizer | Mapping[str, Any]) -> ThetaRegularizer:
    if isinstance(regularizer, ThetaRegularizer):
        return regularizer
    if isinstance(regularizer, Mapping):
        return regularizer_from_dict(dict(regularizer))
    raise TypeError(f"Unsupported theta regularizer: {type(regularizer).__name__}.")


def _mapping_value(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise ValueError(f"Objective modification payload is missing required key {key!r}.")
    return payload[key]


__all__ = [
    "AcceptanceLagrangianModification",
    "AcceptancePenaltyModification",
    "BiasModification",
    "NoiseModification",
    "ObjectiveModificationSpec",
    "RegularizationModification",
    "action_bias_from_dict",
    "action_bias_to_dict",
    "coerce_objective_modification",
    "compose_objective",
    "modification_to_dict",
    "noise_from_dict",
    "noise_to_dict",
]
