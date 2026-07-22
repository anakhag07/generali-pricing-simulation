"""Objective-value modifications.

This package groups wrappers that modify an objective's scalar value while
preserving the same theta-level interface.
"""

from objective.modifications.acceptance import (
    AcceptanceLagrangianObjective,
    AcceptancePenaltyObjective,
)
from objective.modifications.bias import (
    ActionBias,
    ArctanRemainderThetaBias,
    ArctanThetaBias,
    BiasedObjective,
    LinearActionBias,
    LinearThetaBias,
    ThetaBias,
    ThetaBiasBounds,
    ThetaBiasedObjective,
    UpperSupportHingeBias,
)
from objective.modifications.composition import (
    AcceptanceLagrangianModification,
    AcceptancePenaltyModification,
    BiasModification,
    NoiseModification,
    ObjectiveModificationSpec,
    RegularizationModification,
    ThetaBiasModification,
    action_bias_from_dict,
    action_bias_to_dict,
    coerce_objective_modification,
    compose_objective,
    modification_to_dict,
    noise_from_dict,
    noise_to_dict,
    theta_bias_from_dict,
    theta_bias_to_dict,
)
from objective.modifications.noise import (
    HeteroskedasticGaussianNoise,
    HomoskedasticGaussianNoise,
    NoisyObjective,
    NoNoise,
    ObjectiveNoise,
)
from objective.modifications.regularization import (
    ProximalThetaRegularizer,
    RegularizedObjective,
    SupportThetaRegularizer,
    ThetaRegularizer,
    regularizer_from_dict,
    regularizer_to_dict,
)

__all__ = [
    "AcceptanceLagrangianModification",
    "AcceptanceLagrangianObjective",
    "AcceptancePenaltyModification",
    "AcceptancePenaltyObjective",
    "ActionBias",
    "ArctanRemainderThetaBias",
    "ArctanThetaBias",
    "BiasModification",
    "BiasedObjective",
    "HeteroskedasticGaussianNoise",
    "HomoskedasticGaussianNoise",
    "LinearActionBias",
    "LinearThetaBias",
    "NoiseModification",
    "NoisyObjective",
    "NoNoise",
    "ObjectiveModificationSpec",
    "ObjectiveNoise",
    "ProximalThetaRegularizer",
    "RegularizationModification",
    "RegularizedObjective",
    "SupportThetaRegularizer",
    "ThetaRegularizer",
    "ThetaBias",
    "ThetaBiasBounds",
    "ThetaBiasedObjective",
    "ThetaBiasModification",
    "UpperSupportHingeBias",
    "action_bias_from_dict",
    "action_bias_to_dict",
    "coerce_objective_modification",
    "compose_objective",
    "modification_to_dict",
    "noise_from_dict",
    "noise_to_dict",
    "theta_bias_from_dict",
    "theta_bias_to_dict",
    "regularizer_from_dict",
    "regularizer_to_dict",
]
