"""Objective module public API.

This module provides:
- Base interfaces: Objective, Policy
- Sampling: sample_states, default_rng
- Concrete policies: ConstantPolicy, LinearPolicy, SoftmaxPolicy, MLPPolicy
- Policy feature maps: AdditiveChebyshevFeatureMap, TotalDegreePolynomialFeatureMap, IdentityFeatureMap, QuadraticFeatureMap,
  CubicFeatureMap, QuarticFeatureMap, CallableFeatureMap
- Concrete objectives: FixedRegressionObjective, BiasedObjective,
  PlantedLogisticObjective, ModelBasedObjective, PreparedGLMObjective,
  JaxPreparedGLMObjective
- Synthetic ladder: SyntheticFunction, StronglyConvexQuadratic,
  SmoothedNonconvex, PiecewiseConvex, PiecewiseNonconvexDoubleWell,
  SYNTHETIC_LADDER, IMPLEMENTED_SYNTHETIC_LADDER
- Objective modifications: noise, bias, theta regularization, acceptance
  scalarization, and composition specs
- Utility: optimal_u, value_at_constant_u, mean_acceptance_at_constant_u,
  value_for_reporting
"""

from importlib import import_module

from objective.base import (
    Objective,
    Policy,
    default_rng,
    sample_states,
)
from objective.modifications import (
    AcceptanceLagrangianModification,
    AcceptanceLagrangianObjective,
    AcceptancePenaltyModification,
    AcceptancePenaltyObjective,
    ActionBias,
    ArctanRemainderThetaBias,
    ArctanThetaBias,
    BiasedObjective,
    BiasModification,
    ConstantThetaRegularizer,
    HeteroskedasticGaussianNoise,
    HomoskedasticGaussianNoise,
    IntervalDistanceThetaRegularizer,
    LinearActionBias,
    LinearThetaBias,
    NoiseModification,
    NoisyObjective,
    NoNoise,
    ObjectiveModificationSpec,
    ObjectiveNoise,
    ProximalThetaRegularizer,
    RegularizationModification,
    RegularizedObjective,
    SmoothSaturatingIntervalThetaRegularizer,
    SupportThetaRegularizer,
    ThetaRegularizer,
    ThetaBias,
    ThetaBiasBounds,
    ThetaBiasedObjective,
    ThetaBiasModification,
    UpperSupportHingeBias,
    compose_objective,
)
from objective.objectives import (
    FixedRegressionObjective,
    IMPLEMENTED_SYNTHETIC_LADDER,
    ModelBasedObjective,
    PiecewiseConvex,
    PiecewiseNonconvexDoubleWell,
    PlantedLogisticObjective,
    PreparedGLMBatch,
    PreparedGLMObjective,
    SmoothedNonconvex,
    StronglyConvexQuadratic,
    SYNTHETIC_LADDER,
    SyntheticFunction,
    ZerothOrderProofObjective,
    prepare_glm_batch,
    prepare_glm_objective,
)
from objective.policy import (
    AdditiveChebyshevFeatureMap,
    CallableFeatureMap,
    ConstantPolicy,
    CubicFeatureMap,
    FeatureMap,
    FeatureProcessedPolicy,
    IdentityFeatureMap,
    LinearPolicy,
    MLPPolicy,
    QuadraticFeatureMap,
    QuarticFeatureMap,
    SoftmaxPolicy,
    TotalDegreePolynomialFeatureMap,
    mlp_init_theta,
    policy_from_kind,
    policy_theta_dim,
)
from objective.policy_preprocessing import (
    PolicyFeaturePreprocessor,
    fit_policy_feature_preprocessor,
    make_policy_features,
)
from objective.utils import mean_acceptance_at_constant_u, optimal_u, value_at_constant_u, value_for_reporting

_JAX_EXPORTS = {
    "JaxPreparedGLMObjective",
    "JaxPreparedGLMScipyAdapter",
    "prepare_jax_glm_objective",
}


def __getattr__(name: str):
    if name in _JAX_EXPORTS:
        module = import_module("objective.objectives")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = [
    # Base interfaces
    "Objective",
    "Policy",
    "default_rng",
    "sample_states",
    # Concrete policies
    "AdditiveChebyshevFeatureMap",
    "TotalDegreePolynomialFeatureMap",
    "CallableFeatureMap",
    "ConstantPolicy",
    "CubicFeatureMap",
    "FeatureMap",
    "FeatureProcessedPolicy",
    "IdentityFeatureMap",
    "LinearPolicy",
    "MLPPolicy",
    "QuadraticFeatureMap",
    "QuarticFeatureMap",
    "SoftmaxPolicy",
    "mlp_init_theta",
    "policy_from_kind",
    "policy_theta_dim",
    "PolicyFeaturePreprocessor",
    "fit_policy_feature_preprocessor",
    "make_policy_features",
    # Concrete objectives
    "ActionBias",
    "ArctanRemainderThetaBias",
    "ArctanThetaBias",
    "AcceptanceLagrangianModification",
    "AcceptanceLagrangianObjective",
    "AcceptancePenaltyModification",
    "AcceptancePenaltyObjective",
    "BiasModification",
    "BiasedObjective",
    "ConstantThetaRegularizer",
    "FixedRegressionObjective",
    "LinearActionBias",
    "LinearThetaBias",
    "IntervalDistanceThetaRegularizer",
    "ModelBasedObjective",
    "NoiseModification",
    "PlantedLogisticObjective",
    "PreparedGLMBatch",
    "PreparedGLMObjective",
    "ObjectiveModificationSpec",
    "ProximalThetaRegularizer",
    "RegularizationModification",
    "RegularizedObjective",
    "SmoothSaturatingIntervalThetaRegularizer",
    "SupportThetaRegularizer",
    "ThetaRegularizer",
    "ThetaBias",
    "ThetaBiasBounds",
    "ThetaBiasedObjective",
    "ThetaBiasModification",
    "UpperSupportHingeBias",
    # Synthetic ladder
    "IMPLEMENTED_SYNTHETIC_LADDER",
    "PiecewiseConvex",
    "PiecewiseNonconvexDoubleWell",
    "SmoothedNonconvex",
    "StronglyConvexQuadratic",
    "SYNTHETIC_LADDER",
    "SyntheticFunction",
    "ZerothOrderProofObjective",
    "JaxPreparedGLMObjective",
    "JaxPreparedGLMScipyAdapter",
    "prepare_glm_batch",
    "prepare_glm_objective",
    "prepare_jax_glm_objective",
    # Objective noise
    "HeteroskedasticGaussianNoise",
    "HomoskedasticGaussianNoise",
    "NoisyObjective",
    "NoNoise",
    "ObjectiveNoise",
    "compose_objective",
    # Utility
    "mean_acceptance_at_constant_u",
    "optimal_u",
    "value_at_constant_u",
    "value_for_reporting",
]
