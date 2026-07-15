"""Objective module public API.

This module provides:
- Base interfaces: Objective, Policy
- Sampling: sample_states, default_rng
- Concrete policies: ConstantPolicy, LinearPolicy, SoftmaxPolicy, MLPPolicy
- Policy feature maps: IdentityFeatureMap, QuadraticFeatureMap,
  CubicFeatureMap, QuarticFeatureMap, CallableFeatureMap
- Concrete objectives: QuadraticObjective, FixedRegressionObjective,
  BiasedObjective, PlantedLogisticObjective, ModelBasedObjective,
  PreparedGLMObjective, JaxPreparedGLMObjective
- Synthetic ladder: SyntheticFunction, StronglyConvexQuadratic,
  SmoothedNonconvex, PiecewiseConvex, PiecewiseNonconvexDoubleWell,
  SYNTHETIC_LADDER, IMPLEMENTED_SYNTHETIC_LADDER
- Objective noise: ObjectiveNoise, NoNoise, HomoskedasticGaussianNoise,
  HeteroskedasticGaussianNoise, NoisyObjective
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
from objective.objectives import (
    ActionBias,
    BiasedObjective,
    FixedRegressionObjective,
    IMPLEMENTED_SYNTHETIC_LADDER,
    LinearActionBias,
    ModelBasedObjective,
    PiecewiseConvex,
    PiecewiseNonconvexDoubleWell,
    PlantedLogisticObjective,
    QuadraticObjective,
    PreparedGLMBatch,
    PreparedGLMObjective,
    SmoothedNonconvex,
    StronglyConvexQuadratic,
    SYNTHETIC_LADDER,
    SyntheticFunction,
    UpperSupportHingeBias,
    prepare_glm_batch,
    prepare_glm_objective,
)
from objective.noise import (
    HeteroskedasticGaussianNoise,
    HomoskedasticGaussianNoise,
    NoisyObjective,
    NoNoise,
    ObjectiveNoise,
)
from objective.policy import (
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
    "BiasedObjective",
    "FixedRegressionObjective",
    "LinearActionBias",
    "ModelBasedObjective",
    "PlantedLogisticObjective",
    "QuadraticObjective",
    "PreparedGLMBatch",
    "PreparedGLMObjective",
    "UpperSupportHingeBias",
    # Synthetic ladder
    "IMPLEMENTED_SYNTHETIC_LADDER",
    "PiecewiseConvex",
    "PiecewiseNonconvexDoubleWell",
    "SmoothedNonconvex",
    "StronglyConvexQuadratic",
    "SYNTHETIC_LADDER",
    "SyntheticFunction",
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
    # Utility
    "mean_acceptance_at_constant_u",
    "optimal_u",
    "value_at_constant_u",
    "value_for_reporting",
]
