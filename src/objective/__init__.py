"""Objective module public API.

This module provides:
- Base interfaces: Objective, Policy
- Sampling: sample_states, default_rng
- Concrete policies: ConstantPolicy, LinearPolicy, SoftmaxPolicy, MLPPolicy
- Policy feature maps: IdentityFeatureMap, QuadraticFeatureMap,
  CubicFeatureMap, QuarticFeatureMap, CallableFeatureMap
- Concrete objectives: FixedRegressionObjective, PlantedLogisticObjective,
  ModelBasedObjective, PreparedGLMObjective, JaxPreparedGLMObjective
- Objective noise: ObjectiveNoise, NoNoise, HomoskedasticGaussianNoise,
  HeteroskedasticGaussianNoise, NoisyObjective
 - Utility: optimal_u, value_at_constant_u, mean_acceptance_at_constant_u,
   value_for_reporting
"""

from objective.base import (
    Objective,
    Policy,
    default_rng,
    sample_states,
)
from objective.objectives import (
    FixedRegressionObjective,
    ModelBasedObjective,
    PlantedLogisticObjective,
    PreparedGLMBatch,
    PreparedGLMObjective,
    JaxPreparedGLMObjective,
    JaxPreparedGLMScipyAdapter,
    prepare_glm_batch,
    prepare_glm_objective,
    prepare_jax_glm_objective,
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
    "FixedRegressionObjective",
    "ModelBasedObjective",
    "PlantedLogisticObjective",
    "PreparedGLMBatch",
    "PreparedGLMObjective",
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
