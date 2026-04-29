"""Objective module public API.

This module provides:
- Base interfaces: Objective, Policy
- Sampling: sample_states, default_rng
- Concrete policies: ConstantPolicy, LinearPolicy, SoftmaxPolicy
- Policy feature maps: IdentityFeatureMap, QuadraticFeatureMap,
  CallableFeatureMap
- Concrete objectives: FixedRegressionObjective, PlantedLogisticObjective,
  ModelBasedObjective
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
)
from objective.policy import (
    CallableFeatureMap,
    ConstantPolicy,
    FeatureMap,
    FeatureProcessedPolicy,
    IdentityFeatureMap,
    LinearPolicy,
    QuadraticFeatureMap,
    SoftmaxPolicy,
    policy_from_kind,
    policy_theta_dim,
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
    "FeatureMap",
    "FeatureProcessedPolicy",
    "IdentityFeatureMap",
    "LinearPolicy",
    "QuadraticFeatureMap",
    "SoftmaxPolicy",
    "policy_from_kind",
    "policy_theta_dim",
    # Concrete objectives
    "FixedRegressionObjective",
    "ModelBasedObjective",
    "PlantedLogisticObjective",
    # Utility
    "mean_acceptance_at_constant_u",
    "optimal_u",
    "value_at_constant_u",
    "value_for_reporting",
]
