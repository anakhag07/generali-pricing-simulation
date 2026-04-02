"""Objective module public API.

This module provides:
- Base interfaces: Objective, Policy
- Sampling: sample_states, default_rng
- Concrete policies: ConstantPolicy, LinearPolicy, SoftmaxPolicy
- Concrete objectives: FixedRegressionObjective, PlantedLogisticObjective,
  ModelBasedObjective, CSVObjective
- Utility: optimal_u
"""

from objective.base import (
    Objective,
    Policy,
    default_rng,
    sample_states,
)
from objective.objectives import (
    CSVObjective,
    FixedRegressionObjective,
    ModelBasedObjective,
    PlantedLogisticObjective,
)
from objective.policy import (
    ConstantPolicy,
    LinearPolicy,
    SoftmaxPolicy,
    policy_from_kind,
)
from objective.utils import optimal_u

__all__ = [
    # Base interfaces
    "Objective",
    "Policy",
    "default_rng",
    "sample_states",
    # Concrete policies
    "ConstantPolicy",
    "LinearPolicy",
    "SoftmaxPolicy",
    "policy_from_kind",
    # Concrete objectives
    "CSVObjective",
    "FixedRegressionObjective",
    "ModelBasedObjective",
    "PlantedLogisticObjective",
    # Utility
    "optimal_u",
]
