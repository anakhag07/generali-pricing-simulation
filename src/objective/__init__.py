"""Objective module public API.

This module provides:
- Base interfaces: Objective, Policy, StateVector
- Concrete policies: ConstantPolicy, LinearPolicy, SoftmaxPolicy
- Concrete objectives: FixedRegressionObjective, PlantedLogisticObjective
- Utility functions: mean_action, optimal_u, theta_grad_from_u_grad
"""

from objective.base import (
    Objective,
    Policy,
    StateVector,
    default_rng,
)
from objective.objectives import (
    FixedRegressionObjective,
    PlantedLogisticObjective,
)
from objective.policy import (
    ConstantPolicy,
    LinearPolicy,
    SoftmaxPolicy,
    policy_constant,
    policy_from_kind,
    policy_linear,
    policy_softmax,
)
from objective.utils import (
    action_value_at_u,
    mean_action,
    optimal_u,
    theta_grad_from_u_grad,
)

__all__ = [
    # Base interfaces
    "Objective",
    "Policy",
    "StateVector",
    "default_rng",
    # Policy constants
    "policy_constant",
    "policy_linear",
    "policy_softmax",
    # Concrete policies
    "ConstantPolicy",
    "LinearPolicy",
    "SoftmaxPolicy",
    "policy_from_kind",
    # Concrete objectives
    "FixedRegressionObjective",
    "PlantedLogisticObjective",
    # Utility functions
    "mean_action",
    "optimal_u",
    "action_value_at_u",
    "theta_grad_from_u_grad",
]
