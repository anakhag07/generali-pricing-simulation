"""Compatibility exports for action-bias objective modifications.

The canonical implementations live in ``objective.modifications.bias``.
"""

from objective.modifications.bias import (
    ActionBias,
    BiasedObjective,
    LinearActionBias,
    UpperSupportHingeBias,
)

__all__ = ["ActionBias", "BiasedObjective", "LinearActionBias", "UpperSupportHingeBias"]
