"""Concrete objective implementations."""

from objective.objectives.fixed_regression import FixedRegressionObjective
from objective.objectives.planted_logistic import PlantedLogisticObjective

__all__ = [
    "FixedRegressionObjective",
    "PlantedLogisticObjective",
]
