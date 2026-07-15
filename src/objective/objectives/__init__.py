"""Concrete objective implementations."""

from importlib import import_module

from objective.objectives.fixed_regression import FixedRegressionObjective
from objective.objectives.biased import ActionBias, BiasedObjective, LinearActionBias, UpperSupportHingeBias
from objective.objectives.model_based import ModelBasedObjective
from objective.objectives.planted_logistic import PlantedLogisticObjective
from objective.objectives.quadratic import QuadraticObjective
from objective.objectives.synthetic import (
    IMPLEMENTED_SYNTHETIC_LADDER,
    PiecewiseConvex,
    PiecewiseNonconvexDoubleWell,
    SmoothedNonconvex,
    StronglyConvexQuadratic,
    SYNTHETIC_LADDER,
    SyntheticFunction,
)
from objective.objectives.prepared_glm import (
    PreparedGLMBatch,
    PreparedGLMObjective,
    prepare_glm_batch,
    prepare_glm_objective,
)

_JAX_EXPORTS = {
    "JaxPreparedGLMObjective",
    "JaxPreparedGLMScipyAdapter",
    "prepare_jax_glm_objective",
}


def __getattr__(name: str):
    if name in _JAX_EXPORTS:
        module = import_module("objective.objectives.jax_prepared_glm")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = [
    "ActionBias",
    "FixedRegressionObjective",
    "BiasedObjective",
    "LinearActionBias",
    "UpperSupportHingeBias",
    "IMPLEMENTED_SYNTHETIC_LADDER",
    "ModelBasedObjective",
    "PiecewiseConvex",
    "PiecewiseNonconvexDoubleWell",
    "PlantedLogisticObjective",
    "QuadraticObjective",
    "SmoothedNonconvex",
    "StronglyConvexQuadratic",
    "SYNTHETIC_LADDER",
    "SyntheticFunction",
    "PreparedGLMBatch",
    "PreparedGLMObjective",
    "JaxPreparedGLMObjective",
    "JaxPreparedGLMScipyAdapter",
    "prepare_glm_batch",
    "prepare_glm_objective",
    "prepare_jax_glm_objective",
]
