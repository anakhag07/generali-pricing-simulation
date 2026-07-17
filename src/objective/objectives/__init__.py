"""Concrete objective implementations.

Split by provenance: `generali` objectives are bound to the real dataset and
trained artifacts under `src/data`; `synthetic` objectives are self-contained with
analytically known optima. Objective-value wrappers now live in
`objective.modifications` and are re-exported here for compatibility.

This module re-exports the whole surface, so `from objective import X` and
`from objective.objectives import X` are unaffected by the split.
"""

from importlib import import_module

from objective.modifications import (
    ActionBias,
    BiasedObjective,
    LinearActionBias,
    UpperSupportHingeBias,
)
from objective.objectives.generali import (
    ModelBasedObjective,
    PreparedGLMBatch,
    PreparedGLMObjective,
    prepare_glm_batch,
    prepare_glm_objective,
)
from objective.objectives.synthetic import (
    FixedRegressionObjective,
    IMPLEMENTED_SYNTHETIC_LADDER,
    PiecewiseConvex,
    PiecewiseNonconvexDoubleWell,
    PlantedLogisticObjective,
    SmoothedNonconvex,
    StronglyConvexQuadratic,
    SYNTHETIC_LADDER,
    SyntheticFunction,
)

_JAX_EXPORTS = {
    "JaxPreparedGLMObjective",
    "JaxPreparedGLMScipyAdapter",
    "prepare_jax_glm_objective",
}


def __getattr__(name: str):
    if name in _JAX_EXPORTS:
        module = import_module("objective.objectives.generali.jax_prepared_glm")
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
