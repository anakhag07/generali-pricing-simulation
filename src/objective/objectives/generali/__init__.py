"""Generali pricing objectives backed by the real dataset and trained artifacts.

Every objective here depends on `src/data` (the canonical CSV and the fitted
acceptance/loss artifacts), which is what separates it from
`objective.objectives.synthetic`. JAX exports stay lazy so NumPy-only runs never
import JAX.
"""

from importlib import import_module

from objective.objectives.generali.model_based import ModelBasedObjective
from objective.objectives.generali.prepared_glm import (
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
        module = import_module("objective.objectives.generali.jax_prepared_glm")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "JaxPreparedGLMObjective",
    "JaxPreparedGLMScipyAdapter",
    "ModelBasedObjective",
    "PreparedGLMBatch",
    "PreparedGLMObjective",
    "prepare_glm_batch",
    "prepare_glm_objective",
    "prepare_jax_glm_objective",
]
