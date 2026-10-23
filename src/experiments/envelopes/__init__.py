"""Reusable band mathematics, fitted-state replay and retained wild-envelope studies."""
from .core import (CalibrationData, ContainmentResult, EnvelopeValues, FittedEnvelope,
    ProbabilityStatement, TargetDomain, aggregate_coverage, check_finite_containment, evaluate_tightness)
from .fitting import fit_band, save_band, load_band
from .support import fit_support_band
from .linear import LinearModel, QuadraticModel
from .wild import WildLinearDesign

__all__ = ["CalibrationData", "ContainmentResult", "EnvelopeValues", "FittedEnvelope", "ProbabilityStatement",
    "TargetDomain", "aggregate_coverage", "check_finite_containment", "evaluate_tightness", "fit_band",
    "save_band", "load_band", "fit_support_band", "LinearModel", "QuadraticModel", "WildLinearDesign"]
