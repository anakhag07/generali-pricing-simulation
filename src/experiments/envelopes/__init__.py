"""Reusable band mathematics, fitted-state replay and retained wild-envelope studies."""
from .core import (CalibrationData, ContainmentResult, EnvelopeValues, FittedEnvelope,
    ProbabilityStatement, TargetDomain, aggregate_coverage, check_finite_containment, evaluate_tightness)
from .fitting import fit_band, save_band, load_band
from .support import fit_support_band
from .linear import LinearModel, QuadraticModel
from .wild import WildLinearDesign
from .feature_wild import WildFeatureDesign, feature_band_components
from .polynomial import AdditivePolynomial, PolynomialFeatureBand, polynomial_features
from .mlp_head import FrozenResidualMLP, FrozenMLPBand, calibrate_frozen_head
from .structured_residual import LocalizedResidual

__all__ = ["CalibrationData", "ContainmentResult", "EnvelopeValues", "FittedEnvelope", "ProbabilityStatement",
    "TargetDomain", "aggregate_coverage", "check_finite_containment", "evaluate_tightness", "fit_band",
    "save_band", "load_band", "fit_support_band", "LinearModel", "QuadraticModel", "WildLinearDesign",
    "WildFeatureDesign", "feature_band_components", "AdditivePolynomial",
    "PolynomialFeatureBand", "polynomial_features", "FrozenResidualMLP", "FrozenMLPBand",
    "calibrate_frozen_head", "LocalizedResidual"]
