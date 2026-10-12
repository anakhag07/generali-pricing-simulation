"""Reusable uncertainty-envelope construction, coverage, and tightness studies."""

from . import core, linear, reporting, study
from .core import (
    CalibrationData, ContainmentResult, EnvelopeValues, FittedEnvelope,
    ProbabilityStatement, TargetDomain, aggregate_coverage,
    check_finite_containment, evaluate_tightness,
)
from .study import EnvelopeManifest, build_launch_plan, collect, load_manifest, run_study

__all__ = [
    "core", "linear", "reporting", "study",
    "CalibrationData", "ContainmentResult", "EnvelopeValues", "FittedEnvelope",
    "ProbabilityStatement", "TargetDomain", "aggregate_coverage",
    "check_finite_containment", "evaluate_tightness", "EnvelopeManifest",
    "build_launch_plan", "collect", "load_manifest", "run_study",
]
