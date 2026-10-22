"""Reusable uncertainty-envelope construction, coverage, and tightness studies."""

from . import core, linear, reporting, study, diagnostics, wild, wild_study, wild_reporting
from .wild import WildLinearDesign
from .core import (
    CalibrationData, ContainmentResult, EnvelopeValues, FittedEnvelope,
    ProbabilityStatement, TargetDomain, aggregate_coverage,
    check_finite_containment, evaluate_tightness,
)
from .study import EnvelopeManifest, build_launch_plan, collect, load_manifest, run_study
from .diagnostics import write_diagnostic_reports

__all__ = [
    "core", "linear", "reporting", "study", "diagnostics", "wild", "wild_study",
    "wild_reporting", "WildLinearDesign",
    "CalibrationData", "ContainmentResult", "EnvelopeValues", "FittedEnvelope",
    "ProbabilityStatement", "TargetDomain", "aggregate_coverage",
    "check_finite_containment", "evaluate_tightness", "EnvelopeManifest",
    "build_launch_plan", "collect", "load_manifest", "run_study",
    "write_diagnostic_reports",
]
