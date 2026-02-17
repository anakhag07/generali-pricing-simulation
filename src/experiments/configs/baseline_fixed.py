"""Baseline deterministic fixed-regression configuration."""

from __future__ import annotations

from experiments.config import ExperimentConfig, OBJECTIVE_FIXED_REGRESSION

CONFIG = ExperimentConfig(objective_kind=OBJECTIVE_FIXED_REGRESSION)
