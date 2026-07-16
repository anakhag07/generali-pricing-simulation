"""Core experiment configuration, execution, and result APIs."""

from experiments.config import CorrectnessSpec, ExperimentConfig
from experiments.configs import get_config, list_configs
from experiments.execution import ExecutedRun, default_reporter_stack, execute_experiment_run
from experiments.launch import LaunchContext, LaunchPlan, add_launch_args, run_launch_plan
from experiments.manifest import ExperimentManifestResult, ManifestSweepResult, run_experiment_manifest
from experiments.paths import results_root
from experiments.results import ConstantBaselineResult, EstimatorResult, ExperimentResult, OptimizationTrace, PolicyEvaluation
from experiments.run import run_experiment
from experiments.seeds import ResolvedSeedSetup, SeedSetup, resolve_seed_setup

__all__ = [
    "CorrectnessSpec",
    "ExperimentConfig",
    "get_config",
    "list_configs",
    "ExecutedRun",
    "default_reporter_stack",
    "execute_experiment_run",
    "results_root",
    "LaunchContext",
    "LaunchPlan",
    "add_launch_args",
    "run_launch_plan",
    "ExperimentManifestResult",
    "ManifestSweepResult",
    "run_experiment_manifest",
    "ConstantBaselineResult",
    "EstimatorResult",
    "ExperimentResult",
    "OptimizationTrace",
    "PolicyEvaluation",
    "run_experiment",
    "SeedSetup",
    "ResolvedSeedSetup",
    "resolve_seed_setup",
]
