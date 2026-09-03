"""Core experiment configuration, execution, and result APIs."""

from experiments.config import CorrectnessSpec, ExperimentConfig
from experiments.configs import get_config, list_configs
from experiments.execution import ExecutedRun, default_reporter_stack, execute_experiment_run
from experiments.finite_policy_lcb import (
    FinitePolicyLCBManifest,
    FinitePolicyLCBSpec,
    LCBSeedResult,
    evaluate_finite_policy_lcb_seed,
    load_finite_policy_lcb_manifest,
)
from experiments.launch import LaunchContext, LaunchPlan, add_launch_args, run_launch_plan
from experiments.policy_lcb.continuous import (
    ContinuousPolicyLCBManifest,
    ContinuousPolicyLCBSpec,
    ContinuousPolicyValueSpec,
    evaluate_continuous_policy_lcb_seed,
    load_continuous_policy_lcb_manifest,
)
from experiments.policy_lcb.continuous_gp import (
    ContinuousGPVariableLCBManifest,
    ContinuousGPVariableLCBSpec,
    evaluate_continuous_gp_variable_lcb_seed,
    load_continuous_gp_variable_lcb_manifest,
)
from experiments.paths import results_root
from experiments.results import ConstantBaselineResult, EstimatorResult, ExperimentResult, OptimizationTrace, PolicyEvaluation
from experiments.run import run_experiment
from experiments.seeds import ResolvedSeedSetup, SeedSetup, resolve_seed_setup

__all__ = [
    "CorrectnessSpec",
    "ContinuousPolicyLCBManifest",
    "ContinuousPolicyLCBSpec",
    "ContinuousPolicyValueSpec",
    "ContinuousGPVariableLCBManifest",
    "ContinuousGPVariableLCBSpec",
    "ExperimentConfig",
    "get_config",
    "list_configs",
    "ExecutedRun",
    "default_reporter_stack",
    "execute_experiment_run",
    "evaluate_continuous_policy_lcb_seed",
    "evaluate_continuous_gp_variable_lcb_seed",
    "FinitePolicyLCBManifest",
    "FinitePolicyLCBSpec",
    "LCBSeedResult",
    "evaluate_finite_policy_lcb_seed",
    "load_finite_policy_lcb_manifest",
    "load_continuous_policy_lcb_manifest",
    "load_continuous_gp_variable_lcb_manifest",
    "results_root",
    "LaunchContext",
    "LaunchPlan",
    "add_launch_args",
    "run_launch_plan",
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
