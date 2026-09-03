"""Shared manifest dispatch and seed-array launch planning for policy-LCB adapters."""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

from experiments.launch import LaunchContext, LaunchPlan
from experiments.policy_lcb.common import read_json
from experiments.policy_lcb.continuous import (
    ContinuousPolicyLCBManifest,
    collect_continuous_policy_lcb_outputs,
    parse_continuous_policy_lcb_manifest,
    run_continuous_policy_lcb_manifest_seed,
    run_continuous_policy_lcb_manifest_serial,
)
from experiments.policy_lcb.continuous_gp import (
    ContinuousGPVariableLCBManifest,
    collect_continuous_gp_variable_lcb_outputs,
    parse_continuous_gp_variable_lcb_manifest,
    run_continuous_gp_variable_lcb_manifest_seed,
    run_continuous_gp_variable_lcb_manifest_serial,
)
from experiments.policy_lcb.continuous_gp_decomposition import (
    ContinuousGPDecompositionManifest,
    collect_continuous_gp_decomposition_outputs,
    parse_continuous_gp_decomposition_manifest,
    run_continuous_gp_decomposition_manifest_seed,
    run_continuous_gp_decomposition_manifest_serial,
)
from experiments.policy_lcb.finite import (
    FinitePolicyLCBManifest,
    collect_finite_policy_lcb_outputs,
    parse_finite_policy_lcb_manifest,
    run_finite_policy_lcb_manifest_seed,
    run_finite_policy_lcb_manifest_serial,
)
from experiments.policy_lcb.finite_grid import (
    VariableFiniteGridLCBManifest,
    collect_variable_finite_grid_lcb_outputs,
    parse_variable_finite_grid_lcb_manifest,
    run_variable_finite_grid_lcb_manifest_seed,
    run_variable_finite_grid_lcb_manifest_serial,
)


PolicyLCBManifest: TypeAlias = (
    FinitePolicyLCBManifest
    | ContinuousPolicyLCBManifest
    | VariableFiniteGridLCBManifest
    | ContinuousGPVariableLCBManifest
    | ContinuousGPDecompositionManifest
)
POLICY_LCB_MANIFEST_KINDS = frozenset(
    {
        "finite_policy_lcb",
        "continuous_policy_lcb",
        "finite_grid_variable_lcb",
        "continuous_gp_variable_lcb",
        "continuous_gp_regret_decomposition",
    }
)


def load_policy_lcb_manifest(path: str | Path) -> PolicyLCBManifest:
    """Load either registered policy-LCB manifest adapter."""
    manifest_path = Path(path)
    payload = read_json(manifest_path)
    kind = payload.get("kind")
    if kind == "finite_policy_lcb":
        return parse_finite_policy_lcb_manifest(payload, source_path=manifest_path)
    if kind == "continuous_policy_lcb":
        return parse_continuous_policy_lcb_manifest(payload, source_path=manifest_path)
    if kind == "finite_grid_variable_lcb":
        return parse_variable_finite_grid_lcb_manifest(payload, source_path=manifest_path)
    if kind == "continuous_gp_variable_lcb":
        return parse_continuous_gp_variable_lcb_manifest(payload, source_path=manifest_path)
    if kind == "continuous_gp_regret_decomposition":
        return parse_continuous_gp_decomposition_manifest(payload, source_path=manifest_path)
    raise ValueError(f"Unsupported policy-LCB manifest kind {kind!r}.")


def build_policy_lcb_launch_plan(
    manifest: PolicyLCBManifest,
    *,
    runs_root: str | None,
    force: bool,
) -> LaunchPlan:
    """Build one seed-array launch plan for either policy-LCB adapter."""
    return LaunchPlan(
        name=manifest.name,
        task_count=len(manifest.spec.run_seeds),
        requires_jax=False,
        run_task=lambda index, context: _run_seed(
            manifest,
            index,
            context,
            force=force,
        ),
        run_all=lambda context: _run_serial(manifest, context, force=force),
        collect=lambda context: _collect(manifest, context),
        runs_root=runs_root,
        default_launch=manifest.launch.mode,
        default_array=manifest.launch.array == "seed",
    )


def _run_seed(
    manifest: PolicyLCBManifest,
    index: int,
    context: LaunchContext,
    *,
    force: bool,
) -> dict[str, object]:
    if isinstance(manifest, FinitePolicyLCBManifest):
        return run_finite_policy_lcb_manifest_seed(
            manifest,
            index,
            runs_root=context.runs_root,
            force=force,
        )
    if isinstance(manifest, VariableFiniteGridLCBManifest):
        return run_variable_finite_grid_lcb_manifest_seed(
            manifest,
            index,
            runs_root=context.runs_root,
            force=force,
        )
    if isinstance(manifest, ContinuousGPVariableLCBManifest):
        return run_continuous_gp_variable_lcb_manifest_seed(
            manifest,
            index,
            runs_root=context.runs_root,
            force=force,
        )
    if isinstance(manifest, ContinuousGPDecompositionManifest):
        return run_continuous_gp_decomposition_manifest_seed(
            manifest,
            index,
            runs_root=context.runs_root,
            force=force,
        )
    return run_continuous_policy_lcb_manifest_seed(
        manifest,
        index,
        runs_root=context.runs_root,
        force=force,
    )


def _run_serial(
    manifest: PolicyLCBManifest,
    context: LaunchContext,
    *,
    force: bool,
) -> None:
    if isinstance(manifest, FinitePolicyLCBManifest):
        payload = run_finite_policy_lcb_manifest_serial(
            manifest,
            runs_root=context.runs_root,
            force=force,
        )
        print(
            f"Completed {payload['n_delta_runs']} finite-policy LCB runs under "
            f"{payload['project_dir']} ({payload['n_skipped_seeds']} seeds skipped)."
        )
        return
    if isinstance(manifest, VariableFiniteGridLCBManifest):
        payload = run_variable_finite_grid_lcb_manifest_serial(
            manifest,
            runs_root=context.runs_root,
            force=force,
        )
        print(
            f"Completed {payload['n_executed_condition_rows']} finite-grid conditions "
            f"under {payload['project_dir']} ({payload['n_skipped_seeds']} seeds skipped)."
        )
        return
    if isinstance(manifest, ContinuousGPVariableLCBManifest):
        payload = run_continuous_gp_variable_lcb_manifest_serial(
            manifest,
            runs_root=context.runs_root,
            force=force,
        )
        print(
            f"Completed {payload['n_seed_results']} continuous-GP seeds under "
            f"{payload['project_dir']} ({payload['n_skipped_seeds']} seeds skipped)."
        )
        return
    if isinstance(manifest, ContinuousGPDecompositionManifest):
        payload = run_continuous_gp_decomposition_manifest_serial(
            manifest,
            runs_root=context.runs_root,
            force=force,
        )
        print(
            f"Completed {payload['n_seed_results']} continuous-GP decomposition seeds "
            f"under {payload['project_dir']} ({payload['n_skipped_seeds']} seeds skipped)."
        )
        return
    payload = run_continuous_policy_lcb_manifest_serial(
        manifest,
        runs_root=context.runs_root,
        force=force,
    )
    print(
        f"Completed {payload['n_condition_runs']} continuous policy-LCB runs under "
        f"{payload['project_dir']} ({payload['n_skipped_seeds']} seeds skipped)."
    )


def _collect(manifest: PolicyLCBManifest, context: LaunchContext) -> None:
    if isinstance(manifest, FinitePolicyLCBManifest):
        payload = collect_finite_policy_lcb_outputs(manifest, runs_root=context.runs_root)
        print(
            f"Collected {payload['n_selection_rows']} selections and "
            f"{payload['n_policy_rows']} policy rows under {payload['project_dir']}."
        )
        return
    if isinstance(manifest, VariableFiniteGridLCBManifest):
        payload = collect_variable_finite_grid_lcb_outputs(
            manifest, runs_root=context.runs_root
        )
        print(
            f"Collected {payload['n_condition_rows']} condition rows and "
            f"{payload['n_selector_rows']} selector rows under {payload['project_dir']}."
        )
        return
    if isinstance(manifest, ContinuousGPVariableLCBManifest):
        payload = collect_continuous_gp_variable_lcb_outputs(
            manifest, runs_root=context.runs_root
        )
        print(
            f"Collected {payload['n_selector_rows']} continuous-GP selector rows and "
            f"{payload['n_optimizer_rows']} optimizer rows under {payload['project_dir']}."
        )
        return
    if isinstance(manifest, ContinuousGPDecompositionManifest):
        payload = collect_continuous_gp_decomposition_outputs(
            manifest, runs_root=context.runs_root
        )
        print(
            f"Collected {payload['n_condition_rows']} decomposition conditions and "
            f"{payload['n_best_rows']} best checkpoints under {payload['project_dir']}."
        )
        return
    payload = collect_continuous_policy_lcb_outputs(manifest, runs_root=context.runs_root)
    print(
        f"Collected {payload['n_best_rows']} best results and "
        f"{payload['n_start_rows']} start results under {payload['project_dir']}."
    )


__all__ = [
    "POLICY_LCB_MANIFEST_KINDS",
    "PolicyLCBManifest",
    "build_policy_lcb_launch_plan",
    "load_policy_lcb_manifest",
]
