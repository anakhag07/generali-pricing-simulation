"""Finite-policy adapter for lower-confidence-bound validation and reporting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from experiments.paths import results_root
from experiments.policy_lcb.common import (
    ORACLE_TOLERANCE,
    PolicyLCBLaunchSpec,
    gaussian_lcb_quantile,
    independent_joint_coverage,
    number_sequence,
    path_part,
    read_json,
    required_mapping,
    require_type,
    sample_std,
    wilson_interval,
    write_json_atomic,
)
from experiments.seeds import derive_seed, rng_from_seed
from experiments.sweep_reporting import write_rows_csv


_ORACLE_TOLERANCE = ORACLE_TOLERANCE


@dataclass(frozen=True)
class FinitePolicyLCBSpec:
    """Configuration for the finite-policy Gaussian LCB validation."""

    policies: tuple[float, ...]
    deltas: tuple[float, ...]
    epsilon: float
    master_noise_seed: int
    run_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        policies = tuple(float(value) for value in self.policies)
        if not policies or not np.all(np.isfinite(policies)):
            raise ValueError("policies must be a non-empty finite sequence.")
        if any(value < 0.0 or value > 1.0 for value in policies):
            raise ValueError("policies must lie in [0, 1].")
        if any(right <= left for left, right in zip(policies, policies[1:])):
            raise ValueError("policies must be strictly increasing.")

        deltas = tuple(float(value) for value in self.deltas)
        if not deltas or any(not np.isfinite(value) or not 0.0 < value < 1.0 for value in deltas):
            raise ValueError("deltas must be finite values in (0, 1).")
        if len(set(deltas)) != len(deltas):
            raise ValueError("deltas must be unique.")

        epsilon = float(self.epsilon)
        if not np.isfinite(epsilon) or epsilon != 0.0:
            raise ValueError("The exact finite-policy selector requires epsilon=0.")

        master_noise_seed = int(self.master_noise_seed)
        if master_noise_seed < 0:
            raise ValueError("master_noise_seed must be non-negative.")
        run_seeds = tuple(int(seed) for seed in self.run_seeds)
        if not run_seeds or any(seed < 0 for seed in run_seeds):
            raise ValueError("run_seeds must be a non-empty sequence of non-negative integers.")
        if len(set(run_seeds)) != len(run_seeds):
            raise ValueError("run_seeds must be unique.")

        object.__setattr__(self, "policies", policies)
        object.__setattr__(self, "deltas", deltas)
        object.__setattr__(self, "epsilon", epsilon)
        object.__setattr__(self, "master_noise_seed", master_noise_seed)
        object.__setattr__(self, "run_seeds", run_seeds)


@dataclass(frozen=True)
class FinitePolicyLCBLaunchSpec(PolicyLCBLaunchSpec):
    """Launch settings for the finite-policy LCB manifest."""

    pass


@dataclass(frozen=True)
class FinitePolicyLCBManifest:
    """Resolved finite-policy LCB manifest."""

    name: str
    spec: FinitePolicyLCBSpec
    launch: FinitePolicyLCBLaunchSpec
    source_path: Path | None = None

    def project_dir(self, runs_root: str | Path | None = None) -> Path:
        root = results_root() if runs_root is None else Path(runs_root)
        return root / path_part(self.name)

    def seed_result_path(self, run_seed: int, runs_root: str | Path | None = None) -> Path:
        return self.project_dir(runs_root) / "seeds" / f"seed-{run_seed}" / "result.json"


@dataclass(frozen=True)
class LCBPolicyResult:
    """One policy evaluation for a seed and confidence level."""

    run_seed: int
    noise_seed: int
    delta: float
    quantile: float
    policy: float
    z: float
    true_value: float
    surrogate_value: float
    uncertainty_width: float
    half_width: float
    lcb: float
    policy_covered: bool
    selected: bool


@dataclass(frozen=True)
class LCBSelectionResult:
    """Exact finite-policy selection diagnostics for one seed and delta."""

    run_seed: int
    noise_seed: int
    delta: float
    quantile: float
    selected_policy: float
    selected_true_value: float
    selected_surrogate_value: float
    selected_lcb: float
    selected_uncertainty_width: float
    epsilon: float
    lcb_gap: float
    regret: float
    simultaneous_coverage: bool
    selected_interval_valid: bool
    worst_oracle_slack: float
    optimal_comparator_slack: float
    oracle_violation: bool


@dataclass(frozen=True)
class LCBSeedResult:
    """All paired-delta results produced from one Gaussian noise vector."""

    run_seed: int
    noise_seed: int
    z: tuple[float, ...]
    policy_results: tuple[LCBPolicyResult, ...]
    selections: tuple[LCBSelectionResult, ...]


def load_finite_policy_lcb_manifest(path: str | Path) -> FinitePolicyLCBManifest:
    """Load and validate a ``kind=finite_policy_lcb`` JSON manifest."""
    manifest_path = Path(path)
    payload = read_json(manifest_path)
    return parse_finite_policy_lcb_manifest(payload, source_path=manifest_path)


def parse_finite_policy_lcb_manifest(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> FinitePolicyLCBManifest:
    """Validate a finite-policy LCB manifest payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("Finite-policy LCB manifest must be a JSON object.")
    if payload.get("kind") != "finite_policy_lcb":
        raise ValueError("Finite-policy LCB manifest kind must be 'finite_policy_lcb'.")
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("Finite-policy LCB manifest name must be non-empty.")
    require_type(payload, "true_value", "identity")
    require_type(payload, "surrogate", "policy_scaled_gaussian")

    seeds = required_mapping(payload, "seeds")
    launch_payload = required_mapping(payload, "launch")
    mode = str(launch_payload.get("mode") or "")
    if mode not in {"auto", "local", "slurm"}:
        raise ValueError("launch.mode must be auto, local, or slurm.")
    array = str(launch_payload.get("array") or "")
    if array not in {"none", "seed"}:
        raise ValueError("launch.array must be none or seed.")
    array_max_parallel_raw = launch_payload.get("array_max_parallel")
    array_max_parallel = None if array_max_parallel_raw is None else int(array_max_parallel_raw)
    if array_max_parallel is not None and array_max_parallel <= 0:
        raise ValueError("launch.array_max_parallel must be positive when provided.")

    policies = number_sequence(payload.get("policies"), "policies")
    deltas = number_sequence(payload.get("deltas"), "deltas")
    run_seeds_raw = seeds.get("run_seeds")
    if not isinstance(run_seeds_raw, Sequence) or isinstance(run_seeds_raw, (str, bytes)):
        raise ValueError("seeds.run_seeds must be a sequence.")
    spec = FinitePolicyLCBSpec(
        policies=policies,
        deltas=deltas,
        epsilon=float(payload.get("epsilon", np.nan)),
        master_noise_seed=int(seeds.get("master_noise_seed", -1)),
        run_seeds=tuple(int(seed) for seed in run_seeds_raw),
    )
    return FinitePolicyLCBManifest(
        name=name,
        spec=spec,
        launch=FinitePolicyLCBLaunchSpec(
            mode=mode,  # type: ignore[arg-type]
            array=array,  # type: ignore[arg-type]
            array_max_parallel=array_max_parallel,
        ),
        source_path=None if source_path is None else Path(source_path),
    )


def noise_seed_for_run(spec: FinitePolicyLCBSpec, run_seed: int) -> int:
    """Derive the noise seed for one run without involving delta."""
    if int(run_seed) not in spec.run_seeds:
        raise ValueError(f"Unknown run seed {run_seed}.")
    return derive_seed(spec.master_noise_seed, f"finite-policy-lcb:run-seed:{int(run_seed)}")


def evaluate_finite_policy_lcb_seed(
    spec: FinitePolicyLCBSpec,
    run_seed: int,
) -> LCBSeedResult:
    """Draw one policy-noise vector and evaluate every configured delta."""
    noise_seed = noise_seed_for_run(spec, run_seed)
    z = rng_from_seed(noise_seed).normal(size=len(spec.policies))
    return evaluate_finite_policy_lcb_draw(spec, run_seed=run_seed, noise_seed=noise_seed, z=z)


def evaluate_finite_policy_lcb_draw(
    spec: FinitePolicyLCBSpec,
    *,
    run_seed: int,
    noise_seed: int,
    z: Sequence[float],
) -> LCBSeedResult:
    """Evaluate every delta using an explicitly supplied paired noise vector."""
    policies = np.asarray(spec.policies, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    if z_arr.shape != policies.shape or not np.all(np.isfinite(z_arr)):
        raise ValueError("z must be finite and have one entry per policy.")

    true_values = policies.copy()
    surrogate_values = policies + policies * z_arr
    policy_rows: list[LCBPolicyResult] = []
    selections: list[LCBSelectionResult] = []
    for delta in spec.deltas:
        quantile = lcb_quantile(delta, len(policies))
        half_widths = policies * quantile
        uncertainty_widths = 2.0 * half_widths
        lcbs = surrogate_values - half_widths
        covered = np.abs(surrogate_values - true_values) <= half_widths + _ORACLE_TOLERANCE
        selected_index = int(np.argmax(lcbs))
        selected_true = float(true_values[selected_index])
        comparator_slacks = selected_true - (
            true_values - uncertainty_widths - float(spec.epsilon)
        )
        worst_slack = float(np.min(comparator_slacks))
        exact_gap = float(np.max(lcbs) - lcbs[selected_index])
        simultaneous_coverage = bool(np.all(covered))
        selection = LCBSelectionResult(
            run_seed=int(run_seed),
            noise_seed=int(noise_seed),
            delta=float(delta),
            quantile=quantile,
            selected_policy=float(policies[selected_index]),
            selected_true_value=selected_true,
            selected_surrogate_value=float(surrogate_values[selected_index]),
            selected_lcb=float(lcbs[selected_index]),
            selected_uncertainty_width=float(uncertainty_widths[selected_index]),
            epsilon=float(spec.epsilon),
            lcb_gap=exact_gap,
            regret=float(np.max(true_values) - selected_true),
            simultaneous_coverage=simultaneous_coverage,
            selected_interval_valid=bool(covered[selected_index]),
            worst_oracle_slack=worst_slack,
            optimal_comparator_slack=float(comparator_slacks[-1]),
            oracle_violation=bool(worst_slack < -_ORACLE_TOLERANCE),
        )
        selections.append(selection)
        for index, policy in enumerate(policies):
            policy_rows.append(
                LCBPolicyResult(
                    run_seed=int(run_seed),
                    noise_seed=int(noise_seed),
                    delta=float(delta),
                    quantile=quantile,
                    policy=float(policy),
                    z=float(z_arr[index]),
                    true_value=float(true_values[index]),
                    surrogate_value=float(surrogate_values[index]),
                    uncertainty_width=float(uncertainty_widths[index]),
                    half_width=float(half_widths[index]),
                    lcb=float(lcbs[index]),
                    policy_covered=bool(covered[index]),
                    selected=index == selected_index,
                )
            )
    return LCBSeedResult(
        run_seed=int(run_seed),
        noise_seed=int(noise_seed),
        z=tuple(float(value) for value in z_arr),
        policy_results=tuple(policy_rows),
        selections=tuple(selections),
    )


def lcb_quantile(delta: float, policy_count: int) -> float:
    """Return the simultaneous two-sided Gaussian Bonferroni quantile."""
    return gaussian_lcb_quantile(delta, policy_count)


def analytic_joint_coverage(delta: float, policy_count: int) -> float:
    """Return exact joint coverage for independent policy-level Gaussians."""
    return independent_joint_coverage(delta, policy_count)


def finite_policy_lcb_seed_complete(
    manifest: FinitePolicyLCBManifest,
    run_seed: int,
    *,
    runs_root: str | Path | None = None,
) -> bool:
    """Return whether the durable result for one seed exists."""
    return manifest.seed_result_path(run_seed, runs_root).exists()


def run_finite_policy_lcb_manifest_seed(
    manifest: FinitePolicyLCBManifest,
    index: int,
    *,
    runs_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run one seed task, writing a replayable JSON result."""
    if index < 0 or index >= len(manifest.spec.run_seeds):
        raise IndexError(f"Seed task index {index} is out of range.")
    run_seed = manifest.spec.run_seeds[index]
    write_finite_policy_lcb_experiment_readme(manifest, runs_root=runs_root)
    path = manifest.seed_result_path(run_seed, runs_root)
    if path.exists() and not force:
        return {
            "project_dir": str(manifest.project_dir(runs_root)),
            "run_seed": run_seed,
            "skipped": True,
            "n_delta_runs": 0,
        }
    result = evaluate_finite_policy_lcb_seed(manifest.spec, run_seed)
    _write_seed_result(path, result)
    return {
        "project_dir": str(manifest.project_dir(runs_root)),
        "run_seed": run_seed,
        "skipped": False,
        "n_delta_runs": len(result.selections),
    }


def run_finite_policy_lcb_manifest_serial(
    manifest: FinitePolicyLCBManifest,
    *,
    runs_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run every seed serially and collect aggregate outputs."""
    payloads = [
        run_finite_policy_lcb_manifest_seed(
            manifest,
            index,
            runs_root=runs_root,
            force=force,
        )
        for index in range(len(manifest.spec.run_seeds))
    ]
    collected = collect_finite_policy_lcb_outputs(manifest, runs_root=runs_root)
    return {
        **collected,
        "n_delta_runs": sum(int(payload["n_delta_runs"]) for payload in payloads),
        "n_skipped_seeds": sum(bool(payload["skipped"]) for payload in payloads),
    }


def collect_finite_policy_lcb_outputs(
    manifest: FinitePolicyLCBManifest,
    *,
    runs_root: str | Path | None = None,
) -> dict[str, object]:
    """Collect seed JSON files into tables, diagnostics, and plots."""
    project_dir = manifest.project_dir(runs_root)
    write_finite_policy_lcb_experiment_readme(manifest, runs_root=runs_root)
    seed_results = [
        _read_seed_result(manifest.seed_result_path(run_seed, runs_root))
        for run_seed in manifest.spec.run_seeds
    ]
    policy_rows = [asdict(row) for result in seed_results for row in result.policy_results]
    selection_rows = [asdict(row) for result in seed_results for row in result.selections]
    policy_summary = _policy_summary_rows(manifest.spec, policy_rows)
    coverage_summary = _coverage_summary_rows(manifest.spec, selection_rows)
    oracle_summary = _oracle_summary_rows(manifest.spec, selection_rows)

    write_rows_csv(
        project_dir / "seed_policy_values.csv",
        policy_rows,
        tuple(LCBPolicyResult.__dataclass_fields__),
    )
    write_rows_csv(
        project_dir / "seed_selections.csv",
        selection_rows,
        tuple(LCBSelectionResult.__dataclass_fields__),
    )
    write_rows_csv(
        project_dir / "policy_summary.csv",
        policy_summary,
        tuple(policy_summary[0]),
    )
    write_rows_csv(
        project_dir / "coverage_summary.csv",
        coverage_summary,
        tuple(coverage_summary[0]),
    )
    write_rows_csv(
        project_dir / "oracle_summary.csv",
        oracle_summary,
        tuple(oracle_summary[0]),
    )
    _write_plots(manifest.spec, seed_results, policy_summary, coverage_summary, project_dir)
    return {
        "project_dir": str(project_dir),
        "n_seed_results": len(seed_results),
        "n_policy_rows": len(policy_rows),
        "n_selection_rows": len(selection_rows),
    }


def write_finite_policy_lcb_experiment_readme(
    manifest: FinitePolicyLCBManifest,
    *,
    runs_root: str | Path | None = None,
) -> Path:
    """Write a resolved experiment descriptor beside the sweep outputs."""
    project_dir = manifest.project_dir(runs_root)
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / "EXPERIMENT.md"
    source = str(manifest.source_path) if manifest.source_path is not None else "inline payload"
    text = f"""# {manifest.name}

- Manifest source: `{source}`
- Policies: `{list(manifest.spec.policies)}`
- Deltas: `{list(manifest.spec.deltas)}`
- Script epsilon: `{manifest.spec.epsilon}`
- Master noise seed: `{manifest.spec.master_noise_seed}`
- Run seeds: `{list(manifest.spec.run_seeds)}`
- Launch: `{manifest.launch.mode}` / `{manifest.launch.array}`

For every run seed, one vector of independent standard-normal policy noises is
drawn and reused for every delta. The optimizer evaluates the complete finite
policy class and selects the exact maximum lower confidence bound.

The implemented formulas are

```text
V(pi) = pi
V_hat(pi) = pi + pi * Z(pi)
q(delta) = Phi^-1(1 - delta / (2K))
E(pi, delta) = 2 * pi * q(delta)
LCB(pi, delta) = V_hat(pi) - E(pi, delta) / 2
epsilon = 0
```
"""
    path.write_text(text, encoding="utf-8")
    return path


def _write_seed_result(path: Path, result: LCBSeedResult) -> None:
    write_json_atomic(path, asdict(result))


def _read_seed_result(path: Path) -> LCBSeedResult:
    if not path.exists():
        raise FileNotFoundError(f"Missing finite-policy LCB seed result: {path}")
    payload = read_json(path)
    return LCBSeedResult(
        run_seed=int(payload["run_seed"]),
        noise_seed=int(payload["noise_seed"]),
        z=tuple(float(value) for value in payload["z"]),
        policy_results=tuple(LCBPolicyResult(**row) for row in payload["policy_results"]),
        selections=tuple(LCBSelectionResult(**row) for row in payload["selections"]),
    )


def _policy_summary_rows(
    spec: FinitePolicyLCBSpec,
    policy_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for delta in spec.deltas:
        for policy in spec.policies:
            group = [
                row
                for row in policy_rows
                if float(row["delta"]) == delta and float(row["policy"]) == policy
            ]
            lcbs = np.asarray([float(row["lcb"]) for row in group], dtype=float)
            surrogates = np.asarray([float(row["surrogate_value"]) for row in group], dtype=float)
            selected_count = sum(bool(row["selected"]) for row in group)
            rows.append(
                {
                    "delta": delta,
                    "policy": policy,
                    "true_value": policy,
                    "n_seeds": len(group),
                    "selection_count": selected_count,
                    "selection_rate": selected_count / len(group),
                    "mean_surrogate_value": float(np.mean(surrogates)),
                    "std_surrogate_value": sample_std(surrogates),
                    "mean_lcb": float(np.mean(lcbs)),
                    "std_lcb": sample_std(lcbs),
                }
            )
    return rows


def _coverage_summary_rows(
    spec: FinitePolicyLCBSpec,
    selection_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for delta in spec.deltas:
        group = [row for row in selection_rows if float(row["delta"]) == delta]
        covered = sum(bool(row["simultaneous_coverage"]) for row in group)
        low, high = wilson_interval(covered, len(group))
        rows.append(
            {
                "delta": delta,
                "nominal_coverage": 1.0 - delta,
                "analytic_joint_coverage": analytic_joint_coverage(delta, len(spec.policies)),
                "n_seeds": len(group),
                "covered_count": covered,
                "empirical_coverage": covered / len(group),
                "wilson_95_low": low,
                "wilson_95_high": high,
            }
        )
    return rows


def _oracle_summary_rows(
    spec: FinitePolicyLCBSpec,
    selection_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for delta in spec.deltas:
        group = [row for row in selection_rows if float(row["delta"]) == delta]
        covered_group = [row for row in group if bool(row["simultaneous_coverage"])]
        slacks = np.asarray([float(row["worst_oracle_slack"]) for row in group], dtype=float)
        rows.append(
            {
                "delta": delta,
                "n_seeds": len(group),
                "covered_seed_count": len(covered_group),
                "conditional_violation_count": sum(bool(row["oracle_violation"]) for row in covered_group),
                "unconditional_violation_count": sum(bool(row["oracle_violation"]) for row in group),
                "minimum_worst_slack": float(np.min(slacks)),
                "mean_worst_slack": float(np.mean(slacks)),
                "median_worst_slack": float(np.median(slacks)),
                "worst_slack_q05": float(np.quantile(slacks, 0.05)),
                "worst_slack_q95": float(np.quantile(slacks, 0.95)),
            }
        )
    return rows


def _write_plots(
    spec: FinitePolicyLCBSpec,
    seed_results: Sequence[LCBSeedResult],
    policy_summary: Sequence[Mapping[str, object]],
    coverage_summary: Sequence[Mapping[str, object]],
    project_dir: Path,
) -> None:
    plots_dir = project_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_seed_lcb_panels(spec, seed_results, plots_dir / "seeds")
    _plot_paired_selections(spec, seed_results, plots_dir / "paired_seed_selections.png")
    _plot_selection_frequency(spec, policy_summary, plots_dir / "selection_frequency.png")
    _plot_coverage(spec, coverage_summary, plots_dir / "coverage.png")
    _plot_oracle_slack(spec, seed_results, plots_dir / "oracle_slack.png")
    _plot_validity_usefulness(spec, seed_results, coverage_summary, plots_dir / "validity_vs_usefulness.png")


def _plot_seed_lcb_panels(
    spec: FinitePolicyLCBSpec,
    seed_results: Sequence[LCBSeedResult],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for result in seed_results:
        fig, axis = plt.subplots(figsize=(8.5, 5.5))
        for delta in spec.deltas:
            rows = [row for row in result.policy_results if row.delta == delta]
            axis.plot(
                [row.policy for row in rows],
                [row.lcb for row in rows],
                marker="o",
                linewidth=1.5,
                label=fr"$\delta={delta:g}$",
            )
            selected = next(row for row in rows if row.selected)
            axis.scatter([selected.policy], [selected.lcb], s=45, zorder=5)
        axis.plot(spec.policies, spec.policies, color="black", linestyle="--", label=r"$V^\pi=\pi$")
        axis.axhline(0.0, color="0.6", linewidth=0.8)
        axis.set_xlabel(r"Policy $\pi$")
        axis.set_ylabel("Value / lower confidence bound")
        axis.set_title(f"Paired LCBs for noise seed {result.run_seed}")
        axis.legend(ncol=2, fontsize=8)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"seed-{result.run_seed}.png", dpi=160)
        plt.close(fig)


def _plot_paired_selections(
    spec: FinitePolicyLCBSpec,
    seed_results: Sequence[LCBSeedResult],
    path: Path,
) -> None:
    x = np.arange(len(spec.deltas))
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    for result in seed_results:
        selected = [selection.selected_policy for selection in result.selections]
        axis.plot(x, selected, marker="o", linewidth=0.9, alpha=0.45)
    axis.set_xticks(x, [f"{delta:g}" for delta in spec.deltas])
    axis.set_xlabel(r"Failure probability $\delta$")
    axis.set_ylabel(r"Selected policy $\widehat\pi$")
    axis.set_title("Paired seed paths as confidence strengthens")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_selection_frequency(
    spec: FinitePolicyLCBSpec,
    policy_summary: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    x = np.arange(len(spec.deltas))
    fig, axis = plt.subplots(figsize=(9, 5.8))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(spec.policies)))
    for policy, color in zip(spec.policies, colors):
        rows = [row for row in policy_summary if float(row["policy"]) == policy]
        axis.plot(
            x,
            [float(row["selection_rate"]) for row in rows],
            marker="o",
            color=color,
            label=f"{policy:g}",
        )
    axis.set_xticks(x, [f"{delta:g}" for delta in spec.deltas])
    axis.set_xlabel(r"Failure probability $\delta$")
    axis.set_ylabel("Selection frequency")
    axis.set_title("Exact maximum-LCB policy frequencies")
    axis.legend(title=r"$\pi$", ncol=4, fontsize=8)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_coverage(
    spec: FinitePolicyLCBSpec,
    coverage_summary: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    x = np.arange(len(spec.deltas))
    empirical = np.asarray([float(row["empirical_coverage"]) for row in coverage_summary])
    low = np.asarray([float(row["wilson_95_low"]) for row in coverage_summary])
    high = np.asarray([float(row["wilson_95_high"]) for row in coverage_summary])
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.plot(x, [1.0 - delta for delta in spec.deltas], marker="o", label=r"Nominal $1-\delta$")
    axis.plot(
        x,
        [analytic_joint_coverage(delta, len(spec.policies)) for delta in spec.deltas],
        marker="s",
        label="Exact joint coverage",
    )
    axis.errorbar(
        x,
        empirical,
        # Wilson endpoints can differ from 0/1 by a final floating-point ulp.
        yerr=np.maximum(0.0, np.vstack([empirical - low, high - empirical])),
        marker="^",
        capsize=4,
        label="Empirical (Wilson 95%)",
    )
    axis.set_xticks(x, [f"{delta:g}" for delta in spec.deltas])
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel(r"Failure probability $\delta$")
    axis.set_ylabel("Simultaneous coverage")
    axis.set_title("Lower-confidence-bound coverage")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_oracle_slack(
    spec: FinitePolicyLCBSpec,
    seed_results: Sequence[LCBSeedResult],
    path: Path,
) -> None:
    data = [
        [selection.worst_oracle_slack for result in seed_results for selection in result.selections if selection.delta == delta]
        for delta in spec.deltas
    ]
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.boxplot(data, tick_labels=[f"{delta:g}" for delta in spec.deltas], showmeans=True)
    axis.axhline(0.0, color="red", linestyle="--", linewidth=1.0, label="Violation threshold")
    axis.set_xlabel(r"Failure probability $\delta$")
    axis.set_ylabel("Worst comparator oracle slack")
    axis.set_title(r"Proposition 11.2 with $\varepsilon=0$")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_validity_usefulness(
    spec: FinitePolicyLCBSpec,
    seed_results: Sequence[LCBSeedResult],
    coverage_summary: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    mean_selected = []
    mean_regret = []
    for delta in spec.deltas:
        selections = [
            selection
            for result in seed_results
            for selection in result.selections
            if selection.delta == delta
        ]
        mean_selected.append(float(np.mean([row.selected_true_value for row in selections])))
        mean_regret.append(float(np.mean([row.regret for row in selections])))
    x = np.arange(len(spec.deltas))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].plot(
        x,
        [float(row["empirical_coverage"]) for row in coverage_summary],
        marker="o",
        label="Empirical validity",
    )
    axes[0].plot(x, [1.0 - delta for delta in spec.deltas], linestyle="--", label="Nominal")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Coverage")
    axes[0].legend()
    axes[1].plot(x, mean_selected, marker="o", label="Selected true value")
    axes[1].plot(x, mean_regret, marker="s", label="True regret")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    for axis in axes:
        axis.set_xticks(x, [f"{delta:g}" for delta in spec.deltas])
        axis.set_xlabel(r"Failure probability $\delta$")
        axis.grid(alpha=0.25)
    fig.suptitle("Confidence validity versus decision usefulness")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


_sample_std = sample_std
_wilson_interval = wilson_interval


__all__ = [
    "FinitePolicyLCBLaunchSpec",
    "FinitePolicyLCBManifest",
    "FinitePolicyLCBSpec",
    "LCBPolicyResult",
    "LCBSeedResult",
    "LCBSelectionResult",
    "analytic_joint_coverage",
    "collect_finite_policy_lcb_outputs",
    "evaluate_finite_policy_lcb_draw",
    "evaluate_finite_policy_lcb_seed",
    "finite_policy_lcb_seed_complete",
    "lcb_quantile",
    "load_finite_policy_lcb_manifest",
    "noise_seed_for_run",
    "parse_finite_policy_lcb_manifest",
    "run_finite_policy_lcb_manifest_seed",
    "run_finite_policy_lcb_manifest_serial",
    "write_finite_policy_lcb_experiment_readme",
]
