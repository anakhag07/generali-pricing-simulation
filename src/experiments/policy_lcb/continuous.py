"""Continuous shared-Gaussian policy-LCB optimization adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
import csv
from pathlib import Path
from typing import Any, Literal

import numpy as np

from experiments.paths import results_root
from experiments.policy_lcb.common import (
    ORACLE_TOLERANCE,
    PolicyLCBLaunchSpec,
    gaussian_lcb_quantile,
    number_sequence,
    path_part,
    read_json,
    required_mapping,
    require_type,
    shared_gaussian_coverage,
    write_json_atomic,
)
from experiments.seeds import derive_seed, rng_from_seed
from experiments.sweep_reporting import write_rows_csv
from optimization.helpers import finite_difference_theta_grad, stein_difference_theta_grad


ContinuousEstimator = Literal["first_order", "finite_difference", "stein_difference"]
_ALLOWED_ESTIMATORS = {"first_order", "finite_difference", "stein_difference"}


@dataclass(frozen=True)
class ContinuousPolicyValueSpec:
    """True-value curve used by the continuous policy-LCB adapter."""

    kind: Literal["identity", "concave_quadratic"] = "identity"
    a: float = 1.0
    b: float = 0.0

    def __post_init__(self) -> None:
        a = float(self.a)
        b = float(self.b)
        if not np.isfinite(a) or not np.isfinite(b):
            raise ValueError("true_value coefficients a and b must be finite.")
        if self.kind == "identity":
            if a != 1.0 or b != 0.0:
                raise ValueError("identity true_value requires a=1 and b=0.")
        elif self.kind == "concave_quadratic":
            if a <= 0.0 or b <= 0.0:
                raise ValueError("concave_quadratic true_value requires a>0 and b>0.")
        else:
            raise ValueError("true_value.type must be identity or concave_quadratic.")
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "b", b)


@dataclass(frozen=True)
class ContinuousPolicyLCBOptimizerSpec:
    """Projected scalar optimizer settings for a continuous policy-LCB run."""

    step_rule: str
    enabled_estimators: tuple[ContinuousEstimator, ...]
    starts: tuple[float, ...]
    t_steps: int
    step_size: float
    sigma: float
    n_grad_samples: int

    def __post_init__(self) -> None:
        if self.step_rule != "projected_constant":
            raise ValueError("optimizer.step_rule must be 'projected_constant'.")
        estimators = tuple(str(value) for value in self.enabled_estimators)
        if not estimators or len(set(estimators)) != len(estimators):
            raise ValueError("optimizer.enabled_estimators must be non-empty and unique.")
        unknown = set(estimators) - _ALLOWED_ESTIMATORS
        if unknown:
            raise ValueError(f"Unknown continuous policy-LCB estimators: {sorted(unknown)}")
        starts = tuple(float(value) for value in self.starts)
        if not starts or not np.all(np.isfinite(starts)):
            raise ValueError("optimizer.starts must be a non-empty finite sequence.")
        if int(self.t_steps) <= 0:
            raise ValueError("optimizer.t_steps must be positive.")
        if not np.isfinite(self.step_size) or float(self.step_size) <= 0.0:
            raise ValueError("optimizer.step_size must be positive.")
        if not np.isfinite(self.sigma) or float(self.sigma) <= 0.0:
            raise ValueError("optimizer.sigma must be positive.")
        if int(self.n_grad_samples) <= 0:
            raise ValueError("optimizer.n_grad_samples must be positive.")
        object.__setattr__(self, "enabled_estimators", estimators)
        object.__setattr__(self, "starts", starts)
        object.__setattr__(self, "t_steps", int(self.t_steps))
        object.__setattr__(self, "step_size", float(self.step_size))
        object.__setattr__(self, "sigma", float(self.sigma))
        object.__setattr__(self, "n_grad_samples", int(self.n_grad_samples))


@dataclass(frozen=True)
class ContinuousPolicyLCBSpec:
    """Configuration for shared-Gaussian LCB optimization over an interval."""

    policy_domain: tuple[float, float]
    deltas: tuple[float, ...]
    master_noise_seed: int
    master_optimizer_seed: int
    reporting_seed: int
    run_seeds: tuple[int, ...]
    optimizer: ContinuousPolicyLCBOptimizerSpec
    true_value: ContinuousPolicyValueSpec = field(default_factory=ContinuousPolicyValueSpec)

    def __post_init__(self) -> None:
        domain = tuple(float(value) for value in self.policy_domain)
        if domain != (0.0, 1.0):
            raise ValueError("policy_domain must be exactly [0, 1].")
        deltas = tuple(float(value) for value in self.deltas)
        if not deltas or any(not np.isfinite(value) or not 0.0 < value < 1.0 for value in deltas):
            raise ValueError("deltas must be finite values in (0, 1).")
        if len(set(deltas)) != len(deltas):
            raise ValueError("deltas must be unique.")
        run_seeds = tuple(int(seed) for seed in self.run_seeds)
        if not run_seeds or any(seed < 0 for seed in run_seeds):
            raise ValueError("run_seeds must be a non-empty sequence of non-negative integers.")
        if len(set(run_seeds)) != len(run_seeds):
            raise ValueError("run_seeds must be unique.")
        seed_values = (self.master_noise_seed, self.master_optimizer_seed, self.reporting_seed)
        if any(int(seed) < 0 for seed in seed_values):
            raise ValueError("master and reporting seeds must be non-negative.")
        if any(start < domain[0] or start > domain[1] for start in self.optimizer.starts):
            raise ValueError("optimizer.starts must lie in policy_domain.")
        object.__setattr__(self, "policy_domain", domain)
        object.__setattr__(self, "deltas", deltas)
        object.__setattr__(self, "master_noise_seed", int(self.master_noise_seed))
        object.__setattr__(self, "master_optimizer_seed", int(self.master_optimizer_seed))
        object.__setattr__(self, "reporting_seed", int(self.reporting_seed))
        object.__setattr__(self, "run_seeds", run_seeds)


@dataclass(frozen=True)
class ContinuousPolicyLCBLaunchSpec(PolicyLCBLaunchSpec):
    """Launch settings for the continuous policy-LCB manifest."""

    pass


@dataclass(frozen=True)
class ContinuousPolicyLCBManifest:
    """Resolved continuous policy-LCB manifest."""

    name: str
    spec: ContinuousPolicyLCBSpec
    launch: ContinuousPolicyLCBLaunchSpec
    source_path: Path | None = None

    def project_dir(self, runs_root: str | Path | None = None) -> Path:
        root = results_root() if runs_root is None else Path(runs_root)
        return root / path_part(self.name)

    def seed_dir(self, run_seed: int, runs_root: str | Path | None = None) -> Path:
        return self.project_dir(runs_root) / "seeds" / f"seed-{run_seed}"

    def seed_result_path(self, run_seed: int, runs_root: str | Path | None = None) -> Path:
        return self.seed_dir(run_seed, runs_root) / "result.json"

    def seed_trajectory_path(self, run_seed: int, runs_root: str | Path | None = None) -> Path:
        return self.seed_dir(run_seed, runs_root) / "trajectories.csv"


@dataclass(frozen=True)
class ContinuousLCBTrajectoryRow:
    """One projected optimization trajectory point."""

    run_seed: int
    noise_seed: int
    delta: float
    estimator: str
    start_policy: float
    step: int
    policy: float
    loss: float
    lcb: float
    gradient_estimate: float | None


@dataclass(frozen=True)
class ContinuousLCBStartResult:
    """Final diagnostics for one estimator and starting policy."""

    run_seed: int
    noise_seed: int
    stein_seed: int
    z: float
    delta: float
    quantile: float
    estimator: str
    start_policy: float
    final_policy: float
    final_true_value: float
    final_surrogate_value: float
    final_lcb: float
    final_loss: float
    analytic_policy: float
    analytic_lcb: float
    analytic_loss: float
    optimization_error: float
    regret: float
    n_steps: int
    converged: bool
    simultaneous_coverage: bool
    selected_interval_valid: bool
    worst_oracle_slack: float
    optimal_comparator_slack: float
    oracle_violation: bool


@dataclass(frozen=True)
class ContinuousLCBBestResult:
    """Best result across paired starts for one seed, delta, and estimator."""

    run_seed: int
    noise_seed: int
    stein_seed: int
    z: float
    delta: float
    quantile: float
    estimator: str
    selected_start_policy: float
    final_policy: float
    final_true_value: float
    final_surrogate_value: float
    final_lcb: float
    final_loss: float
    analytic_policy: float
    analytic_lcb: float
    analytic_loss: float
    optimization_error: float
    regret: float
    n_steps: int
    converged: bool
    simultaneous_coverage: bool
    selected_interval_valid: bool
    worst_oracle_slack: float
    optimal_comparator_slack: float
    oracle_violation: bool


@dataclass(frozen=True)
class ContinuousLCBSeedResult:
    """All continuous optimizer results produced from one shared Gaussian draw."""

    run_seed: int
    noise_seed: int
    stein_seed: int
    z: float
    start_results: tuple[ContinuousLCBStartResult, ...]
    best_results: tuple[ContinuousLCBBestResult, ...]
    trajectories: tuple[ContinuousLCBTrajectoryRow, ...]


def load_continuous_policy_lcb_manifest(path: str | Path) -> ContinuousPolicyLCBManifest:
    """Load and validate a ``kind=continuous_policy_lcb`` JSON manifest."""
    manifest_path = Path(path)
    return parse_continuous_policy_lcb_manifest(
        read_json(manifest_path),
        source_path=manifest_path,
    )


def parse_continuous_policy_lcb_manifest(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
) -> ContinuousPolicyLCBManifest:
    """Validate a continuous policy-LCB manifest payload."""
    if not isinstance(payload, Mapping):
        raise ValueError("Continuous policy-LCB manifest must be a JSON object.")
    if payload.get("kind") != "continuous_policy_lcb":
        raise ValueError("Continuous policy-LCB manifest kind must be 'continuous_policy_lcb'.")
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("Continuous policy-LCB manifest name must be non-empty.")
    true_value_payload = required_mapping(payload, "true_value")
    true_value_kind = str(true_value_payload.get("type") or "")
    if true_value_kind == "identity":
        true_value = ContinuousPolicyValueSpec()
    elif true_value_kind == "concave_quadratic":
        true_value = ContinuousPolicyValueSpec(
            kind="concave_quadratic",
            a=float(true_value_payload.get("a", np.nan)),
            b=float(true_value_payload.get("b", np.nan)),
        )
    else:
        raise ValueError("true_value.type must be identity or concave_quadratic.")
    require_type(payload, "surrogate", "shared_policy_scaled_gaussian")
    domain = number_sequence(payload.get("policy_domain"), "policy_domain")
    if len(domain) != 2:
        raise ValueError("policy_domain must contain exactly two endpoints.")
    deltas = number_sequence(payload.get("deltas"), "deltas")

    seeds = required_mapping(payload, "seeds")
    run_seeds_raw = seeds.get("run_seeds")
    if not isinstance(run_seeds_raw, Sequence) or isinstance(run_seeds_raw, (str, bytes)):
        raise ValueError("seeds.run_seeds must be a sequence.")

    optimizer_payload = required_mapping(payload, "optimizer")
    estimators_raw = optimizer_payload.get("enabled_estimators")
    if not isinstance(estimators_raw, Sequence) or isinstance(estimators_raw, (str, bytes)):
        raise ValueError("optimizer.enabled_estimators must be a sequence.")
    starts = number_sequence(optimizer_payload.get("starts"), "optimizer.starts")
    optimizer = ContinuousPolicyLCBOptimizerSpec(
        step_rule=str(optimizer_payload.get("step_rule") or ""),
        enabled_estimators=tuple(str(value) for value in estimators_raw),  # type: ignore[arg-type]
        starts=starts,
        t_steps=int(optimizer_payload.get("t_steps", 0)),
        step_size=float(optimizer_payload.get("step_size", np.nan)),
        sigma=float(optimizer_payload.get("sigma", np.nan)),
        n_grad_samples=int(optimizer_payload.get("n_grad_samples", 0)),
    )

    launch_payload = required_mapping(payload, "launch")
    mode = str(launch_payload.get("mode") or "")
    if mode not in {"auto", "local", "slurm"}:
        raise ValueError("launch.mode must be auto, local, or slurm.")
    array = str(launch_payload.get("array") or "")
    if array not in {"none", "seed"}:
        raise ValueError("launch.array must be none or seed.")
    maximum_raw = launch_payload.get("array_max_parallel")
    maximum = None if maximum_raw is None else int(maximum_raw)
    if maximum is not None and maximum <= 0:
        raise ValueError("launch.array_max_parallel must be positive when provided.")

    spec = ContinuousPolicyLCBSpec(
        policy_domain=(domain[0], domain[1]),
        deltas=deltas,
        master_noise_seed=int(seeds.get("master_noise_seed", -1)),
        master_optimizer_seed=int(seeds.get("master_optimizer_seed", -1)),
        reporting_seed=int(seeds.get("reporting_seed", -1)),
        run_seeds=tuple(int(seed) for seed in run_seeds_raw),
        optimizer=optimizer,
        true_value=true_value,
    )
    return ContinuousPolicyLCBManifest(
        name=name,
        spec=spec,
        launch=ContinuousPolicyLCBLaunchSpec(
            mode=mode,  # type: ignore[arg-type]
            array=array,  # type: ignore[arg-type]
            array_max_parallel=maximum,
        ),
        source_path=None if source_path is None else Path(source_path),
    )


def continuous_lcb_quantile(delta: float) -> float:
    """Return the exact two-sided quantile for one shared Gaussian draw."""
    return gaussian_lcb_quantile(delta)


def continuous_true_value(
    policy: float,
    value_spec: ContinuousPolicyValueSpec | None = None,
) -> float:
    r"""Return $$V(\pi)=a\pi-b\pi^2$$ for the configured value curve."""
    value = ContinuousPolicyValueSpec() if value_spec is None else value_spec
    policy_value = float(policy)
    return value.a * policy_value - value.b * policy_value**2


def continuous_surrogate_value(
    policy: float,
    z: float,
    value_spec: ContinuousPolicyValueSpec | None = None,
) -> float:
    r"""Return the shared-Gaussian surrogate $$V(\pi)+\pi Z_s$$."""
    policy_value = float(policy)
    return continuous_true_value(policy_value, value_spec) + policy_value * float(z)


def continuous_lcb_loss(
    policy: float,
    z: float,
    delta: float,
    value_spec: ContinuousPolicyValueSpec | None = None,
) -> float:
    """Return negative LCB for a shared-Gaussian continuous value curve."""
    policy_value = float(policy)
    return (
        -continuous_surrogate_value(policy_value, z, value_spec)
        + policy_value * continuous_lcb_quantile(delta)
    )


def continuous_lcb_slope(z: float, delta: float) -> float:
    """Return the constant negative-LCB slope for the legacy identity curve."""
    return continuous_lcb_quantile(delta) - 1.0 - float(z)


def continuous_lcb_gradient(
    policy: float,
    z: float,
    delta: float,
    value_spec: ContinuousPolicyValueSpec | None = None,
) -> float:
    """Return the exact policy derivative of the negative LCB."""
    value = ContinuousPolicyValueSpec() if value_spec is None else value_spec
    return 2.0 * value.b * float(policy) + continuous_lcb_quantile(delta) - value.a - float(z)


def continuous_lcb_value(
    policy: float,
    z: float,
    delta: float,
    value_spec: ContinuousPolicyValueSpec | None = None,
) -> float:
    """Return the lower confidence bound, the negative of the optimized loss."""
    return -continuous_lcb_loss(policy, z, delta, value_spec)


def continuous_analytic_policy(
    z: float,
    delta: float,
    value_spec: ContinuousPolicyValueSpec | None = None,
) -> float:
    """Return the smallest exact negative-LCB minimizer over $$[0,1]$$."""
    value = ContinuousPolicyValueSpec() if value_spec is None else value_spec
    if value.b == 0.0:
        return 0.0 if continuous_lcb_gradient(0.0, z, delta, value) >= 0.0 else 1.0
    unconstrained = (value.a + float(z) - continuous_lcb_quantile(delta)) / (2.0 * value.b)
    return float(np.clip(unconstrained, 0.0, 1.0))


def continuous_true_optimal_policy(
    value_spec: ContinuousPolicyValueSpec | None = None,
) -> float:
    """Return the smallest exact maximizer of the true value over $$[0,1]$$."""
    value = ContinuousPolicyValueSpec() if value_spec is None else value_spec
    if value.b == 0.0:
        return 0.0 if value.a <= 0.0 else 1.0
    return float(np.clip(value.a / (2.0 * value.b), 0.0, 1.0))


def continuous_noise_seed_for_run(spec: ContinuousPolicyLCBSpec, run_seed: int) -> int:
    """Derive one problem-noise seed without involving delta or estimator."""
    if int(run_seed) not in spec.run_seeds:
        raise ValueError(f"Unknown run seed {run_seed}.")
    return derive_seed(
        spec.master_noise_seed,
        f"continuous-policy-lcb:run-seed:{int(run_seed)}",
    )


def continuous_stein_seed(spec: ContinuousPolicyLCBSpec) -> int:
    """Derive the fixed paired Stein stream shared by all problem draws."""
    return derive_seed(spec.master_optimizer_seed, "continuous-policy-lcb:stein-difference")


def evaluate_continuous_policy_lcb_seed(
    spec: ContinuousPolicyLCBSpec,
    run_seed: int,
) -> ContinuousLCBSeedResult:
    """Draw one shared Gaussian and run every configured optimizer condition."""
    noise_seed = continuous_noise_seed_for_run(spec, run_seed)
    z = float(rng_from_seed(noise_seed).normal())
    return evaluate_continuous_policy_lcb_draw(
        spec,
        run_seed=run_seed,
        noise_seed=noise_seed,
        z=z,
    )


def evaluate_continuous_policy_lcb_draw(
    spec: ContinuousPolicyLCBSpec,
    *,
    run_seed: int,
    noise_seed: int,
    z: float,
) -> ContinuousLCBSeedResult:
    """Run all paired conditions for an explicitly supplied shared Gaussian draw."""
    z_value = float(z)
    if not np.isfinite(z_value):
        raise ValueError("z must be finite.")
    stein_seed = continuous_stein_seed(spec)
    starts: list[ContinuousLCBStartResult] = []
    trajectories: list[ContinuousLCBTrajectoryRow] = []
    best: list[ContinuousLCBBestResult] = []
    for delta in spec.deltas:
        for estimator in spec.optimizer.enabled_estimators:
            group: list[ContinuousLCBStartResult] = []
            for start_policy in spec.optimizer.starts:
                result, trace = _run_continuous_start(
                    spec,
                    run_seed=int(run_seed),
                    noise_seed=int(noise_seed),
                    stein_seed=stein_seed,
                    z=z_value,
                    delta=delta,
                    estimator=estimator,
                    start_policy=start_policy,
                )
                starts.append(result)
                group.append(result)
                trajectories.extend(trace)
            selected = min(
                group,
                key=lambda row: (row.final_loss, row.final_policy, row.start_policy),
            )
            best.append(_best_from_start(selected))
    return ContinuousLCBSeedResult(
        run_seed=int(run_seed),
        noise_seed=int(noise_seed),
        stein_seed=stein_seed,
        z=z_value,
        start_results=tuple(starts),
        best_results=tuple(best),
        trajectories=tuple(trajectories),
    )


def continuous_policy_lcb_seed_complete(
    manifest: ContinuousPolicyLCBManifest,
    run_seed: int,
    *,
    runs_root: str | Path | None = None,
) -> bool:
    """Return whether both durable outputs exist and match the current manifest."""
    result_path = manifest.seed_result_path(run_seed, runs_root)
    trajectory_path = manifest.seed_trajectory_path(run_seed, runs_root)
    if not result_path.exists() or not trajectory_path.exists():
        return False
    try:
        payload = read_json(result_path)
    except (OSError, ValueError):
        return False
    spec = manifest.spec
    return bool(
        payload.get("model") == _model_metadata(spec)
        and payload.get("optimizer") == _optimizer_metadata(spec)
        and payload.get("seed_contract") == _seed_contract_metadata(spec)
        and payload.get("run_seed") == int(run_seed)
        and payload.get("noise_seed") == continuous_noise_seed_for_run(spec, run_seed)
        and payload.get("stein_seed") == continuous_stein_seed(spec)
    )


def run_continuous_policy_lcb_manifest_seed(
    manifest: ContinuousPolicyLCBManifest,
    index: int,
    *,
    runs_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run one shared-Gaussian seed task and persist its summary and traces."""
    if index < 0 or index >= len(manifest.spec.run_seeds):
        raise IndexError(f"Seed task index {index} is out of range.")
    run_seed = manifest.spec.run_seeds[index]
    write_continuous_policy_lcb_experiment_readme(manifest, runs_root=runs_root)
    if continuous_policy_lcb_seed_complete(manifest, run_seed, runs_root=runs_root) and not force:
        return {
            "project_dir": str(manifest.project_dir(runs_root)),
            "run_seed": run_seed,
            "skipped": True,
            "n_condition_runs": 0,
        }
    result = evaluate_continuous_policy_lcb_seed(manifest.spec, run_seed)
    _write_seed_trajectories(manifest.seed_trajectory_path(run_seed, runs_root), result)
    _write_seed_result(manifest.seed_result_path(run_seed, runs_root), result, manifest.spec)
    return {
        "project_dir": str(manifest.project_dir(runs_root)),
        "run_seed": run_seed,
        "skipped": False,
        "n_condition_runs": len(result.start_results),
    }


def run_continuous_policy_lcb_manifest_serial(
    manifest: ContinuousPolicyLCBManifest,
    *,
    runs_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, object]:
    """Run every continuous seed serially and collect aggregate outputs."""
    payloads = [
        run_continuous_policy_lcb_manifest_seed(
            manifest,
            index,
            runs_root=runs_root,
            force=force,
        )
        for index in range(len(manifest.spec.run_seeds))
    ]
    collected = collect_continuous_policy_lcb_outputs(manifest, runs_root=runs_root)
    return {
        **collected,
        "n_condition_runs": sum(int(payload["n_condition_runs"]) for payload in payloads),
        "n_skipped_seeds": sum(bool(payload["skipped"]) for payload in payloads),
    }


def collect_continuous_policy_lcb_outputs(
    manifest: ContinuousPolicyLCBManifest,
    *,
    runs_root: str | Path | None = None,
) -> dict[str, object]:
    """Collect seed outputs into continuous optimizer tables and plots."""
    from experiments.policy_lcb.continuous_reporting import (
        coverage_summary_rows,
        optimizer_summary_rows,
        oracle_summary_rows,
        write_continuous_policy_lcb_plots,
    )

    project_dir = manifest.project_dir(runs_root)
    write_continuous_policy_lcb_experiment_readme(manifest, runs_root=runs_root)
    seed_results = [
        _read_seed_result(
            manifest.seed_result_path(run_seed, runs_root),
            manifest.seed_trajectory_path(run_seed, runs_root),
        )
        for run_seed in manifest.spec.run_seeds
    ]
    draw_rows = [
        {
            "run_seed": result.run_seed,
            "noise_seed": result.noise_seed,
            "stein_seed": result.stein_seed,
            "z": result.z,
        }
        for result in seed_results
    ]
    start_rows = [asdict(row) for result in seed_results for row in result.start_results]
    best_rows = [asdict(row) for result in seed_results for row in result.best_results]
    optimizer_rows = optimizer_summary_rows(manifest.spec, seed_results)
    coverage_rows = coverage_summary_rows(manifest.spec, seed_results)
    oracle_rows = oracle_summary_rows(manifest.spec, seed_results)

    write_rows_csv(project_dir / "seed_draws.csv", draw_rows, tuple(draw_rows[0]))
    write_rows_csv(
        project_dir / "seed_start_results.csv",
        start_rows,
        tuple(ContinuousLCBStartResult.__dataclass_fields__),
    )
    write_rows_csv(
        project_dir / "seed_best_results.csv",
        best_rows,
        tuple(ContinuousLCBBestResult.__dataclass_fields__),
    )
    write_rows_csv(
        project_dir / "optimizer_summary.csv",
        optimizer_rows,
        tuple(optimizer_rows[0]),
    )
    write_rows_csv(
        project_dir / "coverage_summary.csv",
        coverage_rows,
        tuple(coverage_rows[0]),
    )
    write_rows_csv(
        project_dir / "oracle_summary.csv",
        oracle_rows,
        tuple(oracle_rows[0]),
    )
    write_continuous_policy_lcb_plots(
        manifest.spec,
        seed_results,
        optimizer_rows,
        coverage_rows,
        project_dir,
    )
    return {
        "project_dir": str(project_dir),
        "n_seed_results": len(seed_results),
        "n_start_rows": len(start_rows),
        "n_best_rows": len(best_rows),
    }


def write_continuous_policy_lcb_experiment_readme(
    manifest: ContinuousPolicyLCBManifest,
    *,
    runs_root: str | Path | None = None,
) -> Path:
    """Write the resolved continuous policy-LCB experiment descriptor."""
    project_dir = manifest.project_dir(runs_root)
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / "EXPERIMENT.md"
    spec = manifest.spec
    optimizer = spec.optimizer
    source = str(manifest.source_path) if manifest.source_path is not None else "inline payload"
    text = f"""# {manifest.name}

- Manifest source: `{source}`
- Policy domain: `{list(spec.policy_domain)}`
- Deltas: `{list(spec.deltas)}`
- Estimators: `{list(optimizer.enabled_estimators)}`
- Starts: `{list(optimizer.starts)}`
- Projected steps / step size: `{optimizer.t_steps}` / `{optimizer.step_size}`
- Perturbation radius / Stein samples: `{optimizer.sigma}` / `{optimizer.n_grad_samples}`
- Master problem-noise seed: `{spec.master_noise_seed}`
- Fixed paired optimizer seed: `{spec.master_optimizer_seed}`
- Reporting seed: `{spec.reporting_seed}`
- Run seeds: `{list(spec.run_seeds)}`
- Launch: `{manifest.launch.mode}` / `{manifest.launch.array}`
- True value: `{spec.true_value.kind}` with `a={spec.true_value.a:g}`, `b={spec.true_value.b:g}`

Each run seed draws one scalar standard normal `Z_s`. The draw is reused for
every policy in `[0, 1]`, confidence level, estimator, and start. The separate
Stein perturbation stream is deliberately reinitialized identically for every
condition so seed-level spread isolates the problem draws.

```text
V(pi) = {spec.true_value.a:g} * pi - {spec.true_value.b:g} * pi^2
V_hat_s(pi) = V(pi) + pi * Z_s
q(delta) = Phi^-1(1 - delta / 2)
E(pi, delta) = 2 * pi * q(delta)
loss(pi, delta, s) = -LCB
  = {spec.true_value.b:g} * pi^2 + (q(delta) - {spec.true_value.a:g} - Z_s) * pi
```

The absence of a Bonferroni factor is caused by the shared scalar `Z_s`, not by
continuity alone. For every positive policy, the simultaneous standardized
error event reduces to the same event `abs(Z_s) <= q(delta)`.
"""
    path.write_text(text, encoding="utf-8")
    return path


def _write_seed_result(
    path: Path,
    result: ContinuousLCBSeedResult,
    spec: ContinuousPolicyLCBSpec,
) -> None:
    write_json_atomic(
        path,
        {
            "model": _model_metadata(spec),
            "optimizer": _optimizer_metadata(spec),
            "seed_contract": _seed_contract_metadata(spec),
            "run_seed": result.run_seed,
            "noise_seed": result.noise_seed,
            "stein_seed": result.stein_seed,
            "z": result.z,
            "start_results": [asdict(row) for row in result.start_results],
            "best_results": [asdict(row) for row in result.best_results],
        },
    )


def _model_metadata(spec: ContinuousPolicyLCBSpec) -> dict[str, object]:
    return {
        "true_value": {
            "type": spec.true_value.kind,
            "a": spec.true_value.a,
            "b": spec.true_value.b,
        },
        "surrogate": "shared_policy_scaled_gaussian",
        "optimized_quantity": "negative_lcb",
        "policy_domain": list(spec.policy_domain),
        "deltas": list(spec.deltas),
    }


def _optimizer_metadata(spec: ContinuousPolicyLCBSpec) -> dict[str, object]:
    optimizer = spec.optimizer
    return {
        "step_rule": optimizer.step_rule,
        "enabled_estimators": list(optimizer.enabled_estimators),
        "starts": list(optimizer.starts),
        "t_steps": optimizer.t_steps,
        "step_size": optimizer.step_size,
        "sigma": optimizer.sigma,
        "n_grad_samples": optimizer.n_grad_samples,
    }


def _seed_contract_metadata(spec: ContinuousPolicyLCBSpec) -> dict[str, object]:
    return {
        "master_noise_seed": spec.master_noise_seed,
        "master_optimizer_seed": spec.master_optimizer_seed,
        "reporting_seed": spec.reporting_seed,
        "stein_stream_varies_across_run_seeds": False,
    }


def _write_seed_trajectories(path: Path, result: ContinuousLCBSeedResult) -> None:
    rows = [asdict(row) for row in result.trajectories]
    write_rows_csv(path, rows, tuple(ContinuousLCBTrajectoryRow.__dataclass_fields__))


def _read_seed_result(result_path: Path, trajectory_path: Path) -> ContinuousLCBSeedResult:
    if not result_path.exists():
        raise FileNotFoundError(f"Missing continuous policy-LCB seed result: {result_path}")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing continuous policy-LCB trajectories: {trajectory_path}")
    payload = read_json(result_path)
    trajectories: list[ContinuousLCBTrajectoryRow] = []
    with trajectory_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gradient_raw = row["gradient_estimate"]
            trajectories.append(
                ContinuousLCBTrajectoryRow(
                    run_seed=int(row["run_seed"]),
                    noise_seed=int(row["noise_seed"]),
                    delta=float(row["delta"]),
                    estimator=row["estimator"],
                    start_policy=float(row["start_policy"]),
                    step=int(row["step"]),
                    policy=float(row["policy"]),
                    loss=float(row["loss"]),
                    lcb=float(row["lcb"]),
                    gradient_estimate=None if gradient_raw == "" else float(gradient_raw),
                )
            )
    return ContinuousLCBSeedResult(
        run_seed=int(payload["run_seed"]),
        noise_seed=int(payload["noise_seed"]),
        stein_seed=int(payload["stein_seed"]),
        z=float(payload["z"]),
        start_results=tuple(ContinuousLCBStartResult(**row) for row in payload["start_results"]),
        best_results=tuple(ContinuousLCBBestResult(**row) for row in payload["best_results"]),
        trajectories=tuple(trajectories),
    )


def _run_continuous_start(
    spec: ContinuousPolicyLCBSpec,
    *,
    run_seed: int,
    noise_seed: int,
    stein_seed: int,
    z: float,
    delta: float,
    estimator: ContinuousEstimator,
    start_policy: float,
) -> tuple[ContinuousLCBStartResult, tuple[ContinuousLCBTrajectoryRow, ...]]:
    optimizer = spec.optimizer
    policy = float(start_policy)
    trace = [
        _trajectory_row(
            run_seed,
            noise_seed,
            delta,
            estimator,
            start_policy,
            0,
            policy,
            z,
            None,
            spec.true_value,
        )
    ]
    epsilon_sequences = rng_from_seed(stein_seed).normal(
        size=(optimizer.t_steps, optimizer.n_grad_samples, 1)
    )
    converged = False
    for step in range(1, optimizer.t_steps + 1):
        gradient = _gradient_estimate(
            estimator,
            policy=policy,
            z=z,
            delta=delta,
            sigma=optimizer.sigma,
            epsilon_samples=epsilon_sequences[step - 1],
            value_spec=spec.true_value,
        )
        next_policy = float(np.clip(policy - optimizer.step_size * gradient, *spec.policy_domain))
        trace.append(
            _trajectory_row(
                run_seed,
                noise_seed,
                delta,
                estimator,
                start_policy,
                step,
                next_policy,
                z,
                gradient,
                spec.true_value,
            )
        )
        if np.isclose(next_policy, policy, rtol=0.0, atol=1e-12):
            converged = True
            policy = next_policy
            break
        policy = next_policy

    result = _start_result(
        run_seed=run_seed,
        noise_seed=noise_seed,
        stein_seed=stein_seed,
        z=z,
        delta=delta,
        estimator=estimator,
        start_policy=start_policy,
        final_policy=policy,
        n_steps=len(trace) - 1,
        converged=converged,
        value_spec=spec.true_value,
    )
    return result, tuple(trace)


def _gradient_estimate(
    estimator: ContinuousEstimator,
    *,
    policy: float,
    z: float,
    delta: float,
    sigma: float,
    epsilon_samples: np.ndarray,
    value_spec: ContinuousPolicyValueSpec,
) -> float:
    if estimator == "first_order":
        return continuous_lcb_gradient(policy, z, delta, value_spec)
    value_fn = lambda theta: continuous_lcb_loss(float(theta[0]), z, delta, value_spec)
    theta = np.asarray([policy], dtype=float)
    if estimator == "finite_difference":
        return float(
            finite_difference_theta_grad(value_fn, theta, method="central", step=sigma)[0]
        )
    return float(
        stein_difference_theta_grad(
            value_fn,
            theta,
            step=sigma,
            epsilon_samples=epsilon_samples,
        )[0]
    )


def _trajectory_row(
    run_seed: int,
    noise_seed: int,
    delta: float,
    estimator: str,
    start_policy: float,
    step: int,
    policy: float,
    z: float,
    gradient: float | None,
    value_spec: ContinuousPolicyValueSpec,
) -> ContinuousLCBTrajectoryRow:
    loss = continuous_lcb_loss(policy, z, delta, value_spec)
    return ContinuousLCBTrajectoryRow(
        run_seed=run_seed,
        noise_seed=noise_seed,
        delta=float(delta),
        estimator=estimator,
        start_policy=float(start_policy),
        step=step,
        policy=float(policy),
        loss=loss,
        lcb=-loss,
        gradient_estimate=None if gradient is None else float(gradient),
    )


def _start_result(
    *,
    run_seed: int,
    noise_seed: int,
    stein_seed: int,
    z: float,
    delta: float,
    estimator: str,
    start_policy: float,
    final_policy: float,
    n_steps: int,
    converged: bool,
    value_spec: ContinuousPolicyValueSpec,
) -> ContinuousLCBStartResult:
    quantile = continuous_lcb_quantile(delta)
    analytic_policy = continuous_analytic_policy(z, delta, value_spec)
    analytic_loss = continuous_lcb_loss(analytic_policy, z, delta, value_spec)
    final_loss = continuous_lcb_loss(final_policy, z, delta, value_spec)
    optimization_error = max(0.0, final_loss - analytic_loss)
    simultaneous = bool(abs(z) <= quantile + ORACLE_TOLERANCE)
    selected_valid = bool(
        abs(final_policy * z) <= final_policy * quantile + ORACLE_TOLERANCE
    )
    worst_comparator = continuous_oracle_comparator_policy(delta, value_spec)
    true_optimal_policy = continuous_true_optimal_policy(value_spec)
    final_true_value = continuous_true_value(final_policy, value_spec)
    worst_slack = _oracle_slack(
        final_true_value,
        worst_comparator,
        quantile,
        optimization_error,
        value_spec,
    )
    optimal_comparator_slack = _oracle_slack(
        final_true_value,
        true_optimal_policy,
        quantile,
        optimization_error,
        value_spec,
    )
    true_optimal_value = continuous_true_value(true_optimal_policy, value_spec)
    return ContinuousLCBStartResult(
        run_seed=run_seed,
        noise_seed=noise_seed,
        stein_seed=stein_seed,
        z=float(z),
        delta=float(delta),
        quantile=quantile,
        estimator=estimator,
        start_policy=float(start_policy),
        final_policy=float(final_policy),
        final_true_value=final_true_value,
        final_surrogate_value=continuous_surrogate_value(final_policy, z, value_spec),
        final_lcb=-final_loss,
        final_loss=final_loss,
        analytic_policy=analytic_policy,
        analytic_lcb=-analytic_loss,
        analytic_loss=analytic_loss,
        optimization_error=optimization_error,
        regret=max(0.0, true_optimal_value - final_true_value),
        n_steps=int(n_steps),
        converged=bool(converged),
        simultaneous_coverage=simultaneous,
        selected_interval_valid=selected_valid,
        worst_oracle_slack=worst_slack,
        optimal_comparator_slack=optimal_comparator_slack,
        oracle_violation=bool(worst_slack < -ORACLE_TOLERANCE),
    )


def continuous_oracle_comparator_policy(
    delta: float,
    value_spec: ContinuousPolicyValueSpec | None = None,
) -> float:
    """Return the comparator making the continuum oracle inequality tightest."""
    value = ContinuousPolicyValueSpec() if value_spec is None else value_spec
    linear_coefficient = value.a - 2.0 * continuous_lcb_quantile(delta)
    if value.b == 0.0:
        return 0.0 if linear_coefficient <= 0.0 else 1.0
    return float(np.clip(linear_coefficient / (2.0 * value.b), 0.0, 1.0))


def _oracle_slack(
    final_true_value: float,
    comparator_policy: float,
    quantile: float,
    optimization_error: float,
    value_spec: ContinuousPolicyValueSpec,
) -> float:
    comparator_rhs = (
        continuous_true_value(comparator_policy, value_spec)
        - 2.0 * comparator_policy * quantile
        - optimization_error
    )
    return float(final_true_value - comparator_rhs)


def _best_from_start(row: ContinuousLCBStartResult) -> ContinuousLCBBestResult:
    payload = {
        field: getattr(row, field)
        for field in ContinuousLCBBestResult.__dataclass_fields__
        if field != "selected_start_policy"
    }
    return ContinuousLCBBestResult(selected_start_policy=row.start_policy, **payload)


__all__ = [
    "ContinuousLCBBestResult",
    "ContinuousLCBSeedResult",
    "ContinuousLCBStartResult",
    "ContinuousLCBTrajectoryRow",
    "ContinuousPolicyLCBLaunchSpec",
    "ContinuousPolicyLCBManifest",
    "ContinuousPolicyLCBOptimizerSpec",
    "ContinuousPolicyLCBSpec",
    "ContinuousPolicyValueSpec",
    "collect_continuous_policy_lcb_outputs",
    "continuous_analytic_policy",
    "continuous_lcb_gradient",
    "continuous_lcb_loss",
    "continuous_lcb_quantile",
    "continuous_lcb_slope",
    "continuous_lcb_value",
    "continuous_noise_seed_for_run",
    "continuous_oracle_comparator_policy",
    "continuous_surrogate_value",
    "continuous_stein_seed",
    "continuous_true_optimal_policy",
    "continuous_true_value",
    "continuous_policy_lcb_seed_complete",
    "evaluate_continuous_policy_lcb_draw",
    "evaluate_continuous_policy_lcb_seed",
    "load_continuous_policy_lcb_manifest",
    "parse_continuous_policy_lcb_manifest",
    "run_continuous_policy_lcb_manifest_seed",
    "run_continuous_policy_lcb_manifest_serial",
    "shared_gaussian_coverage",
    "write_continuous_policy_lcb_experiment_readme",
]
