"""Continuous shared-Gaussian policy-LCB optimization adapter."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from experiments.paths import results_root
from experiments.policy_lcb.common import (
    ORACLE_TOLERANCE,
    PolicyLCBLaunchSpec,
    gaussian_lcb_quantile,
    path_part,
    shared_gaussian_coverage,
)
from experiments.seeds import derive_seed, rng_from_seed
from optimization.helpers import finite_difference_theta_grad, stein_difference_theta_grad


ContinuousEstimator = Literal["first_order", "finite_difference", "stein_difference"]
_ALLOWED_ESTIMATORS = {"first_order", "finite_difference", "stein_difference"}


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


def continuous_lcb_quantile(delta: float) -> float:
    """Return the exact two-sided quantile for one shared Gaussian draw."""
    return gaussian_lcb_quantile(delta)


def continuous_lcb_loss(policy: float, z: float, delta: float) -> float:
    """Return negative LCB for the shared-Gaussian identity-value model."""
    return float(policy) * continuous_lcb_slope(z, delta)


def continuous_lcb_slope(z: float, delta: float) -> float:
    """Return the exact derivative of the negative LCB with respect to policy."""
    return continuous_lcb_quantile(delta) - 1.0 - float(z)


def continuous_lcb_value(policy: float, z: float, delta: float) -> float:
    """Return the lower confidence bound, the negative of the optimized loss."""
    return -continuous_lcb_loss(policy, z, delta)


def continuous_analytic_policy(z: float, delta: float) -> float:
    """Return the smallest exact minimizer over $$[0,1]$$."""
    return 0.0 if continuous_lcb_slope(z, delta) >= 0.0 else 1.0


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
            )
        )
        if next_policy == policy:
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
) -> float:
    if estimator == "first_order":
        return continuous_lcb_slope(z, delta)
    value_fn = lambda theta: continuous_lcb_loss(float(theta[0]), z, delta)
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
) -> ContinuousLCBTrajectoryRow:
    loss = continuous_lcb_loss(policy, z, delta)
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
) -> ContinuousLCBStartResult:
    quantile = continuous_lcb_quantile(delta)
    analytic_policy = continuous_analytic_policy(z, delta)
    analytic_loss = continuous_lcb_loss(analytic_policy, z, delta)
    final_loss = continuous_lcb_loss(final_policy, z, delta)
    optimization_error = max(0.0, final_loss - analytic_loss)
    simultaneous = bool(abs(z) <= quantile + ORACLE_TOLERANCE)
    selected_valid = bool(
        abs(final_policy * z) <= final_policy * quantile + ORACLE_TOLERANCE
    )
    comparator_values = np.asarray(spec_comparators(), dtype=float)
    comparator_rhs = comparator_values - 2.0 * comparator_values * quantile - optimization_error
    comparator_slacks = final_policy - comparator_rhs
    worst_slack = float(np.min(comparator_slacks))
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
        final_true_value=float(final_policy),
        final_surrogate_value=float(final_policy * (1.0 + z)),
        final_lcb=-final_loss,
        final_loss=final_loss,
        analytic_policy=analytic_policy,
        analytic_lcb=-analytic_loss,
        analytic_loss=analytic_loss,
        optimization_error=optimization_error,
        regret=1.0 - float(final_policy),
        n_steps=int(n_steps),
        converged=bool(converged),
        simultaneous_coverage=simultaneous,
        selected_interval_valid=selected_valid,
        worst_oracle_slack=worst_slack,
        optimal_comparator_slack=float(comparator_slacks[-1]),
        oracle_violation=bool(worst_slack < -ORACLE_TOLERANCE),
    )


def spec_comparators() -> tuple[float, float]:
    """Return endpoints sufficient for the linear continuous oracle check."""
    return (0.0, 1.0)


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
    "continuous_analytic_policy",
    "continuous_lcb_loss",
    "continuous_lcb_quantile",
    "continuous_lcb_slope",
    "continuous_lcb_value",
    "continuous_noise_seed_for_run",
    "continuous_stein_seed",
    "evaluate_continuous_policy_lcb_draw",
    "evaluate_continuous_policy_lcb_seed",
    "shared_gaussian_coverage",
]
