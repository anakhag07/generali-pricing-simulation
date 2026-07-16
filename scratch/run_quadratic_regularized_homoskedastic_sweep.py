"""Sweep theta-space proximal/support regularizers on the noisy quadratic."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import run_quadratic_homoskedastic_sweep as base_sweep  # noqa: E402
from experiments.config import CorrectnessSpec  # noqa: E402
from experiments.launch import (  # noqa: E402
    LaunchContext,
    LaunchPlan,
    add_launch_args,
    read_task_records,
    run_launch_plan,
    task_payloads,
)
from experiments.paths import results_root  # noqa: E402
from experiments.sweep_utils import run_sweep  # noqa: E402
from objective.base import Objective  # noqa: E402
from objective.noise import HomoskedasticGaussianNoise, NoisyObjective  # noqa: E402
from objective.objectives import StronglyConvexQuadratic  # noqa: E402


BASE_PRESET = base_sweep.BASE_PRESET
PROJECT_NAME = "quadratic-regularized-homoskedastic-lbfgsb-sweep"
PILOT_PROJECT_NAME = "quadratic-regularized-homoskedastic-lbfgsb-pilot"
OPTAX_PROJECT_NAME = "quadratic-regularized-homoskedastic-optax-adam-sweep"
OPTAX_PILOT_PROJECT_NAME = "quadratic-regularized-homoskedastic-optax-adam-pilot"
ESTIMATOR = base_sweep.ESTIMATOR
L_BFGS_B = base_sweep.L_BFGS_B
OPTAX_ADAM = base_sweep.OPTAX_ADAM
OPTIMIZERS = base_sweep.OPTIMIZERS

DEFAULT_DIMENSION = base_sweep.DEFAULT_DIMENSION
DEFAULT_STEP_SIZE = base_sweep.DEFAULT_STEP_SIZE
DEFAULT_ARRAY_MAX_PARALLEL = base_sweep.DEFAULT_ARRAY_MAX_PARALLEL
DEFAULT_T_STEPS = base_sweep.DEFAULT_T_STEPS
DEFAULT_NOISE_STDS = base_sweep.DEFAULT_NOISE_STDS
DEFAULT_FD_RADII = base_sweep.DEFAULT_FD_RADII
DEFAULT_RUN_SEEDS = base_sweep.DEFAULT_RUN_SEEDS

PILOT_NOISE_STDS = base_sweep.PILOT_NOISE_STDS
PILOT_FD_RADII = base_sweep.PILOT_FD_RADII
PILOT_RUN_SEEDS = base_sweep.PILOT_RUN_SEEDS

REGULARIZERS = ("none", "proximal", "support")
DEFAULT_REGULARIZERS = ("proximal", "support")
DEFAULT_REGULARIZER_WEIGHTS = (0.01, 0.1, 1.0)
DEFAULT_SUPPORT_GROWTH = 1.0
PROXIMAL_REFERENCE_SOURCES = ("u-center", "zero", "theta0")

FINAL_FIELDNAMES = (
    "regularizer",
    "regularizer_weight",
    "u_center",
    "support_growth",
    "noise_std",
    "fd_radius",
    "noise_to_radius",
    "run_seed",
    "dimension",
    "final_theta_norm",
    "clean_final_objective",
    "noisy_final_objective",
    "exploitation_gap",
    "clean_improvement",
    "runtime_sec",
    "trace_steps",
    "optimizer_success",
    "optimizer_status",
    "optimizer_message",
    "summary_path",
    "run_dir",
)

SUMMARY_METRICS = base_sweep.SUMMARY_METRICS
SUMMARY_STATS = base_sweep.SUMMARY_STATS
SUMMARY_FIELDNAMES = (
    "regularizer",
    "regularizer_weight",
    "u_center",
    "support_growth",
    "noise_std",
    "fd_radius",
    "noise_to_radius",
    "n_seeds",
    "optimizer_success_rate",
    *(f"{metric}_{stat}" for metric in SUMMARY_METRICS for stat in SUMMARY_STATS),
)


@dataclass(frozen=True)
class ThetaRegularizedObjective(Objective):
    """Theta-space scratch wrapper using ``u := theta`` as the action vector."""

    objective: Objective
    proximal_weight: float | None = None
    theta_reference: np.ndarray | None = None
    support_weight: float | None = None
    support_center: float = 0.0
    support_growth: float = 1.0

    def __post_init__(self) -> None:
        proximal_weight = _optional_nonnegative(self.proximal_weight, "proximal_weight")
        support_weight = _optional_nonnegative(self.support_weight, "support_weight")
        support_center = float(self.support_center)
        support_growth = float(self.support_growth)
        if not np.isfinite(support_center):
            raise ValueError("support_center must be finite.")
        if not np.isfinite(support_growth) or support_growth < 0.0:
            raise ValueError("support_growth must be finite and nonnegative.")
        theta_reference = None
        if self.theta_reference is not None:
            theta_reference = np.asarray(self.theta_reference, dtype=float).reshape(-1)
            if theta_reference.ndim != 1 or not np.all(np.isfinite(theta_reference)):
                raise ValueError("theta_reference must be a finite 1D array.")
        object.__setattr__(self, "proximal_weight", proximal_weight)
        object.__setattr__(self, "theta_reference", theta_reference)
        object.__setattr__(self, "support_weight", support_weight)
        object.__setattr__(self, "support_center", support_center)
        object.__setattr__(self, "support_growth", support_growth)

    def with_noise_seed(self, seed: int) -> "ThetaRegularizedObjective":
        with_noise_seed = getattr(self.objective, "with_noise_seed", None)
        if callable(with_noise_seed):
            return replace(self, objective=with_noise_seed(int(seed)))
        return self

    def theta_dim(self, state_dim: int | None = None) -> int:
        theta_dim_fn = getattr(self.objective, "theta_dim", None)
        if callable(theta_dim_fn):
            return int(theta_dim_fn(state_dim))
        base_objective = getattr(self.objective, "base_objective", None)
        theta_dim_fn = getattr(base_objective, "theta_dim", None)
        if callable(theta_dim_fn):
            return int(theta_dim_fn(state_dim))
        raise ValueError("wrapped objective does not expose theta_dim.")

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        theta_arr = _validate_theta(theta)
        value = float(self.objective.value(theta_arr, x_batch))
        return value + self._regularizer_value(theta_arr)

    def base_value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        base_value_fn = getattr(self.objective, "base_value", None)
        if callable(base_value_fn):
            return float(base_value_fn(theta, x_batch))
        return float(self.objective.value(theta, x_batch))

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        theta_arr = _validate_theta(theta)
        clean_objective = getattr(self.objective, "base_objective", self.objective)
        grad = np.asarray(clean_objective.grad(theta_arr, x_batch), dtype=float)
        if grad.shape != theta_arr.shape:
            raise ValueError("wrapped objective gradient shape does not match theta.")
        return grad + self._regularizer_grad(theta_arr)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": type(self).__name__,
            "proximal_weight": self.proximal_weight,
            "theta_reference": None
            if self.theta_reference is None
            else self.theta_reference.tolist(),
            "support_weight": self.support_weight,
            "support_center": float(self.support_center),
            "support_growth": float(self.support_growth),
        }

    def _regularizer_value(self, theta: np.ndarray) -> float:
        value = 0.0
        if self.proximal_weight is not None:
            ref = self._reference_for(theta)
            value += float(self.proximal_weight) * float(np.mean((theta - ref) ** 2))
        if self.support_weight is not None:
            sigma = self.support_growth * np.abs(theta - self.support_center)
            value += float(self.support_weight) * float(np.mean(sigma))
        return float(value)

    def _regularizer_grad(self, theta: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(theta, dtype=float)
        dimension = float(theta.size)
        if self.proximal_weight is not None:
            ref = self._reference_for(theta)
            grad += (2.0 * float(self.proximal_weight) / dimension) * (theta - ref)
        if self.support_weight is not None:
            delta = theta - self.support_center
            support_grad = self.support_growth * np.sign(delta)
            support_grad = np.where(delta == 0.0, 0.0, support_grad)
            grad += (float(self.support_weight) / dimension) * support_grad
        return grad

    def _reference_for(self, theta: np.ndarray) -> np.ndarray:
        if self.theta_reference is None:
            return np.zeros_like(theta, dtype=float)
        if self.theta_reference.shape != theta.shape:
            raise ValueError(
                "theta_reference must have the same length as theta "
                f"({self.theta_reference.size} != {theta.size})."
            )
        return self.theta_reference


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Run the reduced calibration grid.")
    parser.add_argument("--plots-only", action="store_true", help="Rebuild CSVs and plots from saved summaries.")
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--noise-stds", type=float, nargs="+", default=None)
    parser.add_argument("--fd-radii", type=float, nargs="+", default=None)
    parser.add_argument("--run-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--regularizers", choices=REGULARIZERS, nargs="+", default=None)
    parser.add_argument("--regularizer-weights", type=float, nargs="+", default=None)
    parser.add_argument(
        "--u-center",
        type=float,
        default=None,
        help=(
            "Scalar action center. Defaults to the mean of the default u_center vector, "
            "which is theta0 for this quadratic scratch setup."
        ),
    )
    parser.add_argument(
        "--support-growth",
        type=float,
        default=DEFAULT_SUPPORT_GROWTH,
        help="Slope of the support proxy sigma(u)=support_growth*abs(u-u_center).",
    )
    parser.add_argument(
        "--proximal-reference",
        choices=PROXIMAL_REFERENCE_SOURCES,
        default="u-center",
        help="Reference vector for the proximal penalty.",
    )
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default=L_BFGS_B)
    parser.add_argument("--t-steps", type=int, default=None)
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--project-name", default=None)
    add_launch_args(parser)
    parser.set_defaults(array_max_parallel=DEFAULT_ARRAY_MAX_PARALLEL)
    return parser.parse_args(argv)


def _resolved_grid(args: argparse.Namespace) -> tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]:
    return base_sweep._resolved_grid(args)


def _resolved_regularizer_specs(args: argparse.Namespace) -> tuple[tuple[str, float], ...]:
    regularizers = tuple(args.regularizers or DEFAULT_REGULARIZERS)
    weights = tuple(float(value) for value in (args.regularizer_weights or DEFAULT_REGULARIZER_WEIGHTS))
    if not regularizers:
        raise ValueError("regularizers must contain at least one value.")
    if not weights:
        raise ValueError("regularizer weights must contain at least one value.")
    if any(weight < 0.0 or not np.isfinite(weight) for weight in weights):
        raise ValueError("regularizer weights must be finite and nonnegative.")
    specs: list[tuple[str, float]] = []
    seen: set[tuple[str, float]] = set()
    for regularizer in regularizers:
        reg = str(regularizer)
        if reg == "none":
            spec = ("none", 0.0)
            if spec not in seen:
                seen.add(spec)
                specs.append(spec)
            continue
        for weight in weights:
            spec = (reg, float(weight))
            if spec not in seen:
                seen.add(spec)
                specs.append(spec)
    return tuple(specs)


def _project_name(args: argparse.Namespace) -> str:
    if args.project_name:
        return str(args.project_name)
    if args.optimizer == OPTAX_ADAM:
        return OPTAX_PILOT_PROJECT_NAME if args.pilot else OPTAX_PROJECT_NAME
    return PILOT_PROJECT_NAME if args.pilot else PROJECT_NAME


def _resolved_t_steps(args: argparse.Namespace) -> int:
    return base_sweep._resolved_t_steps(args)


def _task_specs(args: argparse.Namespace) -> list[tuple[str, float, float, float]]:
    noise_stds, fd_radii, _ = _resolved_grid(args)
    regularizer_specs = _resolved_regularizer_specs(args)
    return [
        (regularizer, weight, noise_std, fd_radius)
        for regularizer, weight in regularizer_specs
        for noise_std in noise_stds
        for fd_radius in fd_radii
    ]


def _variant_name(
    regularizer: str,
    regularizer_weight: float,
    *,
    u_center: float,
    support_growth: float,
    noise_std: float,
    fd_radius: float,
) -> str:
    return (
        f"regularizer-{regularizer}"
        f"__weight-{_value_label(regularizer_weight)}"
        f"__u-center-{_value_label(u_center)}"
        f"__support-growth-{_value_label(support_growth)}"
        f"__noise-std-{_value_label(noise_std)}"
        f"__fd-radius-{_value_label(fd_radius)}"
    )


def _parse_variant(name: str) -> tuple[str, float, float, float, float, float] | None:
    parts = name.split("__")
    if len(parts) != 6:
        return None
    prefixes = (
        "regularizer-",
        "weight-",
        "u-center-",
        "support-growth-",
        "noise-std-",
        "fd-radius-",
    )
    if any(not part.startswith(prefix) for part, prefix in zip(parts, prefixes, strict=True)):
        return None
    regularizer = parts[0].removeprefix(prefixes[0])
    if regularizer not in REGULARIZERS:
        return None
    try:
        return (
            regularizer,
            float(parts[1].removeprefix(prefixes[1])),
            float(parts[2].removeprefix(prefixes[2])),
            float(parts[3].removeprefix(prefixes[3])),
            float(parts[4].removeprefix(prefixes[4])),
            float(parts[5].removeprefix(prefixes[5])),
        )
    except ValueError:
        return None


def _build_override_list(
    *,
    dimension: int,
    regularizer_specs: Sequence[tuple[str, float]],
    noise_stds: Sequence[float],
    fd_radii: Sequence[float],
    t_steps: int,
    optimizer: str,
    step_size: float,
    u_center_override: float | None,
    support_growth: float,
    proximal_reference: str,
) -> list[dict[str, object]]:
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if t_steps <= 0:
        raise ValueError("t_steps must be positive.")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}.")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive.")
    support_growth = float(support_growth)
    if not np.isfinite(support_growth) or support_growth < 0.0:
        raise ValueError("support_growth must be finite and nonnegative.")
    if proximal_reference not in PROXIMAL_REFERENCE_SOURCES:
        raise ValueError(f"proximal_reference must be one of {PROXIMAL_REFERENCE_SOURCES}.")

    theta0 = _default_theta0(dimension)
    u_center_vector = _u_center_vector(theta0, u_center_override)
    u_center = float(np.mean(u_center_vector))
    reference_vector = _proximal_reference_vector(
        theta0=theta0,
        u_center_vector=u_center_vector,
        source=proximal_reference,
    )
    base_objective = StronglyConvexQuadratic.isotropic(dimension)
    overrides: list[dict[str, object]] = []
    for regularizer, regularizer_weight in regularizer_specs:
        for noise_std in noise_stds:
            for fd_radius in fd_radii:
                noisy_objective = NoisyObjective(
                    base_objective=base_objective,
                    noise=HomoskedasticGaussianNoise(std=float(noise_std)),
                )
                objective = _regularized_objective(
                    noisy_objective,
                    regularizer=regularizer,
                    regularizer_weight=float(regularizer_weight),
                    theta_reference=reference_vector,
                    u_center=u_center,
                    support_growth=support_growth,
                )
                overrides.append(
                    {
                        "_run_name": _variant_name(
                            regularizer,
                            float(regularizer_weight),
                            u_center=u_center,
                            support_growth=support_growth,
                            noise_std=float(noise_std),
                            fd_radius=float(fd_radius),
                        ),
                        "dimension": int(dimension),
                        "objective": objective,
                        "theta0": theta0.copy(),
                        "n_samples": 1,
                        "step_rule": optimizer,
                        "perturbation_space": "theta",
                        "t_steps": int(t_steps),
                        "step_size": float(step_size),
                        "sigma": float(fd_radius),
                        "grad_norm_tol": 1e-8,
                        **({"ftol": 1e-12} if optimizer == L_BFGS_B else {}),
                        "enabled_estimators": (ESTIMATOR,),
                        "correctness": CorrectnessSpec(
                            gradient_source="denoised_exact"
                            if regularizer == "none"
                            else "exact"
                        ),
                        "plot": False,
                        "verbose": False,
                        "wandb_enabled": False,
                    }
                )
    return overrides


def _regularized_objective(
    objective: Objective,
    *,
    regularizer: str,
    regularizer_weight: float,
    theta_reference: np.ndarray,
    u_center: float,
    support_growth: float,
) -> Objective:
    if regularizer == "none":
        return objective
    if regularizer == "proximal":
        return ThetaRegularizedObjective(
            objective=objective,
            proximal_weight=float(regularizer_weight),
            theta_reference=theta_reference.copy(),
            support_center=float(u_center),
            support_growth=float(support_growth),
        )
    if regularizer == "support":
        return ThetaRegularizedObjective(
            objective=objective,
            support_weight=float(regularizer_weight),
            support_center=float(u_center),
            support_growth=float(support_growth),
        )
    raise ValueError(f"Unknown regularizer: {regularizer}")


def _run_grid(
    *,
    project_name: str,
    dimension: int,
    regularizer_specs: Sequence[tuple[str, float]],
    noise_stds: Sequence[float],
    fd_radii: Sequence[float],
    run_seeds: Sequence[int],
    t_steps: int,
    optimizer: str,
    step_size: float,
    u_center_override: float | None,
    support_growth: float,
    proximal_reference: str,
) -> Path:
    sweep = run_sweep(
        base_preset=BASE_PRESET,
        run_seeds=tuple(int(seed) for seed in run_seeds),
        override_list=_build_override_list(
            dimension=dimension,
            regularizer_specs=regularizer_specs,
            noise_stds=noise_stds,
            fd_radii=fd_radii,
            t_steps=t_steps,
            optimizer=optimizer,
            step_size=step_size,
            u_center_override=u_center_override,
            support_growth=support_growth,
            proximal_reference=proximal_reference,
        ),
        vary=("noise",),
        anchor_seed=int(run_seeds[0]),
        fixed={},
        per_seed_plots=False,
        project_name=project_name,
        display_keys=(),
    )
    return sweep.project_dir


def _collect_rows(project_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not project_dir.is_dir():
        return rows
    for variant_dir in sorted(project_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        parsed = _parse_variant(variant_dir.name)
        if parsed is None:
            continue
        regularizer, weight, u_center, support_growth, noise_std, fd_radius = parsed
        for summary_path in sorted(variant_dir.glob("summary-seed-*.json")):
            try:
                summary = _load_json(summary_path)
                row = _summary_row(
                    summary,
                    summary_path,
                    regularizer=regularizer,
                    regularizer_weight=weight,
                    u_center=u_center,
                    support_growth=support_growth,
                    noise_std=noise_std,
                    fd_radius=fd_radius,
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                continue
            rows.append(row)
    return _sort_rows(rows)


def _run_grid_task(
    index: int,
    context: LaunchContext,
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    del context
    regularizer, regularizer_weight, noise_std, fd_radius = _task_specs(args)[index]
    _, _, run_seeds = _resolved_grid(args)
    project_name = _project_name(args)
    theta0 = _default_theta0(int(args.dimension))
    u_center = float(np.mean(_u_center_vector(theta0, args.u_center)))
    support_growth = float(args.support_growth)
    project_dir = _run_grid(
        project_name=project_name,
        dimension=int(args.dimension),
        regularizer_specs=((regularizer, regularizer_weight),),
        noise_stds=(noise_std,),
        fd_radii=(fd_radius,),
        run_seeds=run_seeds,
        t_steps=_resolved_t_steps(args),
        optimizer=str(args.optimizer),
        step_size=float(args.step_size),
        u_center_override=args.u_center,
        support_growth=support_growth,
        proximal_reference=str(args.proximal_reference),
    )
    variant_name = _variant_name(
        regularizer,
        regularizer_weight,
        u_center=u_center,
        support_growth=support_growth,
        noise_std=noise_std,
        fd_radius=fd_radius,
    )
    summary_paths = [
        project_dir / variant_name / f"summary-seed-{seed}.json" for seed in run_seeds
    ]
    missing = [path for path in summary_paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"Task {index} completed without expected summaries: "
            + ", ".join(str(path) for path in missing)
        )
    print(
        f"Completed regularized quadratic task {index}: {variant_name} "
        f"({len(summary_paths)} seeds)."
    )
    return {
        "project_name": project_name,
        "variant_name": variant_name,
        "regularizer": regularizer,
        "regularizer_weight": regularizer_weight,
        "u_center": u_center,
        "support_growth": support_growth,
        "noise_std": noise_std,
        "fd_radius": fd_radius,
        "summary_paths": [str(path) for path in summary_paths],
    }


def _run_grid_serial(context: LaunchContext, *, args: argparse.Namespace) -> None:
    del context
    noise_stds, fd_radii, run_seeds = _resolved_grid(args)
    project_dir = _run_grid(
        project_name=_project_name(args),
        dimension=int(args.dimension),
        regularizer_specs=_resolved_regularizer_specs(args),
        noise_stds=noise_stds,
        fd_radii=fd_radii,
        run_seeds=run_seeds,
        t_steps=_resolved_t_steps(args),
        optimizer=str(args.optimizer),
        step_size=float(args.step_size),
        u_center_override=args.u_center,
        support_growth=float(args.support_growth),
        proximal_reference=str(args.proximal_reference),
    )
    rows = _collect_rows(project_dir)
    if not rows:
        raise RuntimeError(f"No completed regularized quadratic summaries found under {project_dir}.")
    _write_outputs(project_dir, rows, optimizer=str(args.optimizer))
    print(f"Wrote {len(rows)} regularized quadratic sweep rows under {project_dir}.")


def _collect_grid_tasks(context: LaunchContext, *, args: argparse.Namespace) -> None:
    expected_indices = set(range(len(_task_specs(args))))
    records = read_task_records(context)
    actual_indices = {int(record["task_index"]) for record in records}
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        unexpected = sorted(actual_indices - expected_indices)
        raise RuntimeError(
            "Cannot collect incomplete regularized quadratic array: "
            f"missing task indices={missing}, unexpected task indices={unexpected}."
        )
    payloads = task_payloads(context)
    rows = _rows_from_task_payloads(payloads)
    if not rows:
        raise RuntimeError("No completed regularized quadratic rows were produced by array tasks.")
    project_dir = _project_dir(_project_name(args))
    _write_outputs(project_dir, rows, optimizer=str(args.optimizer))
    print(f"Collected {len(payloads)} tasks and wrote {len(rows)} rows under {project_dir}.")


def _rows_from_task_payloads(
    payloads: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for payload in payloads:
        regularizer = str(payload["regularizer"])
        regularizer_weight = float(payload["regularizer_weight"])
        u_center = float(payload["u_center"])
        support_growth = float(payload["support_growth"])
        noise_std = float(payload["noise_std"])
        fd_radius = float(payload["fd_radius"])
        summary_paths = payload.get("summary_paths")
        if not isinstance(summary_paths, list):
            raise TypeError("Regularized quadratic task payload must contain a summary_paths list.")
        for summary_value in summary_paths:
            summary_path = Path(str(summary_value))
            summary = _load_json(summary_path)
            rows.append(
                _summary_row(
                    summary,
                    summary_path,
                    regularizer=regularizer,
                    regularizer_weight=regularizer_weight,
                    u_center=u_center,
                    support_growth=support_growth,
                    noise_std=noise_std,
                    fd_radius=fd_radius,
                )
            )
    return _sort_rows(rows)


def _build_launch_plan(args: argparse.Namespace) -> LaunchPlan:
    use_gpu = str(args.optimizer) == OPTAX_ADAM
    return LaunchPlan(
        name=_project_name(args),
        task_count=len(_task_specs(args)),
        requires_jax=use_gpu,
        run_task=lambda index, context: _run_grid_task(index, context, args=args),
        run_all=lambda context: _run_grid_serial(context, args=args),
        collect=lambda context: _collect_grid_tasks(context, args=args),
        default_launch="auto" if use_gpu else "local",
        default_array=use_gpu,
    )


def _summary_row(
    summary: Mapping[str, Any],
    summary_path: Path,
    *,
    regularizer: str,
    regularizer_weight: float,
    u_center: float,
    support_growth: float,
    noise_std: float,
    fd_radius: float,
) -> dict[str, object]:
    estimator = summary["estimators"][ESTIMATOR]
    trace = summary["trace_summary"][ESTIMATOR]
    theta = np.asarray(estimator["theta"], dtype=float)
    clean_final = float(estimator["final_value"])
    noisy_final = float(trace["final_objective"])
    config = summary["config"]
    resolved_seeds = config["resolved_seed_setup"]
    run_seed = int(resolved_seeds["run_seed"])
    dimension = _objective_dimension(config.get("objective", {}), fallback=theta.size)
    return {
        "regularizer": str(regularizer),
        "regularizer_weight": float(regularizer_weight),
        "u_center": float(u_center),
        "support_growth": float(support_growth),
        "noise_std": float(noise_std),
        "fd_radius": float(fd_radius),
        "noise_to_radius": float(noise_std) / float(fd_radius),
        "run_seed": run_seed,
        "dimension": dimension,
        "final_theta_norm": float(np.linalg.norm(theta)),
        "clean_final_objective": clean_final,
        "noisy_final_objective": noisy_final,
        "exploitation_gap": noisy_final - clean_final,
        "clean_improvement": float(summary["initial_value"]) - clean_final,
        "runtime_sec": float(estimator["runtime_sec"]),
        "trace_steps": int(trace["steps"]),
        "optimizer_success": bool(estimator["optimizer_success"]),
        "optimizer_status": int(estimator["optimizer_status"]),
        "optimizer_message": str(estimator["optimizer_message"]),
        "summary_path": str(summary_path),
        "run_dir": str(summary["run"]["run_dir"]),
    }


def _aggregate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (
                str(row["regularizer"]),
                float(row["regularizer_weight"]),
                float(row["u_center"]),
                float(row["support_growth"]),
                float(row["noise_std"]),
                float(row["fd_radius"]),
            )
            for row in rows
        }
    )
    output: list[dict[str, object]] = []
    for regularizer, weight, u_center, support_growth, noise_std, fd_radius in keys:
        group = [
            row
            for row in rows
            if str(row["regularizer"]) == regularizer
            and float(row["regularizer_weight"]) == weight
            and float(row["u_center"]) == u_center
            and float(row["support_growth"]) == support_growth
            and float(row["noise_std"]) == noise_std
            and float(row["fd_radius"]) == fd_radius
        ]
        summary: dict[str, object] = {
            "regularizer": regularizer,
            "regularizer_weight": weight,
            "u_center": u_center,
            "support_growth": support_growth,
            "noise_std": noise_std,
            "fd_radius": fd_radius,
            "noise_to_radius": noise_std / fd_radius,
            "n_seeds": len(group),
            "optimizer_success_rate": float(np.mean([bool(row["optimizer_success"]) for row in group])),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in group], dtype=float)
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=0))
            summary[f"{metric}_min"] = float(np.min(values))
            summary[f"{metric}_max"] = float(np.max(values))
            summary[f"{metric}_median"] = float(np.median(values))
        output.append(summary)
    return output


def _write_outputs(
    project_dir: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    optimizer: str = L_BFGS_B,
    plot: bool = True,
) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    summaries = _aggregate_rows(rows)
    _write_csv(project_dir / "quadratic_regularized_homoskedastic_finals.csv", FINAL_FIELDNAMES, rows)
    _write_csv(
        project_dir / "quadratic_regularized_homoskedastic_summary.csv",
        SUMMARY_FIELDNAMES,
        summaries,
    )
    if plot and summaries:
        for group_rows, group_summaries in _plot_groups(rows, summaries):
            first = group_summaries[0]
            group_name = _plot_group_name(first)
            base_sweep._write_plots(
                project_dir / "plots" / group_name,
                group_rows,
                group_summaries,
                optimizer=optimizer,
            )


def _plot_groups(
    rows: Sequence[Mapping[str, object]],
    summaries: Sequence[Mapping[str, object]],
) -> list[tuple[list[Mapping[str, object]], list[Mapping[str, object]]]]:
    keys = sorted(
        {
            (
                str(row["regularizer"]),
                float(row["regularizer_weight"]),
                float(row["u_center"]),
                float(row["support_growth"]),
            )
            for row in summaries
        }
    )
    groups: list[tuple[list[Mapping[str, object]], list[Mapping[str, object]]]] = []
    for key in keys:
        regularizer, weight, u_center, support_growth = key
        group_rows = [
            row
            for row in rows
            if (
                str(row["regularizer"]),
                float(row["regularizer_weight"]),
                float(row["u_center"]),
                float(row["support_growth"]),
            )
            == key
        ]
        group_summaries = [
            row
            for row in summaries
            if (
                str(row["regularizer"]),
                float(row["regularizer_weight"]),
                float(row["u_center"]),
                float(row["support_growth"]),
            )
            == key
        ]
        if group_rows and group_summaries:
            groups.append((group_rows, group_summaries))
    return groups


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _sort_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row["regularizer"]),
            float(row["regularizer_weight"]),
            float(row["u_center"]),
            float(row["support_growth"]),
            float(row["noise_std"]),
            float(row["fd_radius"]),
            int(row["run_seed"]),
        ),
    )


def _objective_dimension(payload: Mapping[str, object], *, fallback: int) -> int:
    dimension = payload.get("dimension")
    if dimension is not None:
        return int(dimension)
    nested = payload.get("base_objective")
    if isinstance(nested, Mapping):
        return _objective_dimension(nested, fallback=fallback)
    return int(fallback)


def _default_theta0(dimension: int) -> np.ndarray:
    return np.ones(int(dimension), dtype=float) / np.sqrt(float(dimension))


def _u_center_vector(theta0: np.ndarray, u_center_override: float | None) -> np.ndarray:
    if u_center_override is None:
        return np.asarray(theta0, dtype=float).copy()
    u_center = float(u_center_override)
    if not np.isfinite(u_center):
        raise ValueError("u_center must be finite.")
    return np.full_like(np.asarray(theta0, dtype=float), u_center, dtype=float)


def _proximal_reference_vector(
    *,
    theta0: np.ndarray,
    u_center_vector: np.ndarray,
    source: str,
) -> np.ndarray:
    if source == "u-center":
        return np.asarray(u_center_vector, dtype=float).copy()
    if source == "zero":
        return np.zeros_like(theta0, dtype=float)
    if source == "theta0":
        return np.asarray(theta0, dtype=float).copy()
    raise ValueError(f"Unknown proximal reference source: {source}")


def _plot_group_name(row: Mapping[str, object]) -> str:
    return (
        f"regularizer-{row['regularizer']}"
        f"__weight-{_value_label(float(row['regularizer_weight']))}"
        f"__u-center-{_value_label(float(row['u_center']))}"
        f"__support-growth-{_value_label(float(row['support_growth']))}"
    )


def _optional_nonnegative(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    value_float = float(value)
    if not np.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value_float


def _validate_theta(theta: np.ndarray) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float)
    if theta_arr.ndim != 1:
        raise ValueError("theta must be a 1D array.")
    if theta_arr.size == 0 or not np.all(np.isfinite(theta_arr)):
        raise ValueError("theta must contain at least one finite value.")
    return theta_arr


def _load_json(path: Path) -> dict[str, Any]:
    return base_sweep._load_json(path)


def _value_label(value: float) -> str:
    return base_sweep._value_label(value)


def _project_dir(project_name: str) -> Path:
    return results_root() / str(project_name).replace(" ", "").replace("/", "-")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.plots_only:
        project_dir = _project_dir(_project_name(args))
        rows = _collect_rows(project_dir)
        if not rows:
            raise RuntimeError(
                f"No completed regularized quadratic summaries found under {project_dir}."
            )
        _write_outputs(project_dir, rows, optimizer=str(args.optimizer))
        print(f"Wrote {len(rows)} regularized quadratic sweep rows under {project_dir}.")
        return

    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(
        _build_launch_plan(args),
        args=args,
        argv=original_argv,
        cwd=REPO_ROOT,
    )


if __name__ == "__main__":
    main()
