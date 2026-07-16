"""Sweep theta-keyed homoskedastic noise and finite-difference radius on a quadratic."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

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
from objective.noise import HomoskedasticGaussianNoise, NoisyObjective  # noqa: E402
from objective.objectives import StronglyConvexQuadratic  # noqa: E402


BASE_PRESET = "synthetic_quadratic_base"
PROJECT_NAME = "quadratic-homoskedastic-lbfgsb-sweep"
PILOT_PROJECT_NAME = "quadratic-homoskedastic-lbfgsb-pilot"
OPTAX_PROJECT_NAME = "quadratic-homoskedastic-optax-adam-sweep"
OPTAX_PILOT_PROJECT_NAME = "quadratic-homoskedastic-optax-adam-pilot"
ESTIMATOR = "finite_difference"
L_BFGS_B = "l-bfgs-b"
OPTAX_ADAM = "optax-adam"
OPTIMIZERS = (L_BFGS_B, OPTAX_ADAM)

DEFAULT_DIMENSION = 10
DEFAULT_STEP_SIZE = 0.05
DEFAULT_ARRAY_MAX_PARALLEL = 2
DEFAULT_T_STEPS = {
    L_BFGS_B: 200,
    OPTAX_ADAM: 2000,
}
DEFAULT_NOISE_STDS = (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
DEFAULT_FD_RADII = (1e-4, 1e-3, 1e-2, 1e-1)
DEFAULT_RUN_SEEDS = tuple(range(7, 27))

PILOT_NOISE_STDS = (0.0, 1e-6, 1e-4, 1e-2)
PILOT_FD_RADII = (1e-3, 1e-2, 1e-1)
PILOT_RUN_SEEDS = tuple(range(7, 12))

FINAL_FIELDNAMES = (
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

SUMMARY_METRICS = (
    "final_theta_norm",
    "clean_final_objective",
    "noisy_final_objective",
    "exploitation_gap",
    "clean_improvement",
    "runtime_sec",
    "trace_steps",
)
SUMMARY_STATS = ("mean", "std", "min", "max", "median")
SUMMARY_FIELDNAMES = (
    "noise_std",
    "fd_radius",
    "noise_to_radius",
    "n_seeds",
    "optimizer_success_rate",
    *(f"{metric}_{stat}" for metric in SUMMARY_METRICS for stat in SUMMARY_STATS),
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="Run the reduced 60-run calibration grid.")
    parser.add_argument("--plots-only", action="store_true", help="Rebuild CSVs and plots from saved summaries.")
    parser.add_argument("--dimension", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--noise-stds", type=float, nargs="+", default=None)
    parser.add_argument("--fd-radii", type=float, nargs="+", default=None)
    parser.add_argument("--run-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--optimizer", choices=OPTIMIZERS, default=L_BFGS_B)
    parser.add_argument("--t-steps", type=int, default=None)
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--project-name", default=None)
    add_launch_args(parser)
    parser.set_defaults(array_max_parallel=DEFAULT_ARRAY_MAX_PARALLEL)
    return parser.parse_args(argv)


def _resolved_grid(args: argparse.Namespace) -> tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]:
    default_noise = PILOT_NOISE_STDS if args.pilot else DEFAULT_NOISE_STDS
    default_radii = PILOT_FD_RADII if args.pilot else DEFAULT_FD_RADII
    default_seeds = PILOT_RUN_SEEDS if args.pilot else DEFAULT_RUN_SEEDS
    noise_stds = tuple(float(value) for value in (args.noise_stds or default_noise))
    fd_radii = tuple(float(value) for value in (args.fd_radii or default_radii))
    run_seeds = tuple(int(value) for value in (args.run_seeds or default_seeds))
    if not noise_stds or any(not np.isfinite(value) or value < 0.0 for value in noise_stds):
        raise ValueError("noise standard deviations must be finite and nonnegative.")
    if not fd_radii or any(not np.isfinite(value) or value <= 0.0 for value in fd_radii):
        raise ValueError("finite-difference radii must be finite and positive.")
    if not run_seeds:
        raise ValueError("run seeds must contain at least one value.")
    return noise_stds, fd_radii, run_seeds


def _project_name(args: argparse.Namespace) -> str:
    if args.project_name:
        return str(args.project_name)
    if args.optimizer == OPTAX_ADAM:
        return OPTAX_PILOT_PROJECT_NAME if args.pilot else OPTAX_PROJECT_NAME
    return PILOT_PROJECT_NAME if args.pilot else PROJECT_NAME


def _resolved_t_steps(args: argparse.Namespace) -> int:
    if args.t_steps is not None:
        return int(args.t_steps)
    return DEFAULT_T_STEPS[str(args.optimizer)]


def _task_specs(args: argparse.Namespace) -> list[tuple[float, float]]:
    noise_stds, fd_radii, _ = _resolved_grid(args)
    return [(noise_std, fd_radius) for noise_std in noise_stds for fd_radius in fd_radii]


def _variant_name(noise_std: float, fd_radius: float) -> str:
    return f"noise-std-{_value_label(noise_std)}__fd-radius-{_value_label(fd_radius)}"


def _parse_variant(name: str) -> tuple[float, float] | None:
    prefix = "noise-std-"
    separator = "__fd-radius-"
    if not name.startswith(prefix) or separator not in name:
        return None
    noise_text, radius_text = name.removeprefix(prefix).split(separator, 1)
    try:
        return float(noise_text), float(radius_text)
    except ValueError:
        return None


def _build_override_list(
    *,
    dimension: int,
    noise_stds: Sequence[float],
    fd_radii: Sequence[float],
    t_steps: int,
    optimizer: str,
    step_size: float,
) -> list[dict[str, object]]:
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    if t_steps <= 0:
        raise ValueError("t_steps must be positive.")
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}.")
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive.")
    theta0 = np.ones(dimension, dtype=float) / np.sqrt(float(dimension))
    base_objective = StronglyConvexQuadratic.isotropic(dimension)
    return [
        {
            "_run_name": _variant_name(float(noise_std), float(fd_radius)),
            "dimension": int(dimension),
            "objective": NoisyObjective(
                base_objective=base_objective,
                noise=HomoskedasticGaussianNoise(std=float(noise_std)),
            ),
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
            "correctness": CorrectnessSpec(gradient_source="denoised_exact"),
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
        }
        for noise_std in noise_stds
        for fd_radius in fd_radii
    ]


def _run_grid(
    *,
    project_name: str,
    dimension: int,
    noise_stds: Sequence[float],
    fd_radii: Sequence[float],
    run_seeds: Sequence[int],
    t_steps: int,
    optimizer: str,
    step_size: float,
) -> Path:
    sweep = run_sweep(
        base_preset=BASE_PRESET,
        run_seeds=tuple(int(seed) for seed in run_seeds),
        override_list=_build_override_list(
            dimension=dimension,
            noise_stds=noise_stds,
            fd_radii=fd_radii,
            t_steps=t_steps,
            optimizer=optimizer,
            step_size=step_size,
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
        noise_std, fd_radius = parsed
        for summary_path in sorted(variant_dir.glob("summary-seed-*.json")):
            try:
                summary = _load_json(summary_path)
                row = _summary_row(summary, summary_path, noise_std=noise_std, fd_radius=fd_radius)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                continue
            rows.append(row)
    return sorted(rows, key=lambda row: (float(row["noise_std"]), float(row["fd_radius"]), int(row["run_seed"])))


def _run_grid_task(
    index: int,
    context: LaunchContext,
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    del context
    noise_std, fd_radius = _task_specs(args)[index]
    _, _, run_seeds = _resolved_grid(args)
    project_name = _project_name(args)
    project_dir = _run_grid(
        project_name=project_name,
        dimension=int(args.dimension),
        noise_stds=(noise_std,),
        fd_radii=(fd_radius,),
        run_seeds=run_seeds,
        t_steps=_resolved_t_steps(args),
        optimizer=str(args.optimizer),
        step_size=float(args.step_size),
    )
    variant_name = _variant_name(noise_std, fd_radius)
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
        f"Completed quadratic task {index}: {variant_name} "
        f"({len(summary_paths)} seeds)."
    )
    return {
        "project_name": project_name,
        "variant_name": variant_name,
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
        noise_stds=noise_stds,
        fd_radii=fd_radii,
        run_seeds=run_seeds,
        t_steps=_resolved_t_steps(args),
        optimizer=str(args.optimizer),
        step_size=float(args.step_size),
    )
    rows = _collect_rows(project_dir)
    if not rows:
        raise RuntimeError(f"No completed quadratic sweep summaries found under {project_dir}.")
    _write_outputs(project_dir, rows, optimizer=str(args.optimizer))
    print(f"Wrote {len(rows)} quadratic sweep rows under {project_dir}.")


def _collect_grid_tasks(context: LaunchContext, *, args: argparse.Namespace) -> None:
    expected_indices = set(range(len(_task_specs(args))))
    records = read_task_records(context)
    actual_indices = {int(record["task_index"]) for record in records}
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        unexpected = sorted(actual_indices - expected_indices)
        raise RuntimeError(
            "Cannot collect incomplete quadratic array: "
            f"missing task indices={missing}, unexpected task indices={unexpected}."
        )
    payloads = task_payloads(context)
    rows = _rows_from_task_payloads(payloads)
    if not rows:
        raise RuntimeError("No completed quadratic sweep rows were produced by array tasks.")
    project_dir = _project_dir(_project_name(args))
    _write_outputs(project_dir, rows, optimizer=str(args.optimizer))
    print(f"Collected {len(payloads)} tasks and wrote {len(rows)} rows under {project_dir}.")


def _rows_from_task_payloads(
    payloads: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for payload in payloads:
        noise_std = float(payload["noise_std"])
        fd_radius = float(payload["fd_radius"])
        summary_paths = payload.get("summary_paths")
        if not isinstance(summary_paths, list):
            raise TypeError("Quadratic task payload must contain a summary_paths list.")
        for summary_value in summary_paths:
            summary_path = Path(str(summary_value))
            summary = _load_json(summary_path)
            rows.append(
                _summary_row(
                    summary,
                    summary_path,
                    noise_std=noise_std,
                    fd_radius=fd_radius,
                )
            )
    return sorted(
        rows,
        key=lambda row: (
            float(row["noise_std"]),
            float(row["fd_radius"]),
            int(row["run_seed"]),
        ),
    )


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
    dimension = len(config["objective"]["base_objective"]["w_star"])
    return {
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
    keys = sorted({(float(row["noise_std"]), float(row["fd_radius"])) for row in rows})
    output: list[dict[str, object]] = []
    for noise_std, fd_radius in keys:
        group = [
            row
            for row in rows
            if float(row["noise_std"]) == noise_std and float(row["fd_radius"]) == fd_radius
        ]
        summary: dict[str, object] = {
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
    _write_csv(project_dir / "quadratic_homoskedastic_finals.csv", FINAL_FIELDNAMES, rows)
    _write_csv(project_dir / "quadratic_homoskedastic_summary.csv", SUMMARY_FIELDNAMES, summaries)
    if plot and summaries:
        _write_plots(project_dir / "plots", rows, summaries, optimizer=optimizer)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_plots(
    plot_dir: Path,
    final_rows: Sequence[Mapping[str, object]],
    summary_rows: Sequence[Mapping[str, object]],
    *,
    optimizer: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    _plot_heatmap(
        plot_dir / "final_theta_norm_heatmap.png",
        summary_rows,
        metric="final_theta_norm_median",
        title=r"Median final $\|\hat{\theta}\|_2$ (log10)",
        transform=lambda value: np.log10(max(value, 1e-16)),
        colorbar_label=r"$\log_{10}$ median final $\|\hat{\theta}\|_2$",
    )
    _plot_heatmap(
        plot_dir / "optimizer_success_rate_heatmap.png",
        summary_rows,
        metric="optimizer_success_rate",
        title=f"{_optimizer_label(optimizer)} success rate",
        transform=float,
        colorbar_label="success rate",
        vmin=0.0,
        vmax=1.0,
    )

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    radii = sorted({float(row["fd_radius"]) for row in summary_rows})
    for radius in radii:
        selected = sorted(
            (
                row
                for row in summary_rows
                if float(row["fd_radius"]) == radius and float(row["noise_to_radius"]) > 0.0
            ),
            key=lambda row: float(row["noise_to_radius"]),
        )
        if not selected:
            continue
        ax.plot(
            [float(row["noise_to_radius"]) for row in selected],
            [max(float(row["final_theta_norm_median"]), 1e-16) for row in selected],
            marker="o",
            label=rf"$h={radius:g}$",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"noise-to-radius ratio $\tau/h$")
    ax.set_ylabel(r"median final $\|\hat{\theta}\|_2$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="FD radius")
    fig.tight_layout()
    fig.savefig(plot_dir / "theta_error_vs_noise_to_radius.png", dpi=200)
    plt.close(fig)

    clean = np.asarray([float(row["clean_final_objective"]) for row in final_rows], dtype=float)
    noisy = np.asarray([float(row["noisy_final_objective"]) for row in final_rows], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    scatter = ax.scatter(clean, noisy, c=[float(row["noise_std"]) for row in final_rows], cmap="viridis", alpha=0.75)
    low = float(min(np.min(clean), np.min(noisy)))
    high = float(max(np.max(clean), np.max(noisy)))
    ax.plot([low, high], [low, high], linestyle="--", color="black", linewidth=1.0, label="no exploitation gap")
    ax.set_xlabel("clean final objective")
    ax.set_ylabel("noisy final objective seen by optimizer")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.colorbar(scatter, ax=ax, label=r"noise std $\tau$")
    fig.tight_layout()
    fig.savefig(plot_dir / "clean_vs_noisy_final_objective.png", dpi=200)
    plt.close(fig)


def _plot_heatmap(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    metric: str,
    title: str,
    transform,
    colorbar_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    noise_stds = sorted({float(row["noise_std"]) for row in rows})
    fd_radii = sorted({float(row["fd_radius"]) for row in rows})
    matrix = np.full((len(noise_stds), len(fd_radii)), np.nan, dtype=float)
    for row in rows:
        noise_index = noise_stds.index(float(row["noise_std"]))
        radius_index = fd_radii.index(float(row["fd_radius"]))
        matrix[noise_index, radius_index] = float(transform(float(row[metric])))
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    image = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(fd_radii)), labels=[f"{value:g}" for value in fd_radii])
    ax.set_yticks(range(len(noise_stds)), labels=[f"{value:g}" for value in noise_stds])
    ax.set_xlabel(r"finite-difference radius $h$")
    ax.set_ylabel(r"homoskedastic noise std $\tau$")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _value_label(value: float) -> str:
    return f"{float(value):g}"


def _optimizer_label(optimizer: str) -> str:
    if optimizer == L_BFGS_B:
        return "L-BFGS-B"
    if optimizer == OPTAX_ADAM:
        return "Optax Adam"
    raise ValueError(f"Unknown optimizer: {optimizer}")


def _project_dir(project_name: str) -> Path:
    return results_root() / str(project_name).replace(" ", "").replace("/", "-")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.plots_only:
        project_dir = _project_dir(_project_name(args))
        rows = _collect_rows(project_dir)
        if not rows:
            raise RuntimeError(
                f"No completed quadratic sweep summaries found under {project_dir}."
            )
        _write_outputs(project_dir, rows, optimizer=str(args.optimizer))
        print(f"Wrote {len(rows)} quadratic sweep rows under {project_dir}.")
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
