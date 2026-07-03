"""Run dense planted-logistic homoskedastic-noise sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.config import CorrectnessSpec
from experiments.configs import get_config
from experiments.slurm import assert_jax_gpu_available, submit_to_slurm_if_needed
from experiments.sweep_utils import run_sweep
from objective.noise import HomoskedasticGaussianNoise, NoisyObjective


# =============================================================================
# Experiment-specific sweep definitions
# Everything in this section may mention planted-logistic objectives, noise,
# concrete estimator choices, fixed seeds, theta offsets, or output project names.
# =============================================================================

BASE_PRESET = "planted_logistic_base"
THETA_PROJECT_NAME = "homoskedastic-theta-offset-sweep"
NOISE_PROJECT_NAME = "homoskedastic-noise-sweep"
FIRST_ORDER_TRUTH_SUMMARY = Path(
    "outputs/planted_logistic_base/first_order_truth_20260701_174139/summary.json"
)
# Backward-compatible aliases used by tests and older ad-hoc imports.
PROJECT_NAME = THETA_PROJECT_NAME
DISPLAY_KEYS: tuple[str, ...] = ()
REQUIRED_ESTIMATORS = ("finite_difference", "stein_difference")
NOISE_STD = 0.5

# These dense fill-in sweeps match the existing saved single-seed runs. Existing
# completed variant folders are skipped before dispatching to run_sweep().
RUN_SEEDS: tuple[int, ...] = (7,)
ANCHOR_SEED = 7
VARY: tuple[str, ...] = ("optimizer",)
FIXED_SEEDS: dict[str, int | None] = {"noise": 101}
BASE_THETA = np.asarray(
    [
        0.4054882808450241,
        0.00012799868045781167,
        -4.657524122982136e-05,
        6.221922280809605e-05,
    ],
    dtype=float,
)
THETA_OFFSETS = (
    0.0,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
    0.35,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    4.0,
    5.0,
    7.5,
    10.0,
)
NOISE_STDS = (
    0.0,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.25,
    0.35,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
)

_PLANTED_BASE = get_config(BASE_PRESET)


def _theta0(offset: float) -> np.ndarray:
    return BASE_THETA + float(offset)


def _noisy_objective(noise_std: float) -> NoisyObjective:
    return NoisyObjective(
        base_objective=_PLANTED_BASE.objective,
        noise=HomoskedasticGaussianNoise(std=float(noise_std)),
    )


COMMON_OVERRIDES: dict[str, object] = {
    "enabled_estimators": REQUIRED_ESTIMATORS,
    "correctness": CorrectnessSpec(gradient_source="denoised_exact"),
    "perturbation_space": "u",
    "step_rule": "l-bfgs-b",
    "t_steps": 1000,
    "step_size": 0.001,
    "n_samples": 1000,
    "sigma": 0.05,
    "n_grad_samples": 8,
    "plot": True,
    "verbose": False,
    "wandb_enabled": False,
}


def _build_theta_override_list() -> list[dict[str, object]]:
    return [
        {
            "_run_name": _axis_run_name("theta-offset", offset),
            **COMMON_OVERRIDES,
            "objective": _noisy_objective(NOISE_STD),
            "theta0": _theta0(offset),
        }
        for offset in THETA_OFFSETS
    ]


def _build_noise_override_list() -> list[dict[str, object]]:
    return [
        {
            "_run_name": _axis_run_name("noise-std", noise_std),
            **COMMON_OVERRIDES,
            "objective": _noisy_objective(noise_std),
            "theta0": np.zeros_like(BASE_THETA),
        }
        for noise_std in NOISE_STDS
    ]


# =============================================================================
# Reusable sweep-script helpers
# Helpers in this section must receive run-specific values as parameters. If a
# helper needs BASE_THETA, NOISE_STD, THETA_OFFSETS, NOISE_STDS, NoisyObjective,
# or a concrete estimator list, keep it in the experiment-specific section.
# =============================================================================


def _axis_run_name(axis: str, value: object) -> str:
    return f"{axis}-{_format_sweep_value(value)}"


def _format_sweep_value(value: object) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        text = str(value)
        return text.replace(" ", "").replace("/", "-")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the configured preset sweep.")
    parser.add_argument(
        "--no-sbatch",
        action="store_true",
        help="Run in the current process instead of auto-submitting to ORCD Slurm.",
    )
    return parser.parse_args(argv)


def _override_lists_require_jax(
    override_lists: Sequence[Sequence[Mapping[str, object]]],
) -> bool:
    return any(
        overrides.get("compute_backend") == "jax"
        for override_list in override_lists
        for overrides in override_list
    )


def _missing_overrides(
    *,
    project_name: str,
    override_list: Sequence[Mapping[str, object]],
    required_estimators: Sequence[str],
    runs_root: str = "outputs",
) -> list[dict[str, object]]:
    missing: list[dict[str, object]] = []
    project_dir = Path(runs_root) / project_name
    for overrides in override_list:
        run_name = overrides.get("_run_name")
        if run_name is None:
            raise ValueError("Resume/skipping requires each override to include '_run_name'.")
        if not _variant_is_completed(project_dir / str(run_name), required_estimators):
            missing.append(dict(overrides))
    return missing


def _variant_is_completed(variant_dir: Path, required_estimators: Sequence[str]) -> bool:
    if not variant_dir.is_dir():
        return False
    for summary_path in _summary_paths(variant_dir):
        if _summary_has_estimators(summary_path, required_estimators):
            return True
    return False


def _summary_paths(variant_dir: Path) -> list[Path]:
    paths = sorted(variant_dir.glob("summary-seed-*.json"))
    direct_summary = variant_dir / "summary.json"
    if direct_summary.exists():
        paths.append(direct_summary)
    paths.extend(sorted(variant_dir.glob("seeds/seed-*/summary.json")))
    paths.extend(sorted(variant_dir.glob("*/summary.json")))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def _summary_has_estimators(summary_path: Path, estimators: Sequence[str]) -> bool:
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    estimator_payload = payload.get("estimators", {})
    return all(name in estimator_payload for name in estimators)


def _run_missing_sweep(
    *,
    base_preset: str,
    project_name: str,
    override_list: Sequence[Mapping[str, object]],
    run_seeds: Sequence[int],
    vary: tuple[str, ...],
    anchor_seed: int,
    fixed: Mapping[str, int | None],
    display_keys: Sequence[str],
    required_estimators: Sequence[str],
    runs_root: str = "outputs",
) -> int:
    missing = _missing_overrides(
        project_name=project_name,
        override_list=override_list,
        required_estimators=required_estimators,
        runs_root=runs_root,
    )
    skipped = len(override_list) - len(missing)
    if not missing:
        print(f"No missing variants for '{project_name}' ({skipped} already complete).")
        return 0

    sweep = run_sweep(
        base_preset=base_preset,
        run_seeds=run_seeds,
        override_list=missing,
        vary=vary,
        anchor_seed=anchor_seed,
        fixed=fixed,
        runs_root=runs_root,
        project_name=project_name,
        display_keys=display_keys,
    )
    print(
        f"Completed {len(sweep.run_results)} missing runs for '{project_name}' "
        f"({len(missing)} variants x {len(run_seeds)} seeds; skipped {skipped})."
    )
    return len(sweep.run_results)


# =============================================================================
# Concrete sweep instances
# This section binds the experiment-specific definitions to the reusable helpers.
# =============================================================================


THETA_OVERRIDE_LIST = _build_theta_override_list()
NOISE_OVERRIDE_LIST = _build_noise_override_list()
OVERRIDE_LIST = THETA_OVERRIDE_LIST
SWEEPS: tuple[tuple[str, list[dict[str, object]]], ...] = (
    (THETA_PROJECT_NAME, THETA_OVERRIDE_LIST),
    (NOISE_PROJECT_NAME, NOISE_OVERRIDE_LIST),
)


# =============================================================================
# Experiment-specific distance-reporting helpers
# These helpers are intentionally tied to the two dense homoskedastic sweeps.
# =============================================================================


def _regenerate_distance_plots() -> None:
    if not FIRST_ORDER_TRUTH_SUMMARY.exists():
        print(f"Skipping distance plots; missing truth summary: {FIRST_ORDER_TRUTH_SUMMARY}")
        return
    truth_theta = _theta_from_summary(FIRST_ORDER_TRUTH_SUMMARY, "first_order")
    _write_distance_plot(
        project_name=THETA_PROJECT_NAME,
        truth_theta=truth_theta,
        axis_key="theta_offset",
        x_label="Theta offset added to first-order truth theta",
        title="Final theta distance to first-order truth by offset",
        csv_name="theta_distance_to_first_order_truth_by_offset.csv",
        plot_name="theta_distance_to_first_order_truth_by_offset.png",
    )
    _write_distance_plot(
        project_name=NOISE_PROJECT_NAME,
        truth_theta=truth_theta,
        axis_key="noise_std",
        x_label="Homoskedastic noise std",
        title="Final theta distance to first-order truth by noise",
        csv_name="theta_distance_to_first_order_truth_by_noise.csv",
        plot_name="theta_distance_to_first_order_truth_by_noise.png",
    )


def _write_distance_plot(
    *,
    project_name: str,
    truth_theta: np.ndarray,
    axis_key: str,
    x_label: str,
    title: str,
    csv_name: str,
    plot_name: str,
) -> None:
    project_dir = Path("outputs") / project_name
    rows = _collect_distance_rows(project_dir, truth_theta, axis_key)
    if not rows:
        print(f"Skipping distance plot for '{project_name}'; no summary rows found.")
        return
    _write_distance_csv(project_dir / csv_name, rows, axis_key)
    _plot_distance_rows(project_dir / plot_name, rows, axis_key, x_label, title)
    print(f"Wrote distance plot for '{project_name}' to {project_dir / plot_name}.")


def _collect_distance_rows(
    project_dir: Path,
    truth_theta: np.ndarray,
    axis_key: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not project_dir.is_dir():
        return rows
    for variant_dir in sorted(project_dir.iterdir(), key=_variant_sort_key):
        if not variant_dir.is_dir():
            continue
        for summary_path in _summary_paths(variant_dir):
            try:
                with summary_path.open("r", encoding="utf-8") as handle:
                    summary = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            axis_value = _distance_axis_value(axis_key, variant_dir.name, summary)
            if axis_value is None:
                continue
            for estimator in REQUIRED_ESTIMATORS:
                estimator_payload = summary.get("estimators", {}).get(estimator)
                if estimator_payload is None or "theta" not in estimator_payload:
                    continue
                theta = np.asarray(estimator_payload["theta"], dtype=float)
                rows.append(
                    {
                        axis_key: axis_value,
                        "estimator": estimator,
                        "distance_l2_to_truth": float(np.linalg.norm(theta - truth_theta)),
                        "optimizer_success": estimator_payload.get("optimizer_success", ""),
                        "final_value": float(estimator_payload["final_value"]),
                        "summary_path": str(summary_path),
                    }
                )
    return rows


def _distance_axis_value(
    axis_key: str,
    variant_name: str,
    summary: dict,
) -> float | None:
    if axis_key == "theta_offset" and variant_name.startswith("theta-offset-"):
        return float(variant_name.removeprefix("theta-offset-"))
    if axis_key == "noise_std":
        objective = summary.get("config", {}).get("objective", {})
        noise = objective.get("noise", {}) if isinstance(objective, dict) else {}
        if "std" in noise:
            return float(noise["std"])
        if variant_name.startswith("noise-std-"):
            return float(variant_name.removeprefix("noise-std-"))
    return None


def _theta_from_summary(summary_path: Path, estimator: str) -> np.ndarray:
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return np.asarray(summary["estimators"][estimator]["theta"], dtype=float)


def _write_distance_csv(path: Path, rows: list[dict[str, object]], axis_key: str) -> None:
    fieldnames = [
        axis_key,
        "estimator",
        "distance_l2_to_truth",
        "optimizer_success",
        "final_value",
        "summary_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (float(item[axis_key]), str(item["estimator"]))):
            writer.writerow(row)


def _plot_distance_rows(
    path: Path,
    rows: list[dict[str, object]],
    axis_key: str,
    x_label: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "finite_difference": {"label": "Finite difference", "color": "tab:blue", "marker": "o"},
        "stein_difference": {"label": "Stein difference", "color": "tab:orange", "marker": "s"},
    }
    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.6))
    x_values = sorted({float(row[axis_key]) for row in rows})
    for estimator in REQUIRED_ESTIMATORS:
        selected = [row for row in rows if row["estimator"] == estimator]
        if not selected:
            continue
        xs = sorted({float(row[axis_key]) for row in selected})
        means = [_distance_mean(selected, axis_key, x_value) for x_value in xs]
        stds = [_distance_std(selected, axis_key, x_value) for x_value in xs]
        style = styles[estimator]
        ax.errorbar(
            xs,
            means,
            yerr=stds if any(std > 0.0 for std in stds) else None,
            label=str(style["label"]),
            color=str(style["color"]),
            marker=str(style["marker"]),
            linewidth=1.8,
            markersize=5.5,
            capsize=3.0,
        )
        failed_xs = [x_value for x_value in xs if _distance_has_failure(selected, axis_key, x_value)]
        if failed_xs:
            ax.scatter(
                failed_xs,
                [_distance_mean(selected, axis_key, x_value) for x_value in failed_xs],
                label=f"{style['label']} optimizer_success=False",
                color=str(style["color"]),
                marker="x",
                s=60,
                linewidths=1.5,
                zorder=4,
            )
    _set_symlog_ticks(ax, x_values)
    distances = [float(row["distance_l2_to_truth"]) for row in rows]
    if all(distance > 0.0 for distance in distances):
        ax.set_yscale("log")
    else:
        ax.set_yscale("symlog", linthresh=1e-8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$||\theta_{estimator} - \theta_{first\ order\ truth}||_2$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _distance_mean(rows: list[dict[str, object]], axis_key: str, x_value: float) -> float:
    values = [float(row["distance_l2_to_truth"]) for row in rows if float(row[axis_key]) == x_value]
    return float(np.mean(values))


def _distance_std(rows: list[dict[str, object]], axis_key: str, x_value: float) -> float:
    values = [float(row["distance_l2_to_truth"]) for row in rows if float(row[axis_key]) == x_value]
    return float(np.std(values, ddof=0))


def _distance_has_failure(rows: list[dict[str, object]], axis_key: str, x_value: float) -> bool:
    return any(
        float(row[axis_key]) == x_value and row.get("optimizer_success") is False
        for row in rows
    )


def _set_symlog_ticks(ax: object, values: list[float]) -> None:
    nonzero = [abs(value) for value in values if value != 0.0]
    if nonzero:
        ax.set_xscale("symlog", linthresh=min(nonzero))
    ax.set_xticks(values)
    ax.set_xticklabels([f"{value:g}" for value in values], rotation=45, ha="right")


def _variant_sort_key(path: Path) -> tuple[int, float | str]:
    for prefix in ("theta-offset-", "noise-std-"):
        if path.name.startswith(prefix):
            return (0, float(path.name.removeprefix(prefix)))
    return (1, path.name)


# =============================================================================
# Entry point
# This section wires the concrete sweeps into the reusable launch helpers.
# =============================================================================


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    requires_jax = _override_lists_require_jax([override_list for _, override_list in SWEEPS])

    submission = submit_to_slurm_if_needed(
        requires_jax=requires_jax,
        no_sbatch=args.no_sbatch,
        argv=original_argv,
    )
    if submission is not None:
        print(
            f"Submitted {submission.profile.name} Slurm job {submission.job_id}; "
            f"logs: {submission.profile.output}"
        )
        return

    if requires_jax:
        jax_status = assert_jax_gpu_available([SimpleNamespace(compute_backend="jax")])
        if jax_status is not None:
            print(jax_status)

    n_runs = 0
    for project_name, override_list in SWEEPS:
        n_runs += _run_missing_sweep(
            base_preset=BASE_PRESET,
            project_name=project_name,
            override_list=override_list,
            run_seeds=RUN_SEEDS,
            vary=VARY,
            anchor_seed=ANCHOR_SEED,
            fixed=FIXED_SEEDS,
            display_keys=DISPLAY_KEYS,
            required_estimators=REQUIRED_ESTIMATORS,
        )
    _regenerate_distance_plots()
    print(f"Completed {n_runs} total missing sweep runs for preset '{BASE_PRESET}'.")


if __name__ == "__main__":
    main()
