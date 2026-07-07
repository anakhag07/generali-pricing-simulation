"""Run the combined noise-level x theta-offset planted-logistic grid.

Combines the two existing 1D sweeps (theta-offset at fixed noise, noise at
fixed theta0) into a 2D grid per noise family: for each noise level, vary the
initialization offset $$\\delta$$ in $$\\theta_0 = \\theta^{FO}_{clean} +
\\delta\\,\\mathbf{1}$$ (the same scalar added to every coordinate of the saved
clean first-order truth theta). Run settings are imported from
``scripts/run_sweep.py`` so new runs stay comparable with the saved sweeps; the
fixed-noise theta-offset sweeps (homoskedastic std 0.5, heteroskedastic growth
1.0) are reused as one curve per family instead of being rerun.

Outputs per family project (``homoskedastic-noise-offset-grid`` /
``heteroskedastic-noise-offset-grid``): ``noise_offset_grid_finals.csv`` and,
per estimator, a two-panel figure (final theta distance to the clean
first-order truth | clean-objective gap on the reconstructed train batch) with
one curve per noise level. ``--plots-only`` regenerates outputs from saved
summaries without running. Serial mode skips variant folders that already
contain both estimators (like ``run_sweep.py``); per-seed resume is only
available through ``--launch slurm --array``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from experiments.configs import get_config  # noqa: E402
from experiments.execution import execute_experiment_run  # noqa: E402
from experiments.launch import (  # noqa: E402
    LaunchContext,
    LaunchPlan,
    add_launch_args,
    read_task_records,
    run_launch_plan,
)
from experiments.paths import results_root  # noqa: E402
from experiments.reporting.context import create_run_context  # noqa: E402
from experiments.seeds import replicate_seed_setup  # noqa: E402
from experiments.sweep_utils import expand_sweep_overrides  # noqa: E402
from objective.base import sample_states  # noqa: E402
from objective.objectives import PlantedLogisticObjective  # noqa: E402
from objective.policy import IdentityFeatureMap, SoftmaxPolicy  # noqa: E402
from scripts import run_sweep  # noqa: E402


# =============================================================================
# Grid definition
# Offsets are a subset of run_sweep.THETA_OFFSETS so the saved fixed-noise
# theta-offset sweeps contribute one reused curve per family. Seeds are a
# subset of run_sweep.RUN_SEEDS for a smaller budget; reused curves are
# filtered to the same seeds so error bars stay comparable.
# =============================================================================

LAUNCH_PLAN_NAME = "planted-noise-offset-grid"
HOMO_PROJECT_NAME = "homoskedastic-noise-offset-grid"
HETERO_PROJECT_NAME = "heteroskedastic-noise-offset-grid"
REQUIRED_ESTIMATORS = run_sweep.REQUIRED_ESTIMATORS
RUN_SEEDS: tuple[int, ...] = (7, 8, 9)

GRID_THETA_OFFSETS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
HOMO_NEW_NOISE_STDS = (0.0, 0.1, 2.0)
HETERO_NEW_NOISE_GROWTHS = (0.0, 0.25, 4.0)

ESTIMATOR_LABELS = {
    "finite_difference": "Finite difference",
    "stein_difference": "Stein difference",
}

X_AXIS_LABEL = (
    r"Init offset $\delta$ in "
    r"$\theta_0 = \theta^{\mathrm{FO}}_{\mathrm{clean}} + \delta\,\mathbf{1}$"
    "\n"
    r"(same scalar $\delta$ added to every coordinate of the clean first-order truth $\theta$)"
)
THETA_DISTANCE_LABEL = r"$\|\hat{\theta}_{\mathrm{final}} - \theta^{\mathrm{FO}}_{\mathrm{clean}}\|_2$"
OBJECTIVE_GAP_LABEL = (
    r"$J_{\mathrm{clean}}(\hat{\theta}_{\mathrm{final}}) - "
    r"J_{\mathrm{clean}}(\theta^{\mathrm{FO}}_{\mathrm{clean}})$ (train batch)"
)


@dataclass(frozen=True)
class GridFamily:
    """One noise family (adapter type) of the combined grid."""

    key: str
    project_name: str
    reused_project_name: str
    reused_noise_level: float
    new_noise_levels: tuple[float, ...]
    noise_prefix: str
    axis_key: str
    noise_symbol: str
    legend_title: str
    noise_model_label: str
    noisy_objective: Callable[[float], object]


HOMO_FAMILY = GridFamily(
    key="homoskedastic",
    project_name=HOMO_PROJECT_NAME,
    reused_project_name=run_sweep.THETA_PROJECT_NAME,
    reused_noise_level=float(run_sweep.NOISE_STD),
    new_noise_levels=HOMO_NEW_NOISE_STDS,
    noise_prefix="noise-std",
    axis_key="noise_std",
    noise_symbol=r"\sigma",
    legend_title=r"constant noise std $\sigma$",
    noise_model_label=(
        r"homoskedastic noise $\hat{M}(x,u) = M(x,u) + \varepsilon(x,u)$, "
        r"$\varepsilon \sim \mathcal{N}(0, \sigma^2)$"
    ),
    noisy_objective=run_sweep._noisy_objective,
)
HETERO_FAMILY = GridFamily(
    key="heteroskedastic",
    project_name=HETERO_PROJECT_NAME,
    reused_project_name=run_sweep.HETERO_THETA_PROJECT_NAME,
    reused_noise_level=float(run_sweep.NOISE_GROWTH),
    new_noise_levels=HETERO_NEW_NOISE_GROWTHS,
    noise_prefix="noise-growth",
    axis_key="noise_growth",
    noise_symbol=r"\gamma",
    legend_title=r"noise growth $\gamma$ in $\sigma(u) = \gamma\,|u - u^\ast|$",
    noise_model_label=(
        r"heteroskedastic noise, std $\sigma(u) = \gamma\,|u - u^\ast|$ "
        r"(noiseless at the planted optimum $u^\ast$)"
    ),
    noisy_objective=run_sweep._hetero_noisy_objective,
)
FAMILY_GROUPS: dict[str, tuple[GridFamily, ...]] = {
    "homoskedastic": (HOMO_FAMILY,),
    "heteroskedastic": (HETERO_FAMILY,),
    "all": (HOMO_FAMILY, HETERO_FAMILY),
}


def _grid_run_name(family: GridFamily, noise_level: float, offset: float) -> str:
    level_part = run_sweep._format_sweep_value(noise_level)
    offset_part = run_sweep._format_sweep_value(offset)
    return f"{family.noise_prefix}-{level_part}__theta-offset-{offset_part}"


def _parse_grid_variant(family: GridFamily, variant_name: str) -> tuple[float, float] | None:
    prefix = f"{family.noise_prefix}-"
    separator = "__theta-offset-"
    if not variant_name.startswith(prefix) or separator not in variant_name:
        return None
    level_part, offset_part = variant_name.removeprefix(prefix).split(separator, 1)
    try:
        return float(level_part), float(offset_part)
    except ValueError:
        return None


def _build_grid_override_list(family: GridFamily) -> list[dict[str, object]]:
    return [
        {
            "_run_name": _grid_run_name(family, noise_level, offset),
            **run_sweep.COMMON_OVERRIDES,
            "objective": family.noisy_objective(noise_level),
            "theta0": run_sweep._theta0(offset),
        }
        for noise_level in family.new_noise_levels
        for offset in GRID_THETA_OFFSETS
    ]


# =============================================================================
# Launch wiring
# Mirrors run_sweep.py: array tasks run one (family variant, seed), the serial
# path fills in missing variants, and both paths regenerate the grid plots.
# =============================================================================


def _task_specs(families: Sequence[GridFamily]) -> list[tuple[str, str, dict[str, Any], int]]:
    specs: list[tuple[str, str, dict[str, Any], int]] = []
    for family in families:
        variants = expand_sweep_overrides(
            base_preset=run_sweep.BASE_PRESET,
            override_list=_build_grid_override_list(family),
            display_keys=(),
        )
        for variant_name, overrides in variants:
            for seed in RUN_SEEDS:
                specs.append((family.project_name, variant_name, dict(overrides), int(seed)))
    return specs


def _run_grid_task(
    index: int, context: LaunchContext, *, families: Sequence[GridFamily]
) -> dict[str, object]:
    del context
    project_name, variant_name, overrides, seed = _task_specs(families)[index]
    variant_dir = run_sweep._variant_dir(project_name, variant_name)
    seed_summary = variant_dir / f"summary-seed-{seed}.json"
    payload = {
        "project": project_name,
        "variant": variant_name,
        "run_seed": seed,
        "run_dir": str(variant_dir / "seeds" / f"seed-{seed}"),
        "summary_json": str(seed_summary),
    }
    if run_sweep._summary_has_estimators(seed_summary, REQUIRED_ESTIMATORS):
        print(f"Skipping completed task '{variant_name}' seed {seed} in '{project_name}'.")
        return payload
    seed_setup = replicate_seed_setup(
        seed,
        run_sweep.ANCHOR_SEED,
        vary=run_sweep.VARY,
        fixed=run_sweep.FIXED_SEEDS,
    )
    config = get_config(run_sweep.BASE_PRESET, overrides={**overrides, "seed_setup": seed_setup})
    run_context = create_run_context(
        variant_name,
        run_dir=variant_dir / "seeds" / f"seed-{seed}",
    )
    executed = execute_experiment_run(
        variant_name,
        config,
        run_context=run_context,
        reporter_stack_factory=run_sweep._seed_reporter_stack_factory(variant_dir, seed),
    )
    return {**payload, "run_dir": str(executed.run_context.run_dir)}


def _run_grid_serial(context: LaunchContext, *, families: Sequence[GridFamily]) -> None:
    del context
    n_runs = 0
    for family in families:
        n_runs += run_sweep._run_missing_sweep(
            base_preset=run_sweep.BASE_PRESET,
            project_name=family.project_name,
            override_list=_build_grid_override_list(family),
            run_seeds=RUN_SEEDS,
            vary=run_sweep.VARY,
            anchor_seed=run_sweep.ANCHOR_SEED,
            fixed=run_sweep.FIXED_SEEDS,
            display_keys=(),
            required_estimators=REQUIRED_ESTIMATORS,
        )
    regenerate_grid_plots(families)
    print(f"Completed {n_runs} total missing grid runs for preset '{run_sweep.BASE_PRESET}'.")


def _collect_grid_tasks(context: LaunchContext, *, families: Sequence[GridFamily]) -> None:
    records = read_task_records(context)
    expected = len(_task_specs(families))
    if len(records) != expected:
        raise RuntimeError(
            f"Expected {expected} task records under {context.tasks_dir}, found {len(records)}."
        )
    regenerate_grid_plots(families)
    print(f"Collected {len(records)} grid array tasks under {results_root()}.")


def _build_launch_plan(families: Sequence[GridFamily]) -> LaunchPlan:
    return LaunchPlan(
        name=LAUNCH_PLAN_NAME,
        task_count=len(_task_specs(families)),
        requires_jax=False,
        run_task=partial(_run_grid_task, families=families),
        run_all=partial(_run_grid_serial, families=families),
        collect=partial(_collect_grid_tasks, families=families),
        default_launch="auto",
        default_array=False,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        choices=tuple(FAMILY_GROUPS),
        default="all",
        help="Which noise-family grid(s) to run (default: all).",
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate grid CSVs/plots from saved summaries without running.",
    )
    add_launch_args(parser, default_launch="auto", default_array=False)
    return parser.parse_args(argv)


# =============================================================================
# Metric reconstruction
# summary-seed-*.json stores the NOISY final objective, so the clean-objective
# gap is recomputed by rebuilding the planted-logistic objective from the
# summary config and resampling the train split from data_seed/split_seed.
# =============================================================================


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _train_x(summary: dict[str, Any]) -> np.ndarray:
    config = summary["config"]
    if config.get("x_fixed_shape") is not None:
        raise ValueError("This grid script only supports synthetic sample_states runs.")
    seeds = config["resolved_seed_setup"]
    x_all = sample_states(
        np.random.default_rng(int(seeds["data_seed"])),
        int(config["n_samples"]),
        int(config["state_dim"]),
    )
    test_fraction = float(config.get("test_fraction", 0.0))
    if test_fraction == 0.0:
        return x_all
    shuffled = np.random.default_rng(int(seeds["split_seed"])).permutation(x_all.shape[0]).astype(int)
    n_test = int(round(test_fraction * x_all.shape[0]))
    n_test = min(max(n_test, 1), x_all.shape[0] - 1)
    return x_all[shuffled[n_test:]]


def _base_objective(summary: dict[str, Any]) -> PlantedLogisticObjective:
    objective = summary["config"]["objective"]
    base = objective.get("base_objective", objective)
    if base.get("type") != "PlantedLogisticObjective":
        raise ValueError(f"Expected PlantedLogisticObjective, found {base.get('type')!r}")
    policy_config = base["policy"]
    feature_map = policy_config.get("feature_map", {})
    if policy_config.get("type") != "SoftmaxPolicy" or feature_map.get("kind") != "identity":
        raise ValueError("This grid script expects identity-feature SoftmaxPolicy summaries.")
    policy = SoftmaxPolicy(
        feature_map=IdentityFeatureMap(),
        action_low=float(policy_config["action_low"]),
        action_high=float(policy_config["action_high"]),
    )
    return PlantedLogisticObjective.from_parameters(
        policy=policy,
        alpha=float(base["alpha"]),
        beta=np.asarray(base["beta"], dtype=float),
        bias=float(base["bias"]),
        u_star=float(base["u_star"]),
    )


def _variant_rows(
    variant_dir: Path,
    family: GridFamily,
    noise_level: float,
    offset: float,
    truth_theta: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in RUN_SEEDS:
        summary_path = variant_dir / f"summary-seed-{seed}.json"
        if not summary_path.exists():
            continue
        try:
            summary = _load_json(summary_path)
        except (OSError, json.JSONDecodeError):
            continue
        base_objective = _base_objective(summary)
        x_train = _train_x(summary)
        j_clean_truth = float(base_objective.value(truth_theta, x_train))
        for estimator in REQUIRED_ESTIMATORS:
            estimator_payload = summary.get("estimators", {}).get(estimator)
            if estimator_payload is None or "theta" not in estimator_payload:
                continue
            theta_hat = np.asarray(estimator_payload["theta"], dtype=float)
            j_clean_hat = float(base_objective.value(theta_hat, x_train))
            rows.append(
                {
                    family.axis_key: noise_level,
                    "theta_offset": offset,
                    "estimator": estimator,
                    "run_seed": seed,
                    "theta_distance_to_truth": float(np.linalg.norm(theta_hat - truth_theta)),
                    "clean_objective_gap": j_clean_hat - j_clean_truth,
                    "optimizer_success": estimator_payload.get("optimizer_success", ""),
                    "summary_path": str(summary_path),
                }
            )
    return rows


def _collect_family_rows(family: GridFamily, truth_theta: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    project_dir = run_sweep._project_dir(family.project_name)
    if project_dir.is_dir():
        for variant_dir in sorted(project_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            parsed = _parse_grid_variant(family, variant_dir.name)
            if parsed is None:
                continue
            noise_level, offset = parsed
            rows.extend(_variant_rows(variant_dir, family, noise_level, offset, truth_theta))
    reused_dir = results_root() / run_sweep._path_part(family.reused_project_name)
    if reused_dir.is_dir():
        for offset in GRID_THETA_OFFSETS:
            variant_dir = reused_dir / f"theta-offset-{run_sweep._format_sweep_value(offset)}"
            if variant_dir.is_dir():
                rows.extend(
                    _variant_rows(variant_dir, family, family.reused_noise_level, offset, truth_theta)
                )
    return rows


# =============================================================================
# Outputs: per-family CSV plus per-estimator two-panel figures
# =============================================================================


def regenerate_grid_plots(families: Sequence[GridFamily]) -> None:
    """Rebuild grid CSVs/plots from saved summaries (new grid + reused sweeps)."""
    truth_summary = run_sweep._first_order_truth_summary()
    if not truth_summary.exists():
        print(f"Skipping grid plots; missing truth summary: {truth_summary}")
        return
    truth_theta = run_sweep._theta_from_summary(truth_summary, "first_order")
    for family in families:
        rows = _collect_family_rows(family, truth_theta)
        if not rows:
            print(f"Skipping grid plots for '{family.project_name}'; no summary rows found.")
            continue
        project_dir = run_sweep._project_dir(family.project_name)
        project_dir.mkdir(parents=True, exist_ok=True)
        _write_grid_csv(project_dir / "noise_offset_grid_finals.csv", family, rows)
        for estimator in REQUIRED_ESTIMATORS:
            estimator_rows = [row for row in rows if row["estimator"] == estimator]
            if not estimator_rows:
                continue
            plot_path = project_dir / f"noise_offset_grid_{estimator}.png"
            _plot_family_estimator(plot_path, family, estimator, estimator_rows)
            print(f"Wrote grid plot {plot_path}")


def _write_grid_csv(path: Path, family: GridFamily, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        family.axis_key,
        "theta_offset",
        "estimator",
        "run_seed",
        "theta_distance_to_truth",
        "clean_objective_gap",
        "optimizer_success",
        "summary_path",
    ]
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row[family.axis_key]),
            float(row["theta_offset"]),
            str(row["estimator"]),
            int(row["run_seed"]),
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow(row)


def _metric_stats(
    rows: list[dict[str, object]], metric: str, offset: float
) -> tuple[float, float]:
    values = [float(row[metric]) for row in rows if float(row["theta_offset"]) == offset]
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _plot_family_estimator(
    path: Path,
    family: GridFamily,
    estimator: str,
    rows: list[dict[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import colormaps
    import matplotlib.pyplot as plt

    metrics = (
        ("theta_distance_to_truth", THETA_DISTANCE_LABEL),
        ("clean_objective_gap", OBJECTIVE_GAP_LABEL),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))
    noise_levels = sorted({float(row[family.axis_key]) for row in rows})
    all_offsets = sorted({float(row["theta_offset"]) for row in rows})
    cmap = colormaps["viridis"]
    for level_index, noise_level in enumerate(noise_levels):
        level_rows = [row for row in rows if float(row[family.axis_key]) == noise_level]
        offsets = sorted({float(row["theta_offset"]) for row in level_rows})
        color = cmap(0.85 * level_index / max(len(noise_levels) - 1, 1))
        label = rf"${family.noise_symbol} = {noise_level:g}$"
        if noise_level == 0.0:
            label += " (clean)"
        for ax, (metric, _) in zip(axes, metrics):
            means_stds = [_metric_stats(level_rows, metric, offset) for offset in offsets]
            means = [mean for mean, _ in means_stds]
            stds = [std for _, std in means_stds]
            if metric == "theta_distance_to_truth":
                # Distance is nonnegative: never draw error bars below zero.
                yerr = np.vstack([np.minimum(stds, means), stds])
            else:
                yerr = np.vstack([stds, stds])
            ax.errorbar(
                offsets,
                means,
                yerr=yerr if any(std > 0.0 for std in stds) else None,
                label=label,
                color=color,
                marker="o",
                linewidth=1.8,
                markersize=5.0,
                capsize=3.0,
            )
    for ax, (metric, y_label) in zip(axes, metrics):
        run_sweep._set_symlog_ticks(ax, all_offsets)
        metric_values = [float(row[metric]) for row in rows]
        if metric == "theta_distance_to_truth":
            positive = [value for value in metric_values if value > 0.0]
            if len(positive) == len(metric_values):
                ax.set_yscale("log")
            else:
                # Exact-zero distances (runs that recover the truth theta):
                # keep the axis nonnegative instead of wasting a mirrored
                # negative symlog half.
                linthresh = 0.5 * min(positive) if positive else 1e-8
                ax.set_yscale("symlog", linthresh=linthresh)
                ax.set_ylim(bottom=0.0)
        else:
            ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_xlabel(X_AXIS_LABEL)
        ax.set_ylabel(y_label)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(title=family.legend_title)
    fig.suptitle(
        f"{ESTIMATOR_LABELS.get(estimator, estimator)} — {family.noise_model_label}\n"
        r"mean $\pm$ std over run seeds "
        f"{RUN_SEEDS}; curves = noise level, x = init offset",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =============================================================================
# Entry point
# =============================================================================


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    families = FAMILY_GROUPS[args.families]
    if args.plots_only:
        regenerate_grid_plots(families)
        return
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(_build_launch_plan(families), args=args, argv=original_argv)


if __name__ == "__main__":
    main()
