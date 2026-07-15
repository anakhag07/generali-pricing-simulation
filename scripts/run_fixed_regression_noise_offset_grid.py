"""Run a fixed-regression homoskedastic/heteroskedastic noise-offset grid.

For each noise family, sweep a 2D grid over noise level and initialization
offset. The offset is the scalar ``delta`` added to every coordinate of the
clean first-order reference theta: ``theta0 = theta_clean + delta * 1``.

The script uses ``experiments.sweep_utils.run_sweep`` for the actual runs so
each variant follows the canonical seed-sweep layout (``summary-seed-<seed>.json``
at the variant root, heavy artifacts under ``seeds/seed-<seed>/``). Launch array
tasks are grouped by ``(family, noise-level)`` so all offsets x seeds for that
noise level run warm in one Python process.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

from experiments.config import CorrectnessSpec  # noqa: E402
from experiments.configs import get_config  # noqa: E402
from experiments.launch import (  # noqa: E402
    LaunchContext,
    LaunchPlan,
    add_launch_args,
    read_task_records,
    run_launch_plan,
)
from experiments.paths import results_root  # noqa: E402
from experiments.run import run_experiment  # noqa: E402
from experiments.seeds import replicate_seed_setup  # noqa: E402
from experiments.sweep_utils import run_sweep  # noqa: E402
from objective.base import sample_states  # noqa: E402
from objective.noise import (  # noqa: E402
    HeteroskedasticGaussianNoise,
    HomoskedasticGaussianNoise,
    NoisyObjective,
)
from objective.objectives import FixedRegressionObjective  # noqa: E402
from objective.policy import IdentityFeatureMap, SoftmaxPolicy  # noqa: E402


BASE_PRESET = "fixed_regression_base"
LAUNCH_PLAN_NAME = "fixed-regression-noise-offset-grid"
HOMO_PROJECT_NAME = "fixed-regression-homoskedastic-noise-offset-grid"
HETERO_PROJECT_NAME = "fixed-regression-heteroskedastic-noise-offset-grid"

REQUIRED_ESTIMATORS = ("finite_difference", "stein_difference")
RUN_SEEDS: tuple[int, ...] = (7, 8, 9)
ANCHOR_SEED = 7
VARY: tuple[str, ...] = ("optimizer", "noise")
FIXED_SEEDS: dict[str, int | None] = {}

GRID_THETA_OFFSETS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
HOMO_NOISE_STDS = (0.0, 0.1, 0.5, 2.0)
HETERO_NOISE_GROWTHS = (0.0, 0.25, 1.0, 4.0)

COMMON_OVERRIDES: dict[str, object] = {
    "enabled_estimators": REQUIRED_ESTIMATORS,
    "correctness": CorrectnessSpec(gradient_source="denoised_exact"),
    "perturbation_space": "u",
    "plot": False,
    "verbose": False,
    "wandb_enabled": False,
}

TRUTH_OVERRIDES: dict[str, object] = {
    "enabled_estimators": ("first_order",),
    "correctness": CorrectnessSpec(gradient_source="exact"),
    "plot": False,
    "verbose": False,
    "wandb_enabled": False,
}

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
class TruthReference:
    """Clean first-order reference for the anchored fixed-regression batch."""

    theta: np.ndarray
    final_u: float
    final_value: float
    base_objective: FixedRegressionObjective
    anchor_seed: int


@dataclass(frozen=True)
class GridFamily:
    """One noise family (adapter type) of the fixed-regression grid."""

    key: str
    project_name: str
    noise_levels: tuple[float, ...]
    noise_prefix: str
    axis_key: str
    noise_symbol: str
    legend_title: str
    noise_model_label: str
    noisy_objective: Callable[[float, TruthReference], NoisyObjective]


def _homoskedastic_objective(noise_std: float, reference: TruthReference) -> NoisyObjective:
    return NoisyObjective(
        base_objective=reference.base_objective,
        noise=HomoskedasticGaussianNoise(std=float(noise_std)),
    )


def _heteroskedastic_objective(growth: float, reference: TruthReference) -> NoisyObjective:
    return NoisyObjective(
        base_objective=reference.base_objective,
        noise=HeteroskedasticGaussianNoise(
            base_std=0.0,
            growth=float(growth),
            u_center=reference.final_u,
        ),
    )


HOMO_FAMILY = GridFamily(
    key="homoskedastic",
    project_name=HOMO_PROJECT_NAME,
    noise_levels=HOMO_NOISE_STDS,
    noise_prefix="noise-std",
    axis_key="noise_std",
    noise_symbol=r"\sigma",
    legend_title=r"constant noise std $\sigma$",
    noise_model_label=(
        r"fixed-regression homoskedastic noise $\hat{M}(x,u)=M(x,u)+\varepsilon(x,u)$, "
        r"$\varepsilon \sim \mathcal{N}(0, \sigma^2)$"
    ),
    noisy_objective=_homoskedastic_objective,
)
HETERO_FAMILY = GridFamily(
    key="heteroskedastic",
    project_name=HETERO_PROJECT_NAME,
    noise_levels=HETERO_NOISE_GROWTHS,
    noise_prefix="noise-growth",
    axis_key="noise_growth",
    noise_symbol=r"\gamma",
    legend_title=r"noise growth $\gamma$ in $\sigma(u)=\gamma\,|u-\bar{u}_{\mathrm{truth}}|$",
    noise_model_label=(
        r"fixed-regression heteroskedastic noise, std "
        r"$\sigma(u)=\gamma\,|u-\bar{u}_{\mathrm{truth}}|$ centered at the clean truth mean action"
    ),
    noisy_objective=_heteroskedastic_objective,
)
FAMILY_GROUPS: dict[str, tuple[GridFamily, ...]] = {
    "homoskedastic": (HOMO_FAMILY,),
    "heteroskedastic": (HETERO_FAMILY,),
    "all": (HOMO_FAMILY, HETERO_FAMILY),
}


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
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(RUN_SEEDS))
    parser.add_argument("--anchor-seed", type=int, default=ANCHOR_SEED)
    parser.add_argument(
        "--t-steps",
        type=int,
        default=None,
        help="Optional max-iteration override for both truth and grid runs.",
    )
    add_launch_args(parser, default_launch="auto", default_array=False)
    return parser.parse_args(argv)


def _base_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides = dict(COMMON_OVERRIDES)
    if args.t_steps is not None:
        overrides["t_steps"] = int(args.t_steps)
    return overrides


def _truth_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides = dict(TRUTH_OVERRIDES)
    if args.t_steps is not None:
        overrides["t_steps"] = int(args.t_steps)
    return overrides


def _truth_reference(args: argparse.Namespace) -> TruthReference:
    anchor_seed = int(args.anchor_seed)
    seed_setup = replicate_seed_setup(anchor_seed, anchor_seed, vary=(), fixed=FIXED_SEEDS)
    config = get_config(
        BASE_PRESET,
        overrides={**_truth_overrides(args), "seed_setup": seed_setup},
    )
    result = run_experiment(config)
    first_order = result.results["first_order"]
    objective = result.config.objective
    if not isinstance(objective, FixedRegressionObjective):
        raise ValueError(f"Expected FixedRegressionObjective, found {type(objective).__name__}.")
    return TruthReference(
        theta=np.asarray(first_order.theta, dtype=float),
        final_u=float(first_order.u),
        final_value=float(first_order.value),
        base_objective=objective,
        anchor_seed=anchor_seed,
    )


def _theta0(reference: TruthReference, offset: float) -> np.ndarray:
    return reference.theta + float(offset)


def _grid_run_name(family: GridFamily, noise_level: float, offset: float) -> str:
    level_part = _format_sweep_value(noise_level)
    offset_part = _format_sweep_value(offset)
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


def _family_level_override_list(
    family: GridFamily,
    noise_level: float,
    reference: TruthReference,
    base_overrides: Mapping[str, object],
) -> list[dict[str, object]]:
    return [
        {
            "_run_name": _grid_run_name(family, noise_level, offset),
            **base_overrides,
            "objective": family.noisy_objective(noise_level, reference),
            "theta0": _theta0(reference, offset),
        }
        for offset in GRID_THETA_OFFSETS
    ]


def _build_grid_override_list(
    family: GridFamily,
    reference: TruthReference,
    base_overrides: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    overrides = dict(COMMON_OVERRIDES if base_overrides is None else base_overrides)
    return [
        entry
        for noise_level in family.noise_levels
        for entry in _family_level_override_list(family, noise_level, reference, overrides)
    ]


GridGroup = tuple[GridFamily, float]


def _task_groups(families: Sequence[GridFamily]) -> list[GridGroup]:
    """One warm task per ``(family, noise-level)`` group."""
    return [(family, float(noise_level)) for family in families for noise_level in family.noise_levels]


def _run_grid_task(
    index: int,
    context: LaunchContext,
    *,
    args: argparse.Namespace,
    families: Sequence[GridFamily],
) -> dict[str, object]:
    del context
    family, noise_level = _task_groups(families)[index]
    reference = _truth_reference(args)
    run_seeds = tuple(int(seed) for seed in args.run_seeds)
    overrides = _family_level_override_list(family, noise_level, reference, _base_overrides(args))
    n_runs = _run_missing_variant_sweeps(
        project_name=family.project_name,
        override_list=overrides,
        run_seeds=run_seeds,
        anchor_seed=reference.anchor_seed,
    )
    return {
        "project": family.project_name,
        "family": family.key,
        "noise_level": float(noise_level),
        "n_runs": n_runs,
    }


def _run_grid_serial(context: LaunchContext, *, args: argparse.Namespace, families: Sequence[GridFamily]) -> None:
    del context
    reference = _truth_reference(args)
    run_seeds = tuple(int(seed) for seed in args.run_seeds)
    n_runs = 0
    for family in families:
        for noise_level in family.noise_levels:
            overrides = _family_level_override_list(family, noise_level, reference, _base_overrides(args))
            n_runs += _run_missing_variant_sweeps(
                project_name=family.project_name,
                override_list=overrides,
                run_seeds=run_seeds,
                anchor_seed=reference.anchor_seed,
            )
    regenerate_grid_plots(families, reference=reference, run_seeds=run_seeds)
    print(f"Completed {n_runs} fixed-regression grid runs.")


def _collect_grid_tasks(context: LaunchContext, *, args: argparse.Namespace, families: Sequence[GridFamily]) -> None:
    records = read_task_records(context)
    expected = len(_task_groups(families))
    if len(records) != expected:
        raise RuntimeError(f"Expected {expected} task records under {context.tasks_dir}, found {len(records)}.")
    reference = _truth_reference(args)
    regenerate_grid_plots(families, reference=reference, run_seeds=tuple(int(seed) for seed in args.run_seeds))
    print(f"Collected {len(records)} fixed-regression grid array tasks under {results_root()}.")


def _run_missing_variant_sweeps(
    *,
    project_name: str,
    override_list: Sequence[Mapping[str, object]],
    run_seeds: Sequence[int],
    anchor_seed: int,
) -> int:
    missing = _missing_overrides(
        project_name=project_name,
        override_list=override_list,
        run_seeds=run_seeds,
        required_estimators=REQUIRED_ESTIMATORS,
    )
    skipped = len(override_list) - len(missing)
    if not missing:
        print(f"No missing variants for '{project_name}' ({skipped} already complete).")
        return 0

    n_runs = 0
    for overrides in missing:
        sweep = run_sweep(
            base_preset=BASE_PRESET,
            run_seeds=tuple(int(seed) for seed in run_seeds),
            override_list=[dict(overrides)],
            vary=VARY,
            anchor_seed=int(anchor_seed),
            fixed=FIXED_SEEDS,
            project_name=project_name,
            display_keys=(),
        )
        n_runs += len(sweep.run_results)
    print(
        f"Completed {n_runs} missing runs for '{project_name}' "
        f"({len(missing)} variants x {len(run_seeds)} seeds; skipped {skipped})."
    )
    return n_runs


def _missing_overrides(
    *,
    project_name: str,
    override_list: Sequence[Mapping[str, object]],
    run_seeds: Sequence[int],
    required_estimators: Sequence[str],
) -> list[dict[str, object]]:
    missing: list[dict[str, object]] = []
    project_dir = _project_dir(project_name)
    for overrides in override_list:
        run_name = overrides.get("_run_name")
        if run_name is None:
            raise ValueError("Resume/skipping requires each override to include '_run_name'.")
        if not _variant_all_seeds_completed(
            project_dir / _path_part(run_name),
            run_seeds=run_seeds,
            required_estimators=required_estimators,
        ):
            missing.append(dict(overrides))
    return missing


def _variant_all_seeds_completed(
    variant_dir: Path,
    *,
    run_seeds: Sequence[int],
    required_estimators: Sequence[str],
) -> bool:
    if not variant_dir.is_dir():
        return False
    return all(
        _summary_has_estimators(variant_dir / f"summary-seed-{int(seed)}.json", required_estimators)
        for seed in run_seeds
    )


def _summary_has_estimators(summary_path: Path, estimators: Sequence[str]) -> bool:
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    estimator_payload = payload.get("estimators", {})
    return all(name in estimator_payload for name in estimators)


def regenerate_grid_plots(
    families: Sequence[GridFamily],
    *,
    reference: TruthReference,
    run_seeds: Sequence[int],
) -> None:
    """Rebuild grid CSVs/plots from saved fixed-regression summaries."""
    for family in families:
        rows = _collect_family_rows(family, reference, run_seeds=run_seeds)
        if not rows:
            print(f"Skipping grid plots for '{family.project_name}'; no summary rows found.")
            continue
        project_dir = _project_dir(family.project_name)
        project_dir.mkdir(parents=True, exist_ok=True)
        _write_grid_csv(project_dir / "fixed_regression_noise_offset_grid_finals.csv", family, rows)
        for estimator in REQUIRED_ESTIMATORS:
            estimator_rows = [row for row in rows if row["estimator"] == estimator]
            if not estimator_rows:
                continue
            plot_path = project_dir / f"fixed_regression_noise_offset_grid_{estimator}.png"
            _plot_family_estimator(plot_path, family, estimator, estimator_rows)
            print(f"Wrote grid plot {plot_path}")


def _collect_family_rows(
    family: GridFamily,
    reference: TruthReference,
    *,
    run_seeds: Sequence[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    project_dir = _project_dir(family.project_name)
    if not project_dir.is_dir():
        return rows
    for variant_dir in sorted(project_dir.iterdir()):
        if not variant_dir.is_dir():
            continue
        parsed = _parse_grid_variant(family, variant_dir.name)
        if parsed is None:
            continue
        noise_level, offset = parsed
        rows.extend(_variant_rows(variant_dir, family, noise_level, offset, reference, run_seeds=run_seeds))
    return rows


def _variant_rows(
    variant_dir: Path,
    family: GridFamily,
    noise_level: float,
    offset: float,
    reference: TruthReference,
    *,
    run_seeds: Sequence[int],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    initial_theta_distance = float(np.linalg.norm(_theta0(reference, offset) - reference.theta))
    for seed in run_seeds:
        summary_path = variant_dir / f"summary-seed-{int(seed)}.json"
        if not summary_path.exists():
            continue
        try:
            summary = _load_json(summary_path)
        except (OSError, json.JSONDecodeError):
            continue
        base_objective = _base_objective(summary)
        x_train = _train_x(summary)
        j_clean_truth = float(base_objective.value(reference.theta, x_train))
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
                    "initial_theta_distance_to_truth": initial_theta_distance,
                    "estimator": estimator,
                    "run_seed": int(seed),
                    "theta_distance_to_truth": float(np.linalg.norm(theta_hat - reference.theta)),
                    "clean_objective_gap": j_clean_hat - j_clean_truth,
                    "final_value": j_clean_hat,
                    "truth_final_value": j_clean_truth,
                    "final_u": float(estimator_payload.get("final_u", "nan")),
                    "truth_final_u": float(reference.final_u),
                    "optimizer_success": estimator_payload.get("optimizer_success", ""),
                    "summary_path": str(summary_path),
                }
            )
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _train_x(summary: Mapping[str, Any]) -> np.ndarray:
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


def _base_objective(summary: Mapping[str, Any]) -> FixedRegressionObjective:
    objective = summary["config"]["objective"]
    base = objective.get("base_objective", objective)
    if base.get("type") != "FixedRegressionObjective":
        raise ValueError(f"Expected FixedRegressionObjective, found {base.get('type')!r}.")
    policy_config = base["policy"]
    feature_map = policy_config.get("feature_map", {}) or {}
    if policy_config.get("type") != "SoftmaxPolicy" or feature_map.get("kind") != "identity":
        raise ValueError("This grid script expects identity-feature SoftmaxPolicy summaries.")
    policy = SoftmaxPolicy(
        feature_map=IdentityFeatureMap(),
        action_low=float(policy_config["action_low"]),
        action_high=float(policy_config["action_high"]),
    )
    return FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=np.asarray(base["beta_1"], dtype=float),
        beta_2=float(base["beta_2"]),
        beta_3=np.asarray(base["beta_3"], dtype=float),
        beta_4=float(base["beta_4"]),
    )


def _write_grid_csv(path: Path, family: GridFamily, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        family.axis_key,
        "theta_offset",
        "initial_theta_distance_to_truth",
        "estimator",
        "run_seed",
        "theta_distance_to_truth",
        "clean_objective_gap",
        "final_value",
        "truth_final_value",
        "final_u",
        "truth_final_u",
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


def _metric_stats(rows: list[dict[str, object]], metric: str, offset: float) -> tuple[float, float]:
    values = [float(row[metric]) for row in rows if float(row["theta_offset"]) == offset]
    return float(np.mean(values)), float(np.std(values, ddof=0))


def _plot_family_estimator(path: Path, family: GridFamily, estimator: str, rows: list[dict[str, object]]) -> None:
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
        _set_symlog_ticks(ax, all_offsets)
        metric_values = [float(row[metric]) for row in rows]
        if metric == "theta_distance_to_truth":
            positive = [value for value in metric_values if value > 0.0]
            if len(positive) == len(metric_values):
                ax.set_yscale("log")
            else:
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
        r"mean $\pm$ std over run seeds; curves = noise level, x = init offset",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _set_symlog_ticks(ax: object, values: list[float]) -> None:
    nonzero = [abs(value) for value in values if value != 0.0]
    if nonzero:
        ax.set_xscale("symlog", linthresh=min(nonzero))
    ax.set_xticks(values)
    ax.set_xticklabels([f"{value:g}" for value in values], rotation=45, ha="right")


def _format_sweep_value(value: object) -> str:
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        text = str(value)
        return text.replace(" ", "").replace("/", "-")


def _path_part(value: object) -> str:
    return str(value).replace(" ", "").replace("/", "-")


def _project_dir(project_name: str) -> Path:
    return results_root() / _path_part(project_name)


def _build_launch_plan(args: argparse.Namespace, families: Sequence[GridFamily]) -> LaunchPlan:
    return LaunchPlan(
        name=LAUNCH_PLAN_NAME,
        task_count=len(_task_groups(families)),
        requires_jax=False,
        run_task=partial(_run_grid_task, args=args, families=families),
        run_all=partial(_run_grid_serial, args=args, families=families),
        collect=partial(_collect_grid_tasks, args=args, families=families),
        default_launch="auto",
        default_array=False,
    )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    families = FAMILY_GROUPS[args.families]
    if args.plots_only:
        reference = _truth_reference(args)
        regenerate_grid_plots(families, reference=reference, run_seeds=tuple(int(seed) for seed in args.run_seeds))
        return
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(_build_launch_plan(args, families), args=args, argv=original_argv)


if __name__ == "__main__":
    main()
