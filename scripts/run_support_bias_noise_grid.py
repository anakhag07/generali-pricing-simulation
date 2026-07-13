"""Run the combined noise-level x support-bias planted-logistic grid.

Layers homoskedastic / heteroskedastic Gaussian value noise on top of the
planted-logistic upper-support bias surrogate. The per-variant objective is the
nested wrapper

``NoisyObjective(BiasedObjective(M_star, UpperSupportHingeBias), noise)``

so noise is applied on top of the support-bias surrogate

``M_hat(x, u) = M_star(x, u) - lambda_bias * max(0, u - (u_star + r)) + delta(x, u)``.

For each noise family (homoskedastic std ``sigma``, heteroskedastic growth
``gamma``) the grid varies the bias strength ``lambda_bias`` at a fixed support
radius ``r`` and reports, against the clean unbiased **first-order truth**, the
true objective gap and the off-support action excess. Because the noisy
objective has no analytical gradient (``NoisyObjective.grad`` raises), the grid
runs zeroth-order estimators only (``finite_difference`` / ``stein_difference``).

Everything called "truth" is the clean, unbiased, noise-free planted optimum:
the script solves an anchored clean first-order run once (persisted under
``results/planted_logistic_base/``) and reuses its theta as ``theta^FO_clean``
for the gap/distance reference. The metric reconstruction rebuilds the clean
``PlantedLogisticObjective`` from each summary config (the summary stores the
noisy final objective), so gaps and support excess are computed on the clean
objective, never the surrogate.

Outputs per family project (``homoskedastic-support-bias-noise-grid`` /
``heteroskedastic-support-bias-noise-grid``):
``support_bias_noise_grid_finals.csv`` and, per estimator, a two-panel figure
(true clean-objective gap | mean support excess, curves = noise level, x = bias
strength). ``--plots-only`` regenerates outputs from saved summaries without
running. ``--families`` selects the grid group. Launch-aware with auto Slurm
submit; ``--launch slurm --array`` decomposes into one warm task per
(family, noise-level).

Speed note: each planted-logistic run is only ~10s of work but pays a large
fixed optax import cost per process, so the array decomposition runs one task
per (family, noise-level) -- each task loops its lambda_bias x seed variants in
a single warm process. Never run the grid on the shared login node; use an
``salloc`` compute allocation for ``--launch local``.
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
from experiments.execution import default_reporter_stack, execute_experiment_run  # noqa: E402
from experiments.launch import (  # noqa: E402
    LaunchContext,
    LaunchPlan,
    add_launch_args,
    read_task_records,
    run_launch_plan,
)
from experiments.paths import results_root  # noqa: E402
from experiments.policy_validation import policy_u_values  # noqa: E402
from experiments.reporting.context import create_run_context  # noqa: E402
from experiments.reporting.json_summary import JsonReporter  # noqa: E402
from experiments.seeds import replicate_seed_setup  # noqa: E402
from experiments.sweep_utils import expand_sweep_overrides, run_sweep  # noqa: E402
from objective.base import sample_states  # noqa: E402
from objective.noise import (  # noqa: E402
    HeteroskedasticGaussianNoise,
    HomoskedasticGaussianNoise,
    NoisyObjective,
)
from objective.objectives import (  # noqa: E402
    BiasedObjective,
    PlantedLogisticObjective,
    UpperSupportHingeBias,
)
from objective.policy import IdentityFeatureMap, SoftmaxPolicy  # noqa: E402


# =============================================================================
# Grid settings
# Runtime overrides mirror the noise-offset grid so the two experiments stay
# comparable; the theta-offset axis is replaced by the bias-strength axis and
# theta0 stays at the base preset default (cold start, identical for all runs).
# =============================================================================

BASE_PRESET = "planted_logistic_base"
LAUNCH_PLAN_NAME = "planted-support-bias-noise-grid"
HOMO_PROJECT_NAME = "homoskedastic-support-bias-noise-grid"
HETERO_PROJECT_NAME = "heteroskedastic-support-bias-noise-grid"

RUN_SEEDS: tuple[int, ...] = (7, 8, 9)
# Data/split/theta stay anchored to ANCHOR_SEED so variants remain comparable;
# the estimator perturbation streams and the frozen noise field are redrawn per
# seed. theta0 is a fixed cold start, so error bars capture estimator + noise
# stochasticity only.
ANCHOR_SEED = 7
VARY: tuple[str, ...] = ("optimizer", "noise")
FIXED_SEEDS: dict[str, int | None] = {}

# x-axis: off-support bias strength. Fixed support radius keeps each family
# figure 2D (curves = noise level, x = lambda_bias).
LAMBDA_BIAS_VALUES = (0.0, 0.01, 0.05, 0.1, 0.2)
SUPPORT_RADIUS = 0.05
# curves: noise level per family.
HOMO_NEW_NOISE_STDS = (0.0, 0.1, 0.5, 2.0)
HETERO_NEW_NOISE_GROWTHS = (0.0, 0.25, 1.0, 4.0)

REQUIRED_ESTIMATORS = ("finite_difference", "stein_difference")

# Clean unbiased first-order truth (theta^FO_clean) reference run, persisted so
# --plots-only and the Slurm collector can reuse it without retraining.
TRUTH_RUN_NAME = "support_bias_noise_clean_truth"
N_SAMPLES = 1000

COMMON_OVERRIDES: dict[str, object] = {
    "enabled_estimators": REQUIRED_ESTIMATORS,
    # denoised_exact now unwraps the full NoisyObjective(BiasedObjective(...))
    # stack to the clean planted objective, so the recorded "true" gradient is
    # the first-order truth, not the biased surrogate.
    "correctness": CorrectnessSpec(gradient_source="denoised_exact"),
    "perturbation_space": "u",
    "step_rule": "optax-adam",
    "t_steps": 2000,
    "step_size": 0.05,
    "n_samples": N_SAMPLES,
    "sigma": 0.05,
    "n_grad_samples": 8,
    "plot": False,
    "verbose": False,
    "wandb_enabled": False,
}

_PLANTED_BASE = get_config(BASE_PRESET)
# Center heteroskedastic noise at the planted optimum so the noise floor sits on
# the global-minimum region (noiseless near u_star).
U_STAR = float(_PLANTED_BASE.objective.optimal_u())


# =============================================================================
# Objective composition: noise layered on top of the support-bias surrogate.
# =============================================================================


def _support_bias(lambda_bias: float) -> UpperSupportHingeBias:
    return UpperSupportHingeBias(
        lambda_bias=float(lambda_bias),
        support_center=U_STAR,
        support_radius=SUPPORT_RADIUS,
    )


def _biased_objective(lambda_bias: float) -> BiasedObjective:
    return BiasedObjective(
        base_objective=_PLANTED_BASE.objective,
        bias=_support_bias(lambda_bias),
    )


def _homo_noisy_objective(lambda_bias: float, noise_std: float) -> NoisyObjective:
    return NoisyObjective(
        base_objective=_biased_objective(lambda_bias),
        noise=HomoskedasticGaussianNoise(std=float(noise_std)),
    )


def _hetero_noisy_objective(lambda_bias: float, growth: float) -> NoisyObjective:
    return NoisyObjective(
        base_objective=_biased_objective(lambda_bias),
        noise=HeteroskedasticGaussianNoise(
            base_std=0.0,
            growth=float(growth),
            u_center=U_STAR,
        ),
    )


# =============================================================================
# Grid family definitions
# =============================================================================

ESTIMATOR_LABELS = {
    "finite_difference": "Finite difference",
    "stein_difference": "Stein difference",
}

X_AXIS_LABEL = (
    r"Bias strength $\lambda_{\mathrm{bias}}$ in "
    r"$b(u) = -\lambda_{\mathrm{bias}}\,(u - h)_+$,  $h = u^\ast + r$"
    "\n"
    rf"(fixed support radius $r = {SUPPORT_RADIUS:g}$; noise layered on top of the bias surrogate)"
)
TRUE_GAP_LABEL = (
    r"True gap $J_{\mathrm{clean}}(\hat{\theta}) - "
    r"J_{\mathrm{clean}}(\theta^{\mathrm{FO}}_{\mathrm{clean}})$ (train batch)"
)
SUPPORT_EXCESS_LABEL = r"Mean support excess $n^{-1}\sum_i(\pi_{\hat{\theta}}(x_i) - h)_+$"


@dataclass(frozen=True)
class GridFamily:
    """One noise family (adapter type) of the combined support-bias grid."""

    key: str
    project_name: str
    new_noise_levels: tuple[float, ...]
    noise_prefix: str
    axis_key: str
    noise_symbol: str
    legend_title: str
    noise_model_label: str
    noisy_objective: Callable[[float, float], object]


HOMO_FAMILY = GridFamily(
    key="homoskedastic",
    project_name=HOMO_PROJECT_NAME,
    new_noise_levels=HOMO_NEW_NOISE_STDS,
    noise_prefix="noise-std",
    axis_key="noise_std",
    noise_symbol=r"\sigma",
    legend_title=r"constant noise std $\sigma$",
    noise_model_label=(
        r"homoskedastic noise $\hat{M}(x,u) = M_{\mathrm{bias}}(x,u) + \varepsilon(x,u)$, "
        r"$\varepsilon \sim \mathcal{N}(0, \sigma^2)$"
    ),
    noisy_objective=_homo_noisy_objective,
)
HETERO_FAMILY = GridFamily(
    key="heteroskedastic",
    project_name=HETERO_PROJECT_NAME,
    new_noise_levels=HETERO_NEW_NOISE_GROWTHS,
    noise_prefix="noise-growth",
    axis_key="noise_growth",
    noise_symbol=r"\gamma",
    legend_title=r"noise growth $\gamma$ in $\sigma(u) = \gamma\,|u - u^\ast|$",
    noise_model_label=(
        r"heteroskedastic noise, std $\sigma(u) = \gamma\,|u - u^\ast|$ "
        r"(noiseless at the planted optimum $u^\ast$)"
    ),
    noisy_objective=_hetero_noisy_objective,
)
FAMILY_GROUPS: dict[str, tuple[GridFamily, ...]] = {
    "homoskedastic": (HOMO_FAMILY,),
    "heteroskedastic": (HETERO_FAMILY,),
    "all": (HOMO_FAMILY, HETERO_FAMILY),
}


def _grid_run_name(family: GridFamily, noise_level: float, lambda_bias: float) -> str:
    level_part = _format_sweep_value(noise_level)
    lambda_part = _format_sweep_value(lambda_bias)
    return f"{family.noise_prefix}-{level_part}__lambda-{lambda_part}"


def _parse_grid_variant(family: GridFamily, variant_name: str) -> tuple[float, float] | None:
    prefix = f"{family.noise_prefix}-"
    separator = "__lambda-"
    if not variant_name.startswith(prefix) or separator not in variant_name:
        return None
    level_part, lambda_part = variant_name.removeprefix(prefix).split(separator, 1)
    try:
        return float(level_part), float(lambda_part)
    except ValueError:
        return None


def _family_level_override_list(family: GridFamily, noise_level: float) -> list[dict[str, object]]:
    return [
        {
            "_run_name": _grid_run_name(family, noise_level, lambda_bias),
            **COMMON_OVERRIDES,
            "objective": family.noisy_objective(lambda_bias, noise_level),
        }
        for lambda_bias in LAMBDA_BIAS_VALUES
    ]


def _build_grid_override_list(family: GridFamily) -> list[dict[str, object]]:
    return [
        override
        for noise_level in family.new_noise_levels
        for override in _family_level_override_list(family, noise_level)
    ]


# =============================================================================
# Sweep bookkeeping helpers (results paths, completed-run detection, per-seed
# reporter stacks). Mirrors the noise-offset grid.
# =============================================================================


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


def _variant_dir(project_name: str, variant_name: str) -> Path:
    return _project_dir(project_name) / _path_part(variant_name)


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


def _variant_is_completed(variant_dir: Path, required_estimators: Sequence[str]) -> bool:
    if not variant_dir.is_dir():
        return False
    for summary_path in _summary_paths(variant_dir):
        if _summary_has_estimators(summary_path, required_estimators):
            return True
    return False


def _missing_overrides(
    *,
    project_name: str,
    override_list: Sequence[Mapping[str, object]],
    required_estimators: Sequence[str],
) -> list[dict[str, object]]:
    missing: list[dict[str, object]] = []
    project_dir = _project_dir(project_name)
    for overrides in override_list:
        run_name = overrides.get("_run_name")
        if run_name is None:
            raise ValueError("Resume/skipping requires each override to include '_run_name'.")
        if not _variant_is_completed(project_dir / _path_part(run_name), required_estimators):
            missing.append(dict(overrides))
    return missing


def _seed_reporter_stack_factory(variant_dir: Path, seed: int):
    def factory(config):
        return default_reporter_stack(
            config,
            json_reporter=JsonReporter(
                summary_name=f"summary-seed-{seed}.json",
                summary_dir=variant_dir,
            ),
            include_plots=False,
        )

    return factory


def _run_missing_sweep(
    *,
    project_name: str,
    override_list: Sequence[Mapping[str, object]],
) -> int:
    missing = _missing_overrides(
        project_name=project_name,
        override_list=override_list,
        required_estimators=REQUIRED_ESTIMATORS,
    )
    skipped = len(override_list) - len(missing)
    if not missing:
        print(f"No missing variants for '{project_name}' ({skipped} already complete).")
        return 0

    sweep = run_sweep(
        base_preset=BASE_PRESET,
        run_seeds=RUN_SEEDS,
        override_list=missing,
        vary=VARY,
        anchor_seed=ANCHOR_SEED,
        fixed=FIXED_SEEDS,
        project_name=project_name,
        display_keys=(),
    )
    print(
        f"Completed {len(sweep.run_results)} missing runs for '{project_name}' "
        f"({len(missing)} variants x {len(RUN_SEEDS)} seeds; skipped {skipped})."
    )
    return len(sweep.run_results)


# =============================================================================
# Clean first-order truth reference (theta^FO_clean)
# =============================================================================


def _truth_run_dir() -> Path:
    return results_root() / "planted_logistic_base" / TRUTH_RUN_NAME


def _truth_summary_path() -> Path:
    return _truth_run_dir() / "summary.json"


def _truth_overrides() -> dict[str, object]:
    return {
        # Anchor the truth's train batch to ANCHOR_SEED (all streams) so it
        # matches the variants' fixed data/split, whose data stream is also
        # pinned to ANCHOR_SEED.
        "seed": ANCHOR_SEED,
        "n_samples": N_SAMPLES,
        "step_rule": "l-bfgs-b",
        "t_steps": 1000,
        "enabled_estimators": ("first_order",),
        "correctness": CorrectnessSpec(gradient_source="exact"),
        "perturbation_space": "u",
        "plot": False,
        "verbose": False,
        "wandb_enabled": False,
    }


def _ensure_truth_theta() -> np.ndarray | None:
    """Return theta^FO_clean, solving and persisting the clean run if missing."""
    summary_path = _truth_summary_path()
    if _summary_has_estimators(summary_path, ("first_order",)):
        return _theta_from_summary(summary_path, "first_order")
    config = get_config(BASE_PRESET, overrides=_truth_overrides())
    run_context = create_run_context(TRUTH_RUN_NAME, run_dir=_truth_run_dir())
    executed = execute_experiment_run(
        TRUTH_RUN_NAME,
        config,
        run_context=run_context,
        reporter_stack_factory=lambda cfg: default_reporter_stack(cfg, include_plots=False),
        run_metadata={"preset_name": BASE_PRESET, "variant_name": TRUTH_RUN_NAME},
    )
    print(f"Solved clean first-order truth at {run_context.run_dir}.")
    return np.asarray(executed.result.results["first_order"].theta, dtype=float)


# =============================================================================
# Launch wiring: one warm array task per (family, noise-level).
# =============================================================================

GridGroup = tuple[GridFamily, float, list[tuple[str, dict[str, Any]]]]


def _task_groups(families: Sequence[GridFamily]) -> list[GridGroup]:
    """One group per (family, noise-level); each holds its lambda_bias variants."""
    groups: list[GridGroup] = []
    for family in families:
        for noise_level in family.new_noise_levels:
            variants = expand_sweep_overrides(
                base_preset=BASE_PRESET,
                override_list=_family_level_override_list(family, noise_level),
                display_keys=(),
            )
            groups.append((family, float(noise_level), [(name, dict(ov)) for name, ov in variants]))
    return groups


def _run_grid_task(
    index: int, context: LaunchContext, *, families: Sequence[GridFamily]
) -> dict[str, object]:
    del context
    family, noise_level, variants = _task_groups(families)[index]
    project_name = family.project_name
    subruns: list[dict[str, object]] = []
    for variant_name, overrides in variants:
        variant_dir = _variant_dir(project_name, variant_name)
        for seed in RUN_SEEDS:
            seed_summary = variant_dir / f"summary-seed-{seed}.json"
            run_dir = variant_dir / "seeds" / f"seed-{seed}"
            if _summary_has_estimators(seed_summary, REQUIRED_ESTIMATORS):
                print(f"Skipping completed '{variant_name}' seed {seed} in '{project_name}'.")
                subruns.append({"variant": variant_name, "run_seed": seed, "run_dir": str(run_dir)})
                continue
            seed_setup = replicate_seed_setup(seed, ANCHOR_SEED, vary=VARY, fixed=FIXED_SEEDS)
            config = get_config(BASE_PRESET, overrides={**overrides, "seed_setup": seed_setup})
            run_context = create_run_context(variant_name, run_dir=run_dir)
            executed = execute_experiment_run(
                variant_name,
                config,
                run_context=run_context,
                reporter_stack_factory=_seed_reporter_stack_factory(variant_dir, seed),
            )
            subruns.append(
                {"variant": variant_name, "run_seed": seed, "run_dir": str(executed.run_context.run_dir)}
            )
    return {
        "project": project_name,
        "noise_level": noise_level,
        "n_subruns": len(subruns),
        "subruns": subruns,
    }


def _run_grid_serial(context: LaunchContext, *, families: Sequence[GridFamily]) -> None:
    del context
    n_runs = 0
    for family in families:
        n_runs += _run_missing_sweep(
            project_name=family.project_name,
            override_list=_build_grid_override_list(family),
        )
    regenerate_grid_plots(families)
    print(f"Completed {n_runs} total missing grid runs for preset '{BASE_PRESET}'.")


def _collect_grid_tasks(context: LaunchContext, *, families: Sequence[GridFamily]) -> None:
    records = read_task_records(context)
    expected = len(_task_groups(families))
    if len(records) != expected:
        raise RuntimeError(
            f"Expected {expected} task records under {context.tasks_dir}, found {len(records)}."
        )
    regenerate_grid_plots(families)
    print(f"Collected {len(records)} grid array tasks under {results_root()}.")


def _build_launch_plan(families: Sequence[GridFamily]) -> LaunchPlan:
    return LaunchPlan(
        name=LAUNCH_PLAN_NAME,
        task_count=len(_task_groups(families)),
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
# gap and support excess are recomputed by rebuilding the clean planted
# objective from the summary config and resampling the train split.
# =============================================================================


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _theta_from_summary(summary_path: Path, estimator: str) -> np.ndarray:
    summary = _load_json(summary_path)
    return np.asarray(summary["estimators"][estimator]["theta"], dtype=float)


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


def _build_planted(planted_config: dict[str, Any]) -> PlantedLogisticObjective:
    policy_config = planted_config["policy"]
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
        alpha=float(planted_config["alpha"]),
        beta=np.asarray(planted_config["beta"], dtype=float),
        bias=float(planted_config["bias"]),
        u_star=float(planted_config["u_star"]),
    )


def _planted_and_bias(summary: dict[str, Any]) -> tuple[PlantedLogisticObjective, dict[str, Any]]:
    """Descend NoisyObjective -> BiasedObjective -> PlantedLogisticObjective."""
    node = summary["config"]["objective"]
    bias_dict: dict[str, Any] | None = None
    while node.get("type") != "PlantedLogisticObjective":
        if node.get("type") == "BiasedObjective":
            bias_dict = node.get("bias")
        base = node.get("base_objective")
        if base is None:
            raise ValueError(f"Could not find PlantedLogisticObjective in {node.get('type')!r} chain.")
        node = base
    if bias_dict is None:
        raise ValueError("Expected a BiasedObjective wrapper in the objective chain.")
    return _build_planted(node), bias_dict


def _variant_rows(
    variant_dir: Path,
    family: GridFamily,
    noise_level: float,
    lambda_bias: float,
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
        planted, bias_dict = _planted_and_bias(summary)
        support_upper = float(bias_dict["support_upper"])
        x_train = _train_x(summary)
        j_clean_truth = float(planted.value(truth_theta, x_train))
        for estimator in REQUIRED_ESTIMATORS:
            estimator_payload = summary.get("estimators", {}).get(estimator)
            if estimator_payload is None or "theta" not in estimator_payload:
                continue
            theta_hat = np.asarray(estimator_payload["theta"], dtype=float)
            u_values = np.asarray(policy_u_values(planted, theta_hat, x_train), dtype=float)
            excess = np.maximum(0.0, u_values - support_upper)
            rows.append(
                {
                    family.axis_key: noise_level,
                    "lambda_bias": lambda_bias,
                    "support_radius": SUPPORT_RADIUS,
                    "support_upper": support_upper,
                    "estimator": estimator,
                    "run_seed": seed,
                    "theta_distance_to_truth": float(np.linalg.norm(theta_hat - truth_theta)),
                    "clean_objective_gap": float(planted.value(theta_hat, x_train)) - j_clean_truth,
                    "mean_action": float(np.mean(u_values)),
                    "mean_support_excess": float(np.mean(excess)),
                    "support_violation_rate": float(np.mean(u_values > support_upper)),
                    "optimizer_success": estimator_payload.get("optimizer_success", ""),
                    "summary_path": str(summary_path),
                }
            )
    return rows


def _collect_family_rows(family: GridFamily, truth_theta: np.ndarray) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    project_dir = _project_dir(family.project_name)
    if project_dir.is_dir():
        for variant_dir in sorted(project_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            parsed = _parse_grid_variant(family, variant_dir.name)
            if parsed is None:
                continue
            noise_level, lambda_bias = parsed
            rows.extend(_variant_rows(variant_dir, family, noise_level, lambda_bias, truth_theta))
    return rows


# =============================================================================
# Outputs: per-family CSV plus per-estimator two-panel figures
# =============================================================================


def regenerate_grid_plots(families: Sequence[GridFamily]) -> None:
    """Rebuild grid CSVs/plots from saved summaries against theta^FO_clean."""
    truth_theta = _ensure_truth_theta()
    if truth_theta is None:
        print("Skipping grid plots; clean first-order truth is unavailable.")
        return
    for family in families:
        rows = _collect_family_rows(family, truth_theta)
        if not rows:
            print(f"Skipping grid plots for '{family.project_name}'; no summary rows found.")
            continue
        project_dir = _project_dir(family.project_name)
        project_dir.mkdir(parents=True, exist_ok=True)
        _write_grid_csv(project_dir / "support_bias_noise_grid_finals.csv", family, rows)
        for estimator in REQUIRED_ESTIMATORS:
            estimator_rows = [row for row in rows if row["estimator"] == estimator]
            if not estimator_rows:
                continue
            plot_path = project_dir / f"support_bias_noise_grid_{estimator}.png"
            _plot_family_estimator(plot_path, family, estimator, estimator_rows)
            print(f"Wrote grid plot {plot_path}")


def _write_grid_csv(path: Path, family: GridFamily, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        family.axis_key,
        "lambda_bias",
        "support_radius",
        "support_upper",
        "estimator",
        "run_seed",
        "theta_distance_to_truth",
        "clean_objective_gap",
        "mean_action",
        "mean_support_excess",
        "support_violation_rate",
        "optimizer_success",
        "summary_path",
    ]
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row[family.axis_key]),
            float(row["lambda_bias"]),
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
    rows: list[dict[str, object]], metric: str, lambda_bias: float
) -> tuple[float, float]:
    values = [float(row[metric]) for row in rows if float(row["lambda_bias"]) == lambda_bias]
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
        ("clean_objective_gap", TRUE_GAP_LABEL),
        ("mean_support_excess", SUPPORT_EXCESS_LABEL),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))
    noise_levels = sorted({float(row[family.axis_key]) for row in rows})
    all_lambdas = sorted({float(row["lambda_bias"]) for row in rows})
    cmap = colormaps["viridis"]
    for level_index, noise_level in enumerate(noise_levels):
        level_rows = [row for row in rows if float(row[family.axis_key]) == noise_level]
        lambdas = sorted({float(row["lambda_bias"]) for row in level_rows})
        color = cmap(0.85 * level_index / max(len(noise_levels) - 1, 1))
        label = rf"${family.noise_symbol} = {noise_level:g}$"
        if noise_level == 0.0:
            label += " (clean)"
        for ax, (metric, _) in zip(axes, metrics):
            means_stds = [_metric_stats(level_rows, metric, lam) for lam in lambdas]
            means = [mean for mean, _ in means_stds]
            stds = [std for _, std in means_stds]
            if metric == "mean_support_excess":
                # Excess is nonnegative: never draw error bars below zero.
                yerr = np.vstack([np.minimum(stds, means), stds])
            else:
                yerr = np.vstack([stds, stds])
            ax.errorbar(
                lambdas,
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
        ax.set_xticks(all_lambdas)
        ax.set_xticklabels([f"{value:g}" for value in all_lambdas], rotation=45, ha="right")
        if metric == "clean_objective_gap":
            ax.set_yscale("symlog", linthresh=1e-8)
        else:
            ax.set_ylim(bottom=0.0)
        ax.set_xlabel(X_AXIS_LABEL)
        ax.set_ylabel(y_label)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(title=family.legend_title)
    fig.suptitle(
        f"{ESTIMATOR_LABELS.get(estimator, estimator)} — {family.noise_model_label}\n"
        r"mean $\pm$ std over run seeds "
        f"{RUN_SEEDS}; curves = noise level, x = bias strength",
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
