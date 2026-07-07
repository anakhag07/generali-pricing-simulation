"""Run a planted-logistic noise x gradient-sample sweep as a Slurm array.

Finite difference is included as a deterministic zeroth-order baseline and only
varies the noise field. Stein difference additionally varies
``n_grad_samples``. The default seed policy fixes data, split, and theta while
varying the optimizer perturbation stream and the objective noise stream, so
replicates target estimator/noisy-oracle variance rather than sample or
initialization variance.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

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
    run_launch_plan,
    task_payloads,
)
from experiments.paths import results_root  # noqa: E402
from experiments.reporting.context import create_run_context  # noqa: E402
from experiments.reporting.json_summary import JsonReporter  # noqa: E402
from experiments.seeds import replicate_seed_setup  # noqa: E402
from objective.noise import (  # noqa: E402
    HeteroskedasticGaussianNoise,
    HomoskedasticGaussianNoise,
    NoisyObjective,
)


BASE_PRESET = "planted_logistic_base"
PROJECT_NAME = "planted-logistic-gradient-sample-noise-sweep"
FINITE_DIFFERENCE = "finite_difference"
STEIN_DIFFERENCE = "stein_difference"
RUN_SEEDS = (7, 8, 9)
ANCHOR_SEED = 7
VARY = ("optimizer", "noise")

HOMOSKEDASTIC_STDS = (0.0, 0.5, 2.0)
HETEROSKEDASTIC_GROWTHS = (0.0, 1.0, 4.0)
STEIN_N_GRAD_SAMPLES = (32, 64, 128)

NoiseFamilyName = Literal["homoskedastic", "heteroskedastic"]

FINAL_FIELDNAMES = (
    "project",
    "variant",
    "estimator",
    "run_seed",
    "noise_family",
    "noise_level",
    "noise_std",
    "noise_growth",
    "n_grad_samples",
    "final_value",
    "final_u",
    "runtime_sec",
    "optimizer_success",
    "optimizer_status",
    "summary_path",
    "run_dir",
)
SUMMARY_METRICS = ("final_value", "final_u", "runtime_sec")
SUMMARY_FIELDNAMES = (
    "project",
    "variant",
    "estimator",
    "noise_family",
    "noise_level",
    "n_grad_samples",
    "n_seeds",
    *(f"{metric}_{stat}" for metric in SUMMARY_METRICS for stat in ("mean", "std", "min", "max")),
)


@dataclass(frozen=True)
class NoiseFamily:
    name: NoiseFamilyName
    level_key: str
    levels: tuple[float, ...]


@dataclass(frozen=True)
class SweepVariant:
    name: str
    estimator: str
    noise_family: NoiseFamilyName
    noise_level: float
    n_grad_samples: int | None

    @property
    def noise_std(self) -> float:
        return float(self.noise_level) if self.noise_family == "homoskedastic" else 0.0

    @property
    def noise_growth(self) -> float:
        return float(self.noise_level) if self.noise_family == "heteroskedastic" else 0.0


NOISE_FAMILIES: dict[NoiseFamilyName, NoiseFamily] = {
    "homoskedastic": NoiseFamily(
        name="homoskedastic",
        level_key="std",
        levels=tuple(float(value) for value in HOMOSKEDASTIC_STDS),
    ),
    "heteroskedastic": NoiseFamily(
        name="heteroskedastic",
        level_key="growth",
        levels=tuple(float(value) for value in HETEROSKEDASTIC_GROWTHS),
    ),
}
FAMILY_GROUPS: dict[str, tuple[NoiseFamilyName, ...]] = {
    "all": ("homoskedastic", "heteroskedastic"),
    "homoskedastic": ("homoskedastic",),
    "heteroskedastic": ("heteroskedastic",),
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        choices=tuple(FAMILY_GROUPS),
        default="all",
        help="Noise family grid to run (default: all).",
    )
    parser.add_argument("--run-seeds", type=int, nargs="+", default=list(RUN_SEEDS))
    parser.add_argument("--anchor-seed", type=int, default=ANCHOR_SEED)
    parser.add_argument(
        "--vary",
        nargs="+",
        default=list(VARY),
        help="Seed streams that vary across run seeds (default: optimizer noise).",
    )
    parser.add_argument("--project-name", default=PROJECT_NAME)
    parser.add_argument(
        "--t-steps",
        type=int,
        default=None,
        help="Optional max optimizer iterations override; defaults to the base preset.",
    )
    add_launch_args(parser, default_launch="auto", default_array=True)
    return parser.parse_args(argv)


def _variants(families: Sequence[NoiseFamilyName]) -> list[SweepVariant]:
    variants: list[SweepVariant] = []
    for family_name in families:
        family = NOISE_FAMILIES[family_name]
        for level in family.levels:
            variants.append(
                SweepVariant(
                    name=_variant_name(FINITE_DIFFERENCE, family, level, None),
                    estimator=FINITE_DIFFERENCE,
                    noise_family=family.name,
                    noise_level=float(level),
                    n_grad_samples=None,
                )
            )
            for n_grad_samples in STEIN_N_GRAD_SAMPLES:
                variants.append(
                    SweepVariant(
                        name=_variant_name(STEIN_DIFFERENCE, family, level, int(n_grad_samples)),
                        estimator=STEIN_DIFFERENCE,
                        noise_family=family.name,
                        noise_level=float(level),
                        n_grad_samples=int(n_grad_samples),
                    )
                )
    return variants


def _task_specs(args: argparse.Namespace) -> list[tuple[SweepVariant, int]]:
    families = FAMILY_GROUPS[str(args.families)]
    seeds = tuple(int(seed) for seed in args.run_seeds)
    return [(variant, seed) for variant in _variants(families) for seed in seeds]


def _variant_name(
    estimator: str,
    family: NoiseFamily,
    level: float,
    n_grad_samples: int | None,
) -> str:
    name = f"{estimator}__{family.name}-{family.level_key}-{_value_label(level)}"
    if n_grad_samples is not None:
        name += f"__ngrad-{int(n_grad_samples)}"
    return name


def _config_for_variant(
    variant: SweepVariant,
    seed: int,
    args: argparse.Namespace,
):
    seed_setup = replicate_seed_setup(
        int(seed),
        int(args.anchor_seed),
        vary=tuple(str(item) for item in args.vary),
    )
    overrides: dict[str, Any] = {
        "objective": _noisy_objective(variant),
        "enabled_estimators": (variant.estimator,),
        "correctness": CorrectnessSpec(gradient_source="denoised_exact"),
        "plot": False,
        "verbose": False,
        "wandb_enabled": False,
        "seed_setup": seed_setup,
    }
    if variant.n_grad_samples is not None:
        overrides["n_grad_samples"] = int(variant.n_grad_samples)
    if args.t_steps is not None:
        overrides["t_steps"] = int(args.t_steps)
    return get_config(BASE_PRESET, overrides=overrides)


def _noisy_objective(variant: SweepVariant) -> NoisyObjective:
    base_objective = get_config(BASE_PRESET).objective
    if variant.noise_family == "homoskedastic":
        noise = HomoskedasticGaussianNoise(std=variant.noise_std)
    else:
        noise = HeteroskedasticGaussianNoise(
            base_std=0.0,
            growth=variant.noise_growth,
            u_center=_objective_u_star(base_objective),
        )
    return NoisyObjective(base_objective, noise)


def _objective_u_star(objective: object) -> float:
    optimal_u = getattr(objective, "optimal_u", None)
    if callable(optimal_u):
        return float(optimal_u())
    return float(getattr(objective, "u_star"))


def _run_sweep_task(index: int, context: LaunchContext, *, args: argparse.Namespace) -> dict[str, object]:
    variant, seed = _task_specs(args)[index]
    variant_dir = _variant_dir(args.project_name, variant.name, context.runs_root)
    seed_summary = variant_dir / f"summary-seed-{seed}.json"
    payload = _task_payload(args.project_name, variant, seed, variant_dir, seed_summary)
    if _summary_has_estimator(seed_summary, variant.estimator):
        print(f"Skipping completed task '{variant.name}' seed {seed}.")
        return payload

    config = _config_for_variant(variant, seed, args)
    run_context = create_run_context(
        variant.name,
        run_dir=variant_dir / "seeds" / f"seed-{seed}",
        run_metadata={
            "preset_name": BASE_PRESET,
            "variant_name": variant.name,
            "run_seed": int(seed),
            "noise_family": variant.noise_family,
            "noise_level": float(variant.noise_level),
            "n_grad_samples": variant.n_grad_samples,
            "vary_seed_streams": tuple(str(item) for item in args.vary),
        },
    )
    executed = execute_experiment_run(
        variant.name,
        config,
        run_context=run_context,
        reporter_stack_factory=_seed_reporter_stack_factory(variant_dir, seed),
    )
    return {**payload, "run_dir": str(executed.run_context.run_dir)}


def _task_payload(
    project_name: str,
    variant: SweepVariant,
    seed: int,
    variant_dir: Path,
    seed_summary: Path,
) -> dict[str, object]:
    return {
        "project": project_name,
        "variant": variant.name,
        "estimator": variant.estimator,
        "run_seed": int(seed),
        "noise_family": variant.noise_family,
        "noise_level": float(variant.noise_level),
        "noise_std": float(variant.noise_std),
        "noise_growth": float(variant.noise_growth),
        "n_grad_samples": "" if variant.n_grad_samples is None else int(variant.n_grad_samples),
        "summary_json": str(seed_summary),
        "run_dir": str(variant_dir / "seeds" / f"seed-{seed}"),
    }


def _run_sweep_serial(context: LaunchContext, *, args: argparse.Namespace) -> None:
    payloads = [_run_sweep_task(index, context, args=args) for index in range(len(_task_specs(args)))]
    _write_outputs_from_payloads(payloads, context.runs_root, args.project_name)
    print(f"Completed {len(payloads)} planted-logistic gradient-sample/noise tasks.")


def _collect_sweep_tasks(context: LaunchContext, *, args: argparse.Namespace) -> None:
    payloads = task_payloads(context)
    _write_outputs_from_payloads(payloads, context.runs_root, args.project_name)
    print(f"Collected {len(payloads)} planted-logistic gradient-sample/noise array tasks.")


def _write_outputs_from_payloads(
    payloads: Sequence[Mapping[str, object]],
    runs_root: Path,
    project_name: str,
) -> None:
    rows = _final_rows_from_payloads(payloads)
    if not rows:
        raise ValueError("No final rows were produced.")
    project_dir = _project_dir(project_name, runs_root)
    _write_rows(project_dir / "gradient_sample_noise_sweep_finals.csv", rows, FINAL_FIELDNAMES)
    _write_rows(
        project_dir / "gradient_sample_noise_sweep_summary.csv",
        _aggregate_rows(rows),
        SUMMARY_FIELDNAMES,
    )


def _final_rows_from_payloads(payloads: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for payload in payloads:
        summary_path = Path(str(payload["summary_json"]))
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        estimator = str(payload["estimator"])
        estimator_payload = summary.get("estimators", {}).get(estimator)
        if estimator_payload is None:
            continue
        rows.append(
            {
                "project": str(payload["project"]),
                "variant": str(payload["variant"]),
                "estimator": estimator,
                "run_seed": int(payload["run_seed"]),
                "noise_family": str(payload["noise_family"]),
                "noise_level": float(payload["noise_level"]),
                "noise_std": float(payload["noise_std"]),
                "noise_growth": float(payload["noise_growth"]),
                "n_grad_samples": payload.get("n_grad_samples", ""),
                "final_value": float(estimator_payload["final_value"]),
                "final_u": float(estimator_payload["final_u"]),
                "runtime_sec": float(estimator_payload["runtime_sec"]),
                "optimizer_success": estimator_payload.get("optimizer_success", ""),
                "optimizer_status": estimator_payload.get("optimizer_status", ""),
                "summary_path": str(summary_path),
                "run_dir": str(payload["run_dir"]),
            }
        )
    return rows


def _aggregate_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    keys = sorted({(str(row["variant"]), str(row["estimator"])) for row in rows})
    summary_rows: list[dict[str, object]] = []
    for variant, estimator in keys:
        group = [
            row for row in rows
            if str(row["variant"]) == variant and str(row["estimator"]) == estimator
        ]
        first = group[0]
        summary: dict[str, object] = {
            "project": first["project"],
            "variant": variant,
            "estimator": estimator,
            "noise_family": first["noise_family"],
            "noise_level": first["noise_level"],
            "n_grad_samples": first.get("n_grad_samples", ""),
            "n_seeds": len(group),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray([float(row[metric]) for row in group if _has_float(row.get(metric))], dtype=float)
            if values.size == 0:
                for stat in ("mean", "std", "min", "max"):
                    summary[f"{metric}_{stat}"] = ""
                continue
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=0))
            summary[f"{metric}_min"] = float(np.min(values))
            summary[f"{metric}_max"] = float(np.max(values))
        summary_rows.append(summary)
    return summary_rows


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


def _summary_has_estimator(path: Path, estimator: str) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return estimator in payload.get("estimators", {})


def _write_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (
            str(row.get("estimator", "")),
            str(row.get("noise_family", "")),
            float(row.get("noise_level", 0.0)),
            _optional_int_sort_value(row.get("n_grad_samples")),
            int(row.get("run_seed", 0) or 0),
        ),
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _build_launch_plan(args: argparse.Namespace) -> LaunchPlan:
    return LaunchPlan(
        name=args.project_name,
        task_count=len(_task_specs(args)),
        requires_jax=False,
        run_task=partial(_run_sweep_task, args=args),
        run_all=partial(_run_sweep_serial, args=args),
        collect=partial(_collect_sweep_tasks, args=args),
        default_launch="auto",
        default_array=True,
    )


def _project_dir(project_name: str, runs_root: Path | None = None) -> Path:
    root = results_root() if runs_root is None else Path(runs_root)
    return root / _path_part(project_name)


def _variant_dir(project_name: str, variant_name: str, runs_root: Path | None = None) -> Path:
    return _project_dir(project_name, runs_root) / _path_part(variant_name)


def _path_part(value: object) -> str:
    text = str(value).strip().replace(" ", "-").replace("/", "-")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text) or "run"


def _value_label(value: float) -> str:
    return f"{float(value):g}"


def _has_float(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _optional_int_sort_value(value: object) -> int:
    if value in (None, ""):
        return -1
    return int(value)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    original_argv = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    run_launch_plan(_build_launch_plan(args), args=args, argv=original_argv)


if __name__ == "__main__":
    main()
