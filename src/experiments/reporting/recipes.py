"""Validated post-run report recipes for the generic experiment manifest."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ManifestReportSpec:
    """One named, reproducible post-run report attached to a manifest."""

    name: str
    recipe: str
    options: dict[str, Any]


ReportRunner = Callable[[Path, Mapping[str, Any]], list[Path]]


def parse_manifest_reports(value: Any) -> tuple[ManifestReportSpec, ...]:
    """Validate the optional ``reporting`` array from an experiment manifest."""
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("Manifest reporting must be a JSON array.")
    reports: list[ManifestReportSpec] = []
    seen: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ValueError("Each reporting[] item must be a JSON object.")
        recipe = str(item.get("recipe") or "").strip()
        if recipe not in _REPORT_RUNNERS:
            raise ValueError(
                f"Unknown reporting recipe {recipe!r}; choose from {sorted(_REPORT_RUNNERS)}."
            )
        name = _path_part(item.get("name") or recipe)
        if name in seen:
            raise ValueError(f"Duplicate reporting name {name!r}.")
        options = item.get("options") or {}
        if not isinstance(options, Mapping):
            raise ValueError("Manifest reporting[].options must be a JSON object.")
        seen.add(name)
        reports.append(ManifestReportSpec(name=name, recipe=recipe, options=dict(options)))
    return tuple(reports)


def run_manifest_reports(
    reports: Sequence[ManifestReportSpec],
    *,
    project_dir: str | Path,
) -> list[dict[str, Any]]:
    """Run missing report recipes under the manifest project directory."""
    root = Path(project_dir) / "reports"
    results: list[dict[str, Any]] = []
    for report in reports:
        output_dir = root / report.name
        provenance_path = output_dir / "provenance.json"
        if provenance_path.exists() and not bool(report.options.get("force", False)):
            results.append(
                {
                    "name": report.name,
                    "recipe": report.recipe,
                    "skipped": True,
                    "outputs": [str(provenance_path)],
                }
            )
            continue
        outputs = _REPORT_RUNNERS[report.recipe](output_dir, report.options)
        results.append(
            {
                "name": report.name,
                "recipe": report.recipe,
                "skipped": False,
                "outputs": [str(path) for path in outputs],
            }
        )
    return results


def _real_data_profit_dispersion(
    output_dir: Path,
    options: Mapping[str, Any],
) -> list[Path]:
    from reporting.real_data import run_profit_dispersion_report

    cohort = _mapping(options.get("cohort"), "reporting[].options.cohort")
    action_grid = _mapping(
        options.get("action_grid"), "reporting[].options.action_grid"
    )
    models = options.get("models", ["glm", "exact_spline_xgb"])
    if not isinstance(models, Sequence) or isinstance(models, (str, bytes, bytearray)):
        raise ValueError("reporting[].options.models must be a JSON array.")
    return run_profit_dispersion_report(
        output_dir=output_dir,
        models=models,
        sample_size=int(cohort.get("sample_size", 20_000)),
        seed=int(cohort.get("seed", 20260831)),
        u_min=float(action_grid.get("min", -0.10)),
        u_max=float(action_grid.get("max", 0.20)),
        u_count=int(action_grid.get("count", 301)),
        n_jobs=int(options.get("n_jobs", 1)),
    )


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Manifest {field} must be a JSON object.")
    return value


def _path_part(value: object) -> str:
    result = str(value).strip().replace(" ", "-").replace("/", "-")
    if not result or result in {".", ".."}:
        raise ValueError("Reporting names must contain a path-safe value.")
    return result


_REPORT_RUNNERS: dict[str, ReportRunner] = {
    "real_data_profit_dispersion": _real_data_profit_dispersion,
}


__all__ = [
    "ManifestReportSpec",
    "parse_manifest_reports",
    "run_manifest_reports",
]
