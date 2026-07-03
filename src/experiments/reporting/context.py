"""Run output context helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


@dataclass(frozen=True)
class RunContext:
    """Filesystem context for one executed experiment run."""

    experiment_name: str
    run_id: str
    run_dir: Path
    plots_dir: Path
    started_at: datetime


def create_run_context(
    experiment_name: str,
    runs_root: str = "outputs",
    started_at: datetime | None = None,
    run_dir: Path | None = None,
) -> RunContext:
    """Create the standard output directory context for a run.

    When ``run_dir`` is given it is used verbatim (no timestamp segment), letting a
    caller place several runs -- e.g. per-seed replicates -- under one shared
    parent directory. Otherwise the run lands in ``runs_root/<name>/<timestamp>``.
    """
    timestamp = started_at or datetime.now()
    if run_dir is not None:
        resolved_dir = Path(run_dir)
        run_id = resolved_dir.name
    else:
        run_id = timestamp.strftime("%Y%m%d_%H%M%S")
        safe_name = _sanitize_name(experiment_name)
        resolved_dir = Path(runs_root) / safe_name / run_id
    resolved_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        experiment_name=experiment_name,
        run_id=run_id,
        run_dir=resolved_dir,
        plots_dir=resolved_dir / "plots",
        started_at=timestamp,
    )


def _sanitize_name(name: str) -> str:
    cleaned = name.strip().replace(" ", "-")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", cleaned)
    return cleaned or "run"


__all__ = ["RunContext", "create_run_context"]
