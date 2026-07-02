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
) -> RunContext:
    """Create the standard output directory context for a run."""
    timestamp = started_at or datetime.now()
    run_id = timestamp.strftime("%Y%m%d_%H%M%S")
    safe_name = _sanitize_name(experiment_name)
    run_dir = Path(runs_root) / safe_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        experiment_name=experiment_name,
        run_id=run_id,
        run_dir=run_dir,
        plots_dir=run_dir / "plots",
        started_at=timestamp,
    )


def _sanitize_name(name: str) -> str:
    cleaned = name.strip().replace(" ", "-")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", cleaned)
    return cleaned or "run"


__all__ = ["RunContext", "create_run_context"]
