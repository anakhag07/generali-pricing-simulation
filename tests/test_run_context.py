from __future__ import annotations

from datetime import datetime
from pathlib import Path

from experiments.reporters import create_run_context


def test_create_run_context_defaults_to_outputs(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    started_at = datetime(2026, 1, 1, 0, 0, 0)

    run_context = create_run_context("demo experiment", started_at=started_at)

    assert run_context.run_dir == Path("outputs") / "demo-experiment" / "20260101_000000"
    assert run_context.plots_dir == run_context.run_dir / "plots"
    assert run_context.run_dir.exists()
