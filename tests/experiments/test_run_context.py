from __future__ import annotations

from datetime import datetime
from pathlib import Path

from experiments.reporting import create_run_context


def test_create_run_context_defaults_to_results_root(monkeypatch, tmp_path) -> None:
    results_dir = tmp_path / "results"
    monkeypatch.setenv("GENERALI_RESULTS_ROOT", str(results_dir))
    started_at = datetime(2026, 1, 1, 0, 0, 0)

    run_context = create_run_context("demo experiment", started_at=started_at)

    assert run_context.run_dir == results_dir / "demo-experiment__20260101_000000"
    assert run_context.plots_dir == run_context.run_dir / "plots"
    assert run_context.run_dir.exists()


def test_create_run_context_uses_run_dir_verbatim(tmp_path) -> None:
    run_dir = tmp_path / "variant" / "seeds" / "seed-7"

    run_context = create_run_context("demo experiment", run_dir=run_dir)

    assert run_context.run_dir == run_dir
    assert run_context.run_id == "seed-7"
    assert run_context.run_dir.exists()
