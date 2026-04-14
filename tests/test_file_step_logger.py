"""Tests for FileStepLogger CSV output."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from experiments.reporters import FileStepLogger, RunContext


@pytest.fixture
def run_context(tmp_path: Path) -> RunContext:
    """Create a RunContext pointing to a temporary directory."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        experiment_name="test",
        run_id="20260101_000000",
        run_dir=run_dir,
        plots_dir=run_dir / "plots",
        started_at=datetime(2026, 1, 1, 0, 0, 0),
    )


def test_file_logger_writes_header(run_context: RunContext) -> None:
    """FileStepLogger should write CSV header on start."""
    logger = FileStepLogger()
    logger.on_start(run_context, config=None)  # type: ignore[arg-type]
    logger.on_end(run_context, result=None)  # type: ignore[arg-type]

    csv_path = run_context.run_dir / "steps.csv"
    assert csv_path.exists()
    content = csv_path.read_text()
    assert content.startswith(
        "method,step,u,value,grad_norm,step_size,mean_acceptance,projected_loss,projected_revenue\n"
    )


def test_file_logger_writes_rows(run_context: RunContext) -> None:
    """FileStepLogger should write CSV rows for each log_step call."""
    logger = FileStepLogger()
    logger.on_start(run_context, config=None)  # type: ignore[arg-type]

    logger.log_step(
        "first-order",
        1,
        0.5,
        0.3,
        0.01,
        0.001,
        0.8,
        120.0,
        0.05,
    )
    logger.log_step("first-order", 2, 0.6, 0.25, 0.008, 0.001, 0.75, 120.0, 0.06)
    logger.log_step("spsa", 0, 0.5, 0.3, 0.02, None, None, None, None)

    logger.on_end(run_context, result=None)  # type: ignore[arg-type]

    csv_path = run_context.run_dir / "steps.csv"
    lines = csv_path.read_text().strip().split("\n")

    assert len(lines) == 4  # header + 3 rows
    assert lines[0] == (
        "method,step,u,value,grad_norm,step_size,mean_acceptance,projected_loss,projected_revenue"
    )
    assert lines[1] == "first-order,1,0.500000,0.300000,0.010000,0.001000,0.800000,120.000000,0.050000"
    assert lines[2] == "first-order,2,0.600000,0.250000,0.008000,0.001000,0.750000,120.000000,0.060000"
    assert lines[3] == "spsa,0,0.500000,0.300000,0.020000,,,,"


def test_file_logger_handles_none_grad_norm(run_context: RunContext) -> None:
    """FileStepLogger should handle None grad_norm gracefully."""
    logger = FileStepLogger()
    logger.on_start(run_context, config=None)  # type: ignore[arg-type]

    logger.log_step("test", 1, 0.5, 0.3, None, 0.01, None, None, None)

    logger.on_end(run_context, result=None)  # type: ignore[arg-type]

    csv_path = run_context.run_dir / "steps.csv"
    lines = csv_path.read_text().strip().split("\n")

    assert len(lines) == 2
    assert lines[1] == "test,1,0.500000,0.300000,,0.010000,,,"


def test_file_logger_on_end_closes_file(run_context: RunContext) -> None:
    """FileStepLogger should close the file handle on end."""
    logger = FileStepLogger()
    logger.on_start(run_context, config=None)  # type: ignore[arg-type]
    logger.log_step("test", 1, 0.5, 0.3, 0.01, 0.001, 0.8, 120.0, 0.05)
    logger.on_end(run_context, result=None)  # type: ignore[arg-type]

    # After on_end, _file should be None
    assert logger._file is None

    # Writing after close should be a no-op (not raise)
    logger.log_step("test", 2, 0.6, 0.25, 0.008, 0.001, 0.7, 121.0, 0.06)

    # File content should still only have 1 data row
    csv_path = run_context.run_dir / "steps.csv"
    lines = csv_path.read_text().strip().split("\n")
    assert len(lines) == 2
