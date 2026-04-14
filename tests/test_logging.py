"""Tests for console step logging."""

from __future__ import annotations

from reporting.logging import log_step


def test_log_step_prints_model_metrics(capsys) -> None:
    log_step(
        "first_order",
        3,
        0.52,
        -0.09,
        grad_norm=0.1,
        mean_acceptance=0.81,
        projected_loss=120.0,
        projected_revenue=0.04,
    )

    captured = capsys.readouterr().out.strip()
    assert "[first_order] step=3" in captured
    assert "mean_acceptance=0.8100" in captured
    assert "projected_loss=120.0000" in captured
    assert "revenue=0.0400" in captured
