from __future__ import annotations

from pathlib import Path

import matplotlib.axes

from reporting.visualization import (
    plot_comparison_final_metric,
    plot_comparison_objective_curves,
    plot_comparison_u_curves,
)


def test_comparison_plots_create_expected_files(tmp_path: Path) -> None:
    trace_rows = [
        {"comparison": "constant", "estimator": "first_order", "step": 0, "objective": 1.0, "u": 0.0},
        {"comparison": "constant", "estimator": "first_order", "step": 1, "objective": 0.8, "u": 0.1},
        {"comparison": "linear", "estimator": "first_order", "step": 0, "objective": 1.1, "u": 0.0},
        {"comparison": "linear", "estimator": "first_order", "step": 1, "objective": 0.7, "u": 0.2},
        {"comparison": "constant", "estimator": "spsa", "step": 0, "objective": 1.2, "u": 0.0},
        {"comparison": "constant", "estimator": "spsa", "step": 1, "objective": 0.9, "u": 0.1},
    ]
    final_rows = [
        {
            "comparison": "constant",
            "estimator": "first_order",
            "final_value": 0.8,
            "final_objective_sum": 1.6,
            "final_u": 0.1,
            "mean_acceptance": 0.88,
        },
        {
            "comparison": "linear",
            "estimator": "first_order",
            "final_value": 0.7,
            "final_objective_sum": 1.4,
            "final_u": 0.2,
            "mean_acceptance": 0.91,
        },
        {
            "comparison": "constant",
            "estimator": "spsa",
            "final_value": 0.9,
            "final_objective_sum": 1.8,
            "final_u": 0.15,
            "mean_acceptance": 0.86,
        },
    ]

    plot_comparison_objective_curves(trace_rows, str(tmp_path))
    plot_comparison_u_curves(trace_rows, str(tmp_path))
    plot_comparison_final_metric(
        final_rows,
        str(tmp_path),
        metric_key="final_value",
        metric_label="Final objective value",
        filename="final_objective.png",
    )
    plot_comparison_final_metric(
        final_rows,
        str(tmp_path),
        metric_key="final_objective_sum",
        metric_label="Final summed objective value",
        filename="final_objective_sum.png",
    )
    plot_comparison_final_metric(
        final_rows,
        str(tmp_path),
        metric_key="final_u",
        metric_label="Final u",
        filename="final_u.png",
    )
    plot_comparison_final_metric(
        final_rows,
        str(tmp_path),
        metric_key="mean_acceptance",
        metric_label="Mean acceptance",
        filename="mean_acceptance.png",
    )

    assert (tmp_path / "objective_curves.png").exists()
    assert (tmp_path / "u_curves.png").exists()
    assert (tmp_path / "final_objective.png").exists()


def test_comparison_final_metric_uses_grouped_bars(monkeypatch, tmp_path: Path) -> None:
    final_rows = [
        {"comparison": "constant", "estimator": "first_order", "final_u": 0.8},
        {"comparison": "linear", "estimator": "first_order", "final_u": 0.7},
        {"comparison": "constant", "estimator": "spsa", "final_u": 0.9},
    ]
    bar_calls: list[dict[str, object]] = []
    original_bar = matplotlib.axes.Axes.bar

    def record_bar(self, *args, **kwargs):
        bar_calls.append(dict(kwargs))
        return original_bar(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", record_bar)

    plot_comparison_final_metric(
        final_rows,
        str(tmp_path),
        metric_key="final_u",
        metric_label="Final u",
        filename="final_u.png",
    )

    assert len(bar_calls) == 3
    assert {call["color"] for call in bar_calls} >= {"#1f77b4", "#d62728"}
    assert any(call.get("hatch") for call in bar_calls)
