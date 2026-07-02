from __future__ import annotations

from reporting.visualization import (
    plot_sweep_pareto_frontier,
    plot_sweep_tradeoffs,
)


def _c_points() -> list[dict[str, float | str]]:
    return [
        {
            "run_name": "run-0",
            "estimator": "first_order",
            "c": 0.50,
            "u": 0.10,
            "mean_acceptance": 0.90,
            "value": -1.2,
        },
        {
            "run_name": "run-1",
            "estimator": "first_order",
            "c": 0.95,
            "u": 0.02,
            "mean_acceptance": 0.96,
            "value": -0.85,
        },
        {
            "run_name": "run-0",
            "estimator": "spsa",
            "c": 0.50,
            "u": 0.11,
            "mean_acceptance": 0.89,
            "value": -1.1,
        },
        {
            "run_name": "run-1",
            "estimator": "spsa",
            "c": 0.95,
            "u": 0.03,
            "mean_acceptance": 0.95,
            "value": -0.87,
        },
    ]


def test_plot_generic_sweep_tradeoffs_writes_png(tmp_path) -> None:
    plot_sweep_tradeoffs(
        _c_points(),
        str(tmp_path),
        sweep_key="c",
        sweep_label="Acceptance floor c",
        filename="c_vs_u_acceptance.png",
    )

    assert (tmp_path / "c_vs_u_acceptance.png").exists()


def test_plot_generic_sweep_pareto_frontier_writes_pngs(tmp_path) -> None:
    points = _c_points()
    plot_sweep_pareto_frontier(
        points,
        str(tmp_path),
        sweep_key="c",
        sweep_label="Acceptance floor c",
        y_key="value",
        y_label="Final objective value",
        filename="pareto_objective_acceptance.png",
    )
    plot_sweep_pareto_frontier(
        points,
        str(tmp_path),
        sweep_key="c",
        sweep_label="Acceptance floor c",
        y_key="u",
        y_label="Final u",
        filename="pareto_u_acceptance.png",
    )

    assert (tmp_path / "pareto_objective_acceptance.png").exists()
    assert (tmp_path / "pareto_u_acceptance.png").exists()
