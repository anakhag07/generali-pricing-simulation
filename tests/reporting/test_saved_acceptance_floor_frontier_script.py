from __future__ import annotations

from pathlib import Path

from scripts.plot_saved_acceptance_floor_frontier import _resolve_csv_path, main


def _write_sweep_csv(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "run_name,estimator,c,u,mean_acceptance,value,constraint_violation",
                "acceptance_floor-0.5,first_order,0.5,0.33,0.56,-39.9,0.0",
                "acceptance_floor-0.5,spsa,0.5,0.34,0.55,-39.8,0.0",
                "acceptance_floor-0.95,first_order,0.95,-0.01,0.95,3.7,0.0",
            ]
        ),
        encoding="utf-8",
    )


def test_resolve_csv_path_prefers_latest_frontier_directory(tmp_path) -> None:
    older = tmp_path / "acceptance_floor_frontier_20260424_001823"
    newer = tmp_path / "acceptance_floor_frontier_20260424_002443"
    older.mkdir()
    newer.mkdir()
    _write_sweep_csv(older / "acceptance_floor_sweep.csv")
    _write_sweep_csv(newer / "acceptance_floor_sweep.csv")

    resolved = _resolve_csv_path(tmp_path)

    assert resolved == newer / "acceptance_floor_sweep.csv"


def test_main_writes_first_order_only_pareto_plots(tmp_path) -> None:
    csv_path = tmp_path / "acceptance_floor_sweep.csv"
    plot_dir = tmp_path / "plots"
    _write_sweep_csv(csv_path)

    main([str(csv_path), "--output-dir", str(plot_dir)])

    assert (plot_dir / "pareto_objective_acceptance_first_order.png").exists()
    assert (plot_dir / "pareto_u_acceptance_first_order.png").exists()
