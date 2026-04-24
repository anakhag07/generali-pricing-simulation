"""Plot Pareto frontiers from a saved acceptance-floor sweep CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

from reporting.visualization import _plot_sweep_pareto_frontier


def _resolve_csv_path(source: Path) -> Path:
    if source.is_file():
        if source.name != "acceptance_floor_sweep.csv":
            raise ValueError(
                f"Expected 'acceptance_floor_sweep.csv', got '{source.name}'."
            )
        return source

    direct_csv = source / "acceptance_floor_sweep.csv"
    if direct_csv.is_file():
        return direct_csv

    frontier_csvs = sorted(
        path
        for path in source.glob("acceptance_floor_frontier_*/acceptance_floor_sweep.csv")
        if path.is_file()
    )
    if frontier_csvs:
        return frontier_csvs[-1]

    raise FileNotFoundError(
        "Could not find 'acceptance_floor_sweep.csv' at the given path or under "
        "'acceptance_floor_frontier_*' subdirectories."
    )


def _load_rows(csv_path: Path) -> list[dict[str, float | str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, float | str]] = []
        for row in reader:
            rows.append(
                {
                    "run_name": row["run_name"],
                    "estimator": row["estimator"],
                    "c": float(row["c"]),
                    "u": float(row["u"]),
                    "mean_acceptance": float(row["mean_acceptance"]),
                    "value": float(row["value"]),
                }
            )
    if not rows:
        raise ValueError(f"No sweep rows found in '{csv_path}'.")
    return rows


def _filter_rows(
    rows: Sequence[dict[str, float | str]], estimator: str
) -> list[dict[str, float | str]]:
    filtered = [row for row in rows if row["estimator"] == estimator]
    if not filtered:
        raise ValueError(f"No rows found for estimator '{estimator}'.")
    return filtered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot acceptance-floor Pareto frontiers from saved sweep outputs "
            "without rerunning optimization."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help=(
            "Path to an acceptance_floor_sweep.csv file, a frontier directory that "
            "contains it, or a project output directory with "
            "acceptance_floor_frontier_* subdirectories."
        ),
    )
    parser.add_argument(
        "--estimator",
        default="first_order",
        help="Estimator to plot from the saved sweep CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where the filtered Pareto plots will be written.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    csv_path = _resolve_csv_path(args.source)
    output_dir = args.output_dir if args.output_dir is not None else csv_path.parent
    rows = _filter_rows(_load_rows(csv_path), estimator=args.estimator)

    plot_dir = str(output_dir)
    _plot_sweep_pareto_frontier(
        rows,
        plot_dir,
        sweep_key="c",
        sweep_label="Acceptance floor c",
        y_key="value",
        y_label="Final objective value",
        filename=f"pareto_objective_acceptance_{args.estimator}.png",
    )
    _plot_sweep_pareto_frontier(
        rows,
        plot_dir,
        sweep_key="c",
        sweep_label="Acceptance floor c",
        y_key="u",
        y_label="Final u",
        filename=f"pareto_u_acceptance_{args.estimator}.png",
    )

    print(f"Read saved sweep rows from {csv_path}.")
    print(f"Wrote Pareto plots for estimator '{args.estimator}' to {output_dir}.")


if __name__ == "__main__":
    main()
