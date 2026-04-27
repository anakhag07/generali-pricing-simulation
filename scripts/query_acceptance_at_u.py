"""Query mean acceptance for constant action values from a config preset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.configs import get_config
from objective import default_rng, mean_acceptance_at_constant_u, sample_states


@dataclass(frozen=True)
class MeanAcceptanceRow:
    """Mean acceptance for one fixed action value."""

    u: float
    n: int
    mean_acceptance: float


def _resolve_x_array(config: object, n_rows: int | None) -> np.ndarray:
    x_fixed = getattr(config, "x_fixed")
    if x_fixed is not None:
        x_arr = np.asarray(x_fixed, dtype=float)
    else:
        rng = default_rng(int(getattr(config, "seed")))
        x_arr = sample_states(
            rng,
            int(getattr(config, "n_samples")),
            int(getattr(config, "state_dim")),
        )

    if n_rows is not None:
        if n_rows <= 0:
            raise ValueError("n_rows must be positive when provided.")
        x_arr = x_arr[:n_rows]
    if x_arr.shape[0] == 0:
        raise ValueError("No rows available for acceptance query.")
    return x_arr


def query_mean_acceptance(
    config: object,
    u_values: Sequence[float],
    *,
    n_rows: int | None = None,
) -> list[MeanAcceptanceRow]:
    """Evaluate mean acceptance at each constant action value."""
    if len(u_values) == 0:
        raise ValueError("At least one u value is required.")

    x_arr = _resolve_x_array(config, n_rows)
    objective = getattr(config, "objective")
    rows: list[MeanAcceptanceRow] = []
    for u in u_values:
        u_val = float(u)
        mean_acceptance = mean_acceptance_at_constant_u(objective, x_arr, u_val)
        if mean_acceptance is None:
            raise ValueError(
                "Config objective does not support mean_acceptance_at_u(x_batch, u)."
            )
        rows.append(
            MeanAcceptanceRow(
                u=u_val,
                n=int(x_arr.shape[0]),
                mean_acceptance=float(mean_acceptance),
            )
        )
    return rows


def _format_table(rows: Sequence[MeanAcceptanceRow]) -> str:
    lines = [f"{'u':>12} {'n':>8} {'mean_acceptance':>18}"]
    for row in rows:
        lines.append(f"{row.u:12.6f} {row.n:8d} {row.mean_acceptance:18.6f}")
    return "\n".join(lines)


def _write_csv(rows: Sequence[MeanAcceptanceRow], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["u", "n", "mean_acceptance"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "u": row.u,
                    "n": row.n,
                    "mean_acceptance": row.mean_acceptance,
                }
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query mean acceptance for constant u values without running optimization."
    )
    parser.add_argument(
        "--preset",
        default="real_data_glm_softmax_policy_base",
        help="Config preset to load. Defaults to real_data_glm_softmax_policy_base.",
    )
    parser.add_argument(
        "--u",
        type=float,
        nargs="+",
        required=True,
        help="One or more constant action values to evaluate.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="Use only the first N rows from the preset's state batch.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path for writing u,n,mean_acceptance as CSV.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config = get_config(args.preset)
    rows = query_mean_acceptance(config, args.u, n_rows=args.n_rows)
    print(_format_table(rows))
    if args.csv is not None:
        _write_csv(rows, args.csv)
        print(f"Wrote acceptance query CSV to {args.csv}.")


if __name__ == "__main__":
    main()
