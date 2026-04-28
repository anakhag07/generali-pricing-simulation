"""Query mean acceptance for constant action values from a config preset."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from data.loader import FEATURE_COLS_GLM, FEATURE_COLS_XGB, load_observed_u_array
from experiments.configs import get_config
from objective import default_rng, mean_acceptance_at_constant_u, sample_states

DEFAULT_MODEL_PRESETS: dict[str, str] = {
    "glm": "real_data_glm_softmax_policy_base",
    "xgb": "real_data_xgb_base",
}
DEFAULT_OUTPUT_ROOT = Path("outputs") / "acceptance_queries"


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


def _resolve_u_values(
    explicit_u: Sequence[float] | None,
    u_count: int | None,
    u_min: float,
    u_max: float,
) -> np.ndarray:
    if explicit_u is not None and u_count is not None:
        raise ValueError("Provide either explicit --u values or --u-count, not both.")
    if explicit_u is None and u_count is None:
        raise ValueError("Provide either explicit --u values or --u-count.")
    if explicit_u is not None:
        if len(explicit_u) == 0:
            raise ValueError("At least one u value is required.")
        return np.asarray([float(u) for u in explicit_u], dtype=float)
    if u_count is None or u_count <= 0:
        raise ValueError("u_count must be positive when provided.")
    if u_min > u_max:
        raise ValueError("u_min must be <= u_max.")
    return np.linspace(float(u_min), float(u_max), int(u_count), dtype=float)


def _resolve_preset_and_model_type(
    preset: str | None,
    model_type: Literal["glm", "xgb"] | None,
) -> tuple[str, Literal["glm", "xgb"] | None]:
    if preset is not None and model_type is not None:
        raise ValueError("Provide either --preset or --model-type, not both.")
    if model_type is not None:
        return DEFAULT_MODEL_PRESETS[model_type], model_type
    return preset or DEFAULT_MODEL_PRESETS["glm"], None


def _infer_model_type(config: object) -> Literal["glm", "xgb"]:
    state_dim = int(getattr(config, "state_dim"))
    if state_dim == len(FEATURE_COLS_GLM):
        return "glm"
    if state_dim == len(FEATURE_COLS_XGB):
        return "xgb"
    raise ValueError("Could not infer model type from config.state_dim.")


def _resolve_output_dir(
    output_root: Path,
    output_subdir: str | None,
    model_type: Literal["glm", "xgb"],
) -> Path:
    return output_root / (output_subdir if output_subdir is not None else model_type)


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


def _plot_acceptance_curve(rows: Sequence[MeanAcceptanceRow], output_dir: Path) -> Path:
    if not rows:
        raise ValueError("At least one result row is required to plot acceptance.")
    output_dir.mkdir(parents=True, exist_ok=True)
    u_values = np.asarray([row.u for row in rows], dtype=float)
    mean_acceptance = np.asarray([row.mean_acceptance for row in rows], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.75))
    ax.plot(u_values, mean_acceptance, marker="o", linewidth=1.8, markersize=3.5)
    ax.set_xlabel("constant u")
    ax.set_ylabel("Mean acceptance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "constant_u_acceptance_curve.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _plot_constant_u_histogram(
    observed_u: np.ndarray,
    u_values: Sequence[float],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    observed_u_arr = np.asarray(observed_u, dtype=float).reshape(-1)
    u_arr = np.asarray(u_values, dtype=float).reshape(-1)
    if observed_u_arr.size == 0:
        raise ValueError("observed_u must contain at least one value.")
    if u_arr.size == 0:
        raise ValueError("u_values must contain at least one value.")

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.75))
    ax.hist(
        observed_u_arr,
        bins="auto",
        density=True,
        label="observed U",
        color="#bdbdbd",
        edgecolor="#969696",
        alpha=0.65,
        linewidth=0.8,
    )
    ax.plot(
        u_arr,
        np.zeros_like(u_arr),
        linestyle="None",
        marker="|",
        markersize=12,
        markeredgewidth=1.2,
        color="#08519c",
        alpha=0.8,
        label="sampled constant u",
        transform=ax.get_xaxis_transform(),
    )
    ax.set_xlabel("u")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "constant_u_histograms.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


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
        default=None,
        help="Config preset to load. Defaults to real_data_glm_softmax_policy_base.",
    )
    parser.add_argument(
        "--model-type",
        choices=("glm", "xgb"),
        default=None,
        help="Convenience selector for the default GLM or XGB softmax preset.",
    )
    parser.add_argument(
        "--u",
        type=float,
        nargs="+",
        default=None,
        help="One or more constant action values to evaluate.",
    )
    parser.add_argument(
        "--u-count",
        type=int,
        default=None,
        help="Number of evenly spaced constant u values to sample.",
    )
    parser.add_argument(
        "--u-min",
        type=float,
        default=-0.5,
        help="Minimum u value for --u-count grids. Defaults to -0.5.",
    )
    parser.add_argument(
        "--u-max",
        type=float,
        default=0.5,
        help="Maximum u value for --u-count grids. Defaults to 0.5.",
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for plots. Defaults to outputs/acceptance_queries.",
    )
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Subdirectory under --output-root. Defaults to glm or xgb.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    u_values = _resolve_u_values(args.u, args.u_count, args.u_min, args.u_max)
    preset, requested_model_type = _resolve_preset_and_model_type(args.preset, args.model_type)
    config = get_config(preset)
    model_type = requested_model_type or _infer_model_type(config)
    rows = query_mean_acceptance(config, u_values, n_rows=args.n_rows)
    print(_format_table(rows))
    if args.csv is not None:
        _write_csv(rows, args.csv)
        print(f"Wrote acceptance query CSV to {args.csv}.")

    output_dir = _resolve_output_dir(args.output_root, args.output_subdir, model_type)
    observed_u = load_observed_u_array(model_type, n_rows=rows[0].n)
    histogram_path = _plot_constant_u_histogram(observed_u, u_values, output_dir)
    curve_path = _plot_acceptance_curve(rows, output_dir)
    print(f"Wrote constant-u histogram to {histogram_path}.")
    print(f"Wrote acceptance curve to {curve_path}.")


if __name__ == "__main__":
    main()
