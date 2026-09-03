"""Collect and render all-customer XGBoost feature-effect curves.

For vehicle age, policy tenure, and customer age, this task fixes the selected
feature to each value on a robust integer grid while retaining every other
customer covariate.  It evaluates acceptance at ``u=0.08`` and predicted claims
for all eligible customers, then stores and plots the pointwise population mean
and standard deviation.

The collected CSV makes plot-only reruns fast.  Pass ``--recompute`` to rerun
model inference.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.loader import eligible_csv_row_indices, load_model_artifacts, load_x_frame
from scripts.analyze_model_acceptance_features import (
    _predict_acceptance_matrix,
    _predict_loss,
)


DEFAULT_RESULTS_ROOT = Path(
    os.environ.get(
        "GENERALI_RESULTS_ROOT",
        Path.home() / "projects" / "generali-pricing" / "results",
    )
)
DEFAULT_ANALYSIS_DIR = (
    DEFAULT_RESULTS_ROOT
    / "model-acceptance-feature-analysis"
    / "sweeps"
    / "20260809_105106"
)
ACCEPTANCE_COLOR = "tab:blue"
CLAIMS_COLOR = "tab:red"


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    column: str
    label: str
    grid: np.ndarray


FEATURE_SPECS = (
    FeatureSpec(
        key="vehicle_age",
        column="X_vehicle_age",
        label="Vehicle Age",
        grid=np.arange(0.0, 38.0, 1.0),
    ),
    FeatureSpec(
        key="policy_tenure",
        column="X_policy_tenure",
        label="Policy Tenure",
        grid=np.arange(0.0, 21.0, 1.0),
    ),
    FeatureSpec(
        key="customer_age",
        column="X_age",
        label="Customer Age",
        # Include the supported adult lower tail; the only younger records are
        # two implausible observations at ages 9 and 15.
        grid=np.arange(18.0, 87.0, 1.0),
    ),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="PDF destination; defaults to ANALYSIS_DIR/plots.",
    )
    parser.add_argument("--fixed-u", type=float, default=0.08)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute model predictions even when the collected CSV exists.",
    )
    args = parser.parse_args(argv)
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.n_jobs <= 0:
        parser.error("--n-jobs must be positive")
    return args


def _set_thread_count(artifact: object, n_jobs: int) -> None:
    model = getattr(artifact, "model")
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=n_jobs)


def _empty_stats() -> dict[str, dict[str, np.ndarray]]:
    return {
        spec.key: {
            "acceptance_sum": np.zeros(spec.grid.size, dtype=float),
            "acceptance_sum_sq": np.zeros(spec.grid.size, dtype=float),
            "claims_sum": np.zeros(spec.grid.size, dtype=float),
            "claims_sum_sq": np.zeros(spec.grid.size, dtype=float),
        }
        for spec in FEATURE_SPECS
    }


def _collect_effects(
    *,
    fixed_u: float,
    chunk_size: int,
    n_jobs: int,
) -> pd.DataFrame:
    eligible = eligible_csv_row_indices("linear")
    base_frame = load_x_frame("linear", row_indices=eligible)
    acceptance_artifact, claims_artifact = load_model_artifacts("xgb")
    _set_thread_count(acceptance_artifact, n_jobs)
    _set_thread_count(claims_artifact, n_jobs)

    stats = _empty_stats()
    n_chunks = int(np.ceil(len(base_frame) / chunk_size))
    for chunk_index, start in enumerate(range(0, len(base_frame), chunk_size), start=1):
        stop = min(start + chunk_size, len(base_frame))
        customer_chunk = base_frame.iloc[start:stop].copy()
        for spec in FEATURE_SPECS:
            feature_chunk = customer_chunk.copy()
            current = stats[spec.key]
            for grid_index, value in enumerate(spec.grid):
                feature_chunk[spec.column] = value
                acceptance = _predict_acceptance_matrix(
                    acceptance_artifact,
                    feature_chunk,
                    [fixed_u],
                )[:, 0]
                claims = _predict_loss(claims_artifact, feature_chunk)
                current["acceptance_sum"][grid_index] += float(
                    np.sum(acceptance, dtype=float)
                )
                current["acceptance_sum_sq"][grid_index] += float(
                    np.sum(acceptance * acceptance, dtype=float)
                )
                current["claims_sum"][grid_index] += float(
                    np.sum(claims, dtype=float)
                )
                current["claims_sum_sq"][grid_index] += float(
                    np.sum(claims * claims, dtype=float)
                )
            print(
                f"chunk {chunk_index}/{n_chunks}: {spec.label} complete "
                f"({stop - start} customers)",
                flush=True,
            )

    rows: list[dict[str, object]] = []
    n_customers = len(base_frame)
    for spec in FEATURE_SPECS:
        current = stats[spec.key]
        for target in ("acceptance", "claims"):
            sums = current[f"{target}_sum"]
            sums_sq = current[f"{target}_sum_sq"]
            means = sums / n_customers
            variance = np.maximum(sums_sq / n_customers - means * means, 0.0)
            standard_deviations = np.sqrt(variance)
            for value, mean, std in zip(
                spec.grid,
                means,
                standard_deviations,
                strict=True,
            ):
                rows.append(
                    {
                        "model": "xgb",
                        "feature": spec.key,
                        "feature_column": spec.column,
                        "feature_label": spec.label,
                        "feature_value": float(value),
                        "target": target,
                        "n_customers": n_customers,
                        "mean": float(mean),
                        "std": float(std),
                        "fixed_u": fixed_u if target == "acceptance" else np.nan,
                        "std_ddof": 0,
                    }
                )
    return pd.DataFrame(rows)


def _validate_collected(frame: pd.DataFrame, *, fixed_u: float) -> None:
    required = {
        "model",
        "feature",
        "feature_label",
        "feature_value",
        "target",
        "n_customers",
        "mean",
        "std",
        "fixed_u",
        "std_ddof",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Collected feature-effect CSV is missing {sorted(missing)}")
    expected = {(spec.key, target) for spec in FEATURE_SPECS for target in ("acceptance", "claims")}
    observed = set(zip(frame["feature"], frame["target"], strict=True))
    if observed != expected:
        raise ValueError("Collected feature-effect CSV has unexpected feature/target pairs")
    for spec in FEATURE_SPECS:
        for target in ("acceptance", "claims"):
            values = np.sort(
                frame.loc[
                    frame["feature"].eq(spec.key) & frame["target"].eq(target),
                    "feature_value",
                ].to_numpy(dtype=float)
            )
            if not np.array_equal(values, spec.grid):
                raise ValueError(
                    f"Collected CSV grid differs for {spec.key}/{target}; "
                    "rerun with --recompute"
                )
    acceptance_u = frame.loc[frame["target"].eq("acceptance"), "fixed_u"]
    if not np.allclose(acceptance_u, fixed_u):
        raise ValueError("Collected CSV fixed_u differs; rerun with --recompute")


def _plot_effect(
    frame: pd.DataFrame,
    *,
    feature_label: str,
    target: str,
    output_path: Path,
) -> None:
    ordered = frame.sort_values("feature_value")
    x = ordered["feature_value"].to_numpy(dtype=float)
    mean = ordered["mean"].to_numpy(dtype=float)
    std = ordered["std"].to_numpy(dtype=float)
    lower = mean - std
    upper = mean + std
    if target == "acceptance":
        lower = np.clip(lower, 0.0, 1.0)
        upper = np.clip(upper, 0.0, 1.0)
        color = ACCEPTANCE_COLOR
        target_label = "Acceptance Probability"
    else:
        color = CLAIMS_COLOR
        target_label = "Claims"

    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.fill_between(x, lower, upper, color=color, alpha=0.20, linewidth=0)
    ax.plot(x, mean, color=color, linewidth=3)
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_xlabel(feature_label, fontsize=12)
    ax.set_ylabel(
        "Acceptance Probability" if target == "acceptance" else "Predicted Claims",
        fontsize=12,
    )
    ax.set_title(
        f"Predicted Effect of {feature_label} on {target_label}",
        fontsize=16,
    )
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    ax.margins(y=0.12)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)


def _render_effects(frame: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec in FEATURE_SPECS:
        for target in ("acceptance", "claims"):
            selected = frame.loc[
                frame["feature"].eq(spec.key) & frame["target"].eq(target)
            ]
            output_path = output_dir / f"xgb_{spec.key}_vs_{target}.pdf"
            _plot_effect(
                selected,
                feature_label=spec.label,
                target=target,
                output_path=output_path,
            )
            written.append(output_path)
    return written


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    analysis_dir = args.analysis_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else analysis_dir / "plots"
    )
    collected_path = analysis_dir / "xgb_customer_feature_effect_mean_std.csv"
    if args.recompute or not collected_path.is_file():
        collected = _collect_effects(
            fixed_u=args.fixed_u,
            chunk_size=args.chunk_size,
            n_jobs=args.n_jobs,
        )
        collected.to_csv(collected_path, index=False)
        print(f"Wrote {len(collected)} rows to {collected_path}")
    else:
        collected = pd.read_csv(collected_path)
        print(f"Reusing {collected_path}")
    _validate_collected(collected, fixed_u=args.fixed_u)
    written = _render_effects(collected, output_dir)
    print(f"Wrote {len(written)} PDFs to {output_dir}")
    for path in written:
        print(path.name)


if __name__ == "__main__":
    main()
