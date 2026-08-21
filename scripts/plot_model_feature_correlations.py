"""Render correlation diagnostics for the model-feature analysis outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data.dataset_metadata import DATASET_PATH, OBSERVED_CHURN_COL
from data.loader import (
    eligible_csv_row_indices,
    load_observed_loss_array,
    load_observed_u_array,
    load_x_frame,
)
from experiments.paths import results_root


PROJECT_NAME = "model-acceptance-feature-analysis"
MODEL_ORDER = ("glm", "xgb", "spline")
MODEL_LABELS = {"glm": "GLM", "xgb": "XGBoost", "spline": "Spline"}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        help="Collected sweep directory; defaults to the latest collected sweep.",
    )
    parser.add_argument("--sample-n-rows", type=int, default=50_000)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args(argv)
    if args.sample_n_rows <= 0:
        raise ValueError("sample_n_rows must be positive.")
    if args.top_k <= 0:
        raise ValueError("top_k must be positive.")
    return args


def _latest_analysis_dir() -> Path:
    sweeps = results_root() / PROJECT_NAME / "sweeps"
    candidates = sorted(
        path
        for path in sweeps.iterdir()
        if path.is_dir() and (path / "feature_importance.csv").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No collected analysis sweep found under {sweeps}.")
    return candidates[-1]


def _sample_rows(n_rows: int, seed: int) -> np.ndarray:
    eligible = eligible_csv_row_indices("linear")
    sample_size = min(n_rows, len(eligible))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(eligible, size=sample_size, replace=False))


def _observed_acceptance(row_indices: np.ndarray) -> np.ndarray:
    churn = pd.read_csv(DATASET_PATH, sep=";", usecols=[OBSERVED_CHURN_COL])
    return 1.0 - churn.iloc[row_indices][OBSERVED_CHURN_COL].to_numpy(float)


def _ordered_top_features(
    importance: pd.DataFrame,
    *,
    target: str,
    top_k: int,
) -> list[str]:
    rows = importance[
        (importance["target"] == target) & (importance["rank"] <= top_k)
    ].copy()
    rows["model_order"] = rows["model"].map(
        {name: index for index, name in enumerate(MODEL_ORDER)}
    )
    rows = rows.sort_values(["model_order", "rank"])
    return list(dict.fromkeys(rows["feature"].tolist()))


def _numeric_top_features(
    frame: pd.DataFrame,
    importance: pd.DataFrame,
    *,
    target: str,
    top_k: int,
) -> tuple[list[str], list[str]]:
    top_features = _ordered_top_features(importance, target=target, top_k=top_k)
    numeric = [
        feature
        for feature in top_features
        if pd.api.types.is_numeric_dtype(frame[feature]) and feature != "X_district"
    ]
    excluded = [feature for feature in top_features if feature not in numeric]
    return numeric, excluded


def _display_label(column: str) -> str:
    aliases = {
        "historical_u": "Historical u",
        "observed_acceptance": "Observed acceptance",
        "observed_loss": "Observed claims",
    }
    return aliases.get(column, column.removeprefix("X_").replace("_", " ").title())


def _plot_correlation_matrix(
    correlation: pd.DataFrame,
    *,
    title: str,
    output_path: Path,
) -> None:
    n_columns = len(correlation.columns)
    figure_size = max(7.0, 0.72 * n_columns + 2.5)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    image = ax.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    labels = [_display_label(column) for column in correlation.columns]
    ax.set_xticks(np.arange(n_columns), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(n_columns), labels=labels)
    for row in range(n_columns):
        for column in range(n_columns):
            value = correlation.iloc[row, column]
            color = "white" if abs(value) >= 0.55 else "black"
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=color)
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman correlation")
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), format="pdf")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _importance_rank_correlations(importance: pd.DataFrame) -> dict[str, pd.DataFrame]:
    correlations: dict[str, pd.DataFrame] = {}
    for target in ("acceptance", "loss"):
        pivot = importance[importance["target"] == target].pivot(
            index="feature", columns="model", values="importance_mean"
        )
        pivot = pivot.reindex(columns=MODEL_ORDER)
        correlations[target] = pivot.corr(method="spearman")
    return correlations


def _plot_importance_rank_correlations(
    correlations: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
    image = None
    for ax, target in zip(axes, ("acceptance", "loss"), strict=True):
        correlation = correlations[target]
        image = ax.imshow(correlation.to_numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
        labels = [MODEL_LABELS[model] for model in correlation.columns]
        ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = correlation.iloc[row, column]
                color = "white" if value <= 0.45 else "black"
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=color)
        ax.set_title(target.title())
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.046, pad=0.04)
    colorbar.set_label("Spearman rank correlation")
    fig.suptitle("Agreement of feature-importance rankings across models")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_correlations(
    analysis_dir: Path,
    *,
    sample_n_rows: int,
    sample_seed: int,
    top_k: int,
) -> dict[str, object]:
    importance_path = analysis_dir / "feature_importance.csv"
    importance = pd.read_csv(importance_path)
    row_indices = _sample_rows(sample_n_rows, sample_seed)
    frame = load_x_frame("linear", row_indices=row_indices)
    frame["historical_u"] = load_observed_u_array("linear", row_indices=row_indices)
    frame["observed_acceptance"] = _observed_acceptance(row_indices)
    frame["observed_loss"] = load_observed_loss_array("linear", row_indices=row_indices)

    selected: dict[str, list[str]] = {}
    excluded: dict[str, list[str]] = {}
    for target, outcome_columns in (
        ("acceptance", ["historical_u", "observed_acceptance"]),
        ("loss", ["observed_loss"]),
    ):
        numeric, excluded_features = _numeric_top_features(
            frame, importance, target=target, top_k=top_k
        )
        columns = [*numeric, *outcome_columns]
        correlation = frame[columns].corr(method="spearman")
        correlation.to_csv(analysis_dir / f"{target}_top_feature_spearman.csv")
        display_target = "claims" if target == "loss" else target
        _plot_correlation_matrix(
            correlation,
            title=f"Top {display_target} features: customer-level Spearman correlations",
            output_path=analysis_dir / f"{target}_top_feature_spearman.png",
        )
        selected[target] = numeric
        excluded[target] = excluded_features

    rank_correlations = _importance_rank_correlations(importance)
    rank_rows = []
    for target, matrix in rank_correlations.items():
        for model_a in matrix.index:
            for model_b in matrix.columns:
                rank_rows.append(
                    {
                        "target": target,
                        "model_a": model_a,
                        "model_b": model_b,
                        "spearman_rank_correlation": matrix.loc[model_a, model_b],
                    }
                )
    pd.DataFrame(rank_rows).to_csv(
        analysis_dir / "feature_importance_rank_correlations.csv", index=False
    )
    _plot_importance_rank_correlations(
        rank_correlations,
        analysis_dir / "feature_importance_rank_correlations.png",
    )

    metadata: dict[str, object] = {
        "analysis_dir": str(analysis_dir.resolve()),
        "sample_n_rows": int(len(row_indices)),
        "sample_seed": int(sample_seed),
        "top_k_per_model_target": int(top_k),
        "selected_numeric_features": selected,
        "excluded_categorical_or_code_features": excluded,
        "correlation": "Spearman",
    }
    (analysis_dir / "correlation_analysis.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    analysis_dir = args.analysis_dir or _latest_analysis_dir()
    metadata = render_correlations(
        analysis_dir,
        sample_n_rows=args.sample_n_rows,
        sample_seed=args.sample_seed,
        top_k=args.top_k,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
