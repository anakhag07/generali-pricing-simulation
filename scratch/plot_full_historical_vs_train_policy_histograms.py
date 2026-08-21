"""Plot full historical and train-policy price-change distributions.

Historical pricing uses every canonical CSV row with a non-missing observed U.
Optimized pricing replays the saved train-fitted policy on its bound train rows.
All histograms use one-percentage-point bins and normalized density.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.dataset_metadata import OBSERVED_U_COL
from data.loader import dataset_csv_path
from experiments.policy_artifacts import load_policy_artifact


DEFAULT_RUN_DIR = Path(
    "/home/anakhag/projects/generali-pricing/results/"
    "glm-softmax-80-20-first-order/glm-softmax-80-20-first-order"
)
DEFAULT_POLICY_ARTIFACT = (
    DEFAULT_RUN_DIR
    / "seeds"
    / "seed-8"
    / "policies"
    / "first_order"
    / "policy.json"
)
DEFAULT_OUTPUT_DIR = (
    Path("/home/anakhag/projects/generali-pricing/results")
    / "policy-histogram-analysis"
    / "glm-softmax-80-20-first-order"
)

HISTORICAL_COLOR = "#7f7f7f"
OPTIMIZED_COLOR = "#9467bd"
BIN_WIDTH_PERCENTAGE_POINTS = 1.0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-artifact", type=Path, default=DEFAULT_POLICY_ARTIFACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional PNG preview directory; PDFs remain canonical.",
    )
    return parser.parse_args(argv)


def load_distributions(policy_artifact: Path) -> tuple[np.ndarray, np.ndarray, object]:
    historical_frame = pd.read_csv(
        dataset_csv_path(),
        sep=";",
        usecols=[OBSERVED_U_COL],
    )
    historical_u = historical_frame[OBSERVED_U_COL].dropna().to_numpy(dtype=float)

    artifact = load_policy_artifact(policy_artifact)
    if artifact.estimator != "first_order":
        raise ValueError(f"Expected first_order, got {artifact.estimator!r}.")
    if artifact.policy_head.type != "SoftmaxPolicy":
        raise ValueError(f"Expected SoftmaxPolicy, got {artifact.policy_head.type!r}.")
    if not np.allclose(
        (artifact.policy_head.action_low, artifact.policy_head.action_high),
        (-0.1, 0.2),
    ):
        raise ValueError("Expected optimized price bounds [-0.1, 0.2].")
    optimized_u = np.asarray(artifact.predict_u(split="train"), dtype=float).reshape(-1)

    for name, values in {
        "historical_u": historical_u,
        "optimized_u": optimized_u,
    }.items():
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError(f"{name} must be a non-empty finite array.")
    return historical_u, optimized_u, artifact


def percentage_point_bins(values_percent: np.ndarray) -> np.ndarray:
    lower = np.floor(float(np.min(values_percent)) + 1e-10)
    upper = np.ceil(float(np.max(values_percent)) - 1e-10)
    if np.isclose(lower, upper):
        upper = lower + BIN_WIDTH_PERCENTAGE_POINTS
    edges = np.arange(
        lower,
        upper + BIN_WIDTH_PERCENTAGE_POINTS,
        BIN_WIDTH_PERCENTAGE_POINTS,
        dtype=float,
    )
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def distribution_summary(
    label: str,
    population: str,
    values: np.ndarray,
) -> dict[str, object]:
    percent = 100.0 * np.asarray(values, dtype=float)
    quantiles = np.quantile(percent, [0.0, 0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "policy": label,
        "population": population,
        "n_customers": int(percent.size),
        "minimum_percent": float(np.min(percent)),
        "maximum_percent": float(np.max(percent)),
        "p0_percent": float(quantiles[0]),
        "p1_percent": float(quantiles[1]),
        "p25_percent": float(quantiles[2]),
        "median_p50_percent": float(quantiles[3]),
        "p75_percent": float(quantiles[4]),
        "p99_percent": float(quantiles[5]),
        "mean_percent": float(np.mean(percent)),
        "population_std_percent": float(np.std(percent, ddof=0)),
    }


def histogram_records(
    *,
    plot: str,
    series: str,
    values_percent: np.ndarray,
    bins: np.ndarray,
) -> list[dict[str, object]]:
    counts, edges = np.histogram(values_percent, bins=bins)
    widths = np.diff(edges)
    density = counts / (values_percent.size * widths)
    return [
        {
            "plot": plot,
            "series": series,
            "bin_left_percent": float(left),
            "bin_right_percent": float(right),
            "count": int(count),
            "density": float(bin_density),
        }
        for left, right, count, bin_density in zip(
            edges[:-1],
            edges[1:],
            counts,
            density,
            strict=True,
        )
    ]


def save_figure(
    fig: plt.Figure,
    *,
    pdf_path: Path,
    preview_path: Path | None,
) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf")
    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_path, dpi=160)
    plt.close(fig)


def plot_single_histogram(
    values_percent: np.ndarray,
    bins: np.ndarray,
    *,
    color: str,
    title: str,
    pdf_path: Path,
    preview_path: Path | None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.hist(
        values_percent,
        bins=bins,
        density=True,
        color=color,
        edgecolor=color,
        alpha=0.75,
        linewidth=0.5,
    )
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Price Change (%)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    save_figure(fig, pdf_path=pdf_path, preview_path=preview_path)


def plot_overlay_histogram(
    historical_percent: np.ndarray,
    optimized_percent: np.ndarray,
    bins: np.ndarray,
    *,
    pdf_path: Path,
    preview_path: Path | None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.hist(
        historical_percent,
        bins=bins,
        density=True,
        color=HISTORICAL_COLOR,
        edgecolor=HISTORICAL_COLOR,
        alpha=0.55,
        linewidth=0.5,
        label="Historical: full available population",
    )
    ax.hist(
        optimized_percent,
        bins=bins,
        density=True,
        color=OPTIMIZED_COLOR,
        edgecolor=OPTIMIZED_COLOR,
        alpha=0.65,
        linewidth=0.5,
        label="Optimized: training population",
    )
    ax.set_title("Historical and Optimized Price Changes", fontsize=16)
    ax.set_xlabel("Price Change (%)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    save_figure(fig, pdf_path=pdf_path, preview_path=preview_path)


def latex_percent(value: float) -> str:
    return f"{value:.2f}\\%"


def write_overleaf_itemize(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = ["\\begin{itemize}"]
    for summary in summaries:
        label = str(summary["policy"])
        population = str(summary["population"])
        n_customers = f"{int(summary['n_customers']):,}".replace(",", "{,}")
        lines.append(
            "  \\item \\textbf{" + label + "} "
            f"({population}; $n={n_customers}$): "
            f"minimum ${latex_percent(float(summary['minimum_percent']))}$; "
            f"maximum ${latex_percent(float(summary['maximum_percent']))}$; "
            f"$P_0={latex_percent(float(summary['p0_percent']))}$; "
            f"$P_1={latex_percent(float(summary['p1_percent']))}$; "
            f"$P_{{99}}={latex_percent(float(summary['p99_percent']))}$; "
            f"$P_{{25}}={latex_percent(float(summary['p25_percent']))}$; "
            f"$P_{{75}}={latex_percent(float(summary['p75_percent']))}$; and "
            f"median $P_{{50}}={latex_percent(float(summary['median_p50_percent']))}$."
        )
    lines.append("\\end{itemize}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    policy_path = args.policy_artifact.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    plots_dir = output_dir / "plots"
    preview_dir = args.preview_dir.expanduser().resolve() if args.preview_dir else None

    historical_u, optimized_u, artifact = load_distributions(policy_path)
    historical_percent = 100.0 * historical_u
    optimized_percent = 100.0 * optimized_u
    historical_bins = percentage_point_bins(historical_percent)
    optimized_bins = percentage_point_bins(optimized_percent)
    overlay_bins = historical_bins

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_single_histogram(
        historical_percent,
        historical_bins,
        color=HISTORICAL_COLOR,
        title="Historical Price Changes — Full Available Population",
        pdf_path=plots_dir / "historical_price_changes_full_population.pdf",
        preview_path=preview_dir / "historical_price_changes_full_population.png"
        if preview_dir
        else None,
    )
    plot_single_histogram(
        optimized_percent,
        optimized_bins,
        color=OPTIMIZED_COLOR,
        title="Optimized Price Changes — Training Population",
        pdf_path=plots_dir / "optimized_price_changes_train_population.pdf",
        preview_path=preview_dir / "optimized_price_changes_train_population.png"
        if preview_dir
        else None,
    )
    plot_overlay_histogram(
        historical_percent,
        optimized_percent,
        overlay_bins,
        pdf_path=plots_dir / "historical_full_vs_optimized_train_overlay.pdf",
        preview_path=preview_dir / "historical_full_vs_optimized_train_overlay.png"
        if preview_dir
        else None,
    )

    summaries = [
        distribution_summary(
            "Historical pricing",
            "full available historical population",
            historical_u,
        ),
        distribution_summary(
            "Optimized pricing",
            "train-fitted policy evaluated on training customers",
            optimized_u,
        ),
    ]
    pd.DataFrame(summaries).to_csv(output_dir / "pricing_summary_statistics.csv", index=False)
    write_overleaf_itemize(output_dir / "pricing_summary_statistics.tex", summaries)

    records = []
    records.extend(
        histogram_records(
            plot="historical_full",
            series="historical",
            values_percent=historical_percent,
            bins=historical_bins,
        )
    )
    records.extend(
        histogram_records(
            plot="optimized_train",
            series="optimized",
            values_percent=optimized_percent,
            bins=optimized_bins,
        )
    )
    records.extend(
        histogram_records(
            plot="overlay",
            series="historical",
            values_percent=historical_percent,
            bins=overlay_bins,
        )
    )
    records.extend(
        histogram_records(
            plot="overlay",
            series="optimized",
            values_percent=optimized_percent,
            bins=overlay_bins,
        )
    )
    pd.DataFrame(records).to_csv(output_dir / "histogram_bin_data.csv", index=False)

    config = {
        "policy_artifact": str(policy_path),
        "dataset": str(dataset_csv_path().resolve()),
        "historical_population": (
            "All canonical CSV rows with non-missing observed historical U."
        ),
        "optimized_population": "Saved train split bound to the train-fitted policy artifact.",
        "historical_n": int(historical_u.size),
        "optimized_train_n": int(optimized_u.size),
        "estimator": artifact.estimator,
        "bin_width_percentage_points": BIN_WIDTH_PERCENTAGE_POINTS,
        "histogram_normalization": "density",
        "colors": {
            "historical": HISTORICAL_COLOR,
            "optimized": OPTIMIZED_COLOR,
        },
    }
    (output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
