"""Plot historical and train-policy price changes on the decimal [-0.1, 0.2] range."""

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

from data.loader import dataset_csv_path
from scratch.plot_full_historical_vs_train_policy_histograms import (
    DEFAULT_OUTPUT_DIR as BASE_OUTPUT_DIR,
    DEFAULT_POLICY_ARTIFACT,
    load_distributions,
)


DEFAULT_OUTPUT_DIR = BASE_OUTPUT_DIR / "decimal-range-minus0.1-to-0.2"
RANGE_LOW = -0.1
RANGE_HIGH = 0.2
BIN_WIDTH = 0.01

# Exact xcolor conversions:
# purple!70!black = 70% xcolor purple (0.75, 0, 0.25) + 30% black.
# black!55 = 55% black + 45% white; black!60 = 60% black + 40% white.
OPTIMIZED_COLOR = "#86002d"
HISTORICAL_COLOR = "#737373"
HISTORICAL_TEXT_COLOR = "#666666"


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


def bounded(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    tolerance = 1e-12
    selected = array[
        (array >= RANGE_LOW - tolerance) & (array <= RANGE_HIGH + tolerance)
    ]
    return np.clip(selected, RANGE_LOW, RANGE_HIGH)


def fixed_bins() -> np.ndarray:
    n_bins = int(round((RANGE_HIGH - RANGE_LOW) / BIN_WIDTH))
    edges = np.linspace(RANGE_LOW, RANGE_HIGH, n_bins + 1, dtype=float)
    edges[-1] = np.nextafter(RANGE_HIGH, np.inf)
    return edges


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


def plot_single(
    values: np.ndarray,
    *,
    color: str,
    title: str,
    pdf_path: Path,
    preview_path: Path | None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.hist(
        values,
        bins=fixed_bins(),
        density=True,
        color=color,
        edgecolor=color,
        linewidth=0.5,
        alpha=0.75,
    )
    ax.set_xlim(RANGE_LOW, RANGE_HIGH)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Price Change", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    save_figure(fig, pdf_path=pdf_path, preview_path=preview_path)


def plot_overlay(
    historical: np.ndarray,
    optimized: np.ndarray,
    *,
    pdf_path: Path,
    preview_path: Path | None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    ax.hist(
        historical,
        bins=fixed_bins(),
        density=True,
        color=HISTORICAL_COLOR,
        edgecolor=HISTORICAL_COLOR,
        linewidth=0.5,
        alpha=0.55,
        label="Historical",
    )
    ax.hist(
        optimized,
        bins=fixed_bins(),
        density=True,
        color=OPTIMIZED_COLOR,
        edgecolor=OPTIMIZED_COLOR,
        linewidth=0.5,
        alpha=0.68,
        label="Optimized",
    )
    ax.set_xlim(RANGE_LOW, RANGE_HIGH)
    ax.set_title("Historical and Optimized Price Changes", fontsize=16)
    ax.set_xlabel("Price Change", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.tick_params(labelsize=10)
    legend = ax.legend(fontsize=10)
    legend_texts = legend.get_texts()
    if len(legend_texts) == 2:
        legend_texts[0].set_color(HISTORICAL_TEXT_COLOR)
        legend_texts[1].set_color(OPTIMIZED_COLOR)
    ax.grid(alpha=0.25)
    save_figure(fig, pdf_path=pdf_path, preview_path=preview_path)


def decimal_summary(
    label: str,
    population: str,
    values: np.ndarray,
) -> dict[str, object]:
    array = np.asarray(values, dtype=float).reshape(-1)
    quantiles = np.quantile(array, [0.0, 0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "policy": label,
        "population": population,
        "n_customers": int(array.size),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "p0": float(quantiles[0]),
        "p1": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median_p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p99": float(quantiles[5]),
        "mean": float(np.mean(array)),
        "population_std": float(np.std(array, ddof=0)),
    }


def latex_decimal(value: float) -> str:
    return f"{value:.4f}"


def write_itemize(path: Path, summaries: list[dict[str, object]]) -> None:
    lines = ["\\begin{itemize}"]
    for summary in summaries:
        n_customers = f"{int(summary['n_customers']):,}".replace(",", "{,}")
        lines.append(
            "  \\item \\textbf{" + str(summary["policy"]) + "} "
            f"({summary['population']}; $n={n_customers}$): "
            f"minimum ${latex_decimal(float(summary['minimum']))}$; "
            f"maximum ${latex_decimal(float(summary['maximum']))}$; "
            f"$P_0={latex_decimal(float(summary['p0']))}$; "
            f"$P_1={latex_decimal(float(summary['p1']))}$; "
            f"$P_{{99}}={latex_decimal(float(summary['p99']))}$; "
            f"$P_{{25}}={latex_decimal(float(summary['p25']))}$; "
            f"$P_{{75}}={latex_decimal(float(summary['p75']))}$; and "
            f"median $P_{{50}}={latex_decimal(float(summary['median_p50']))}$."
        )
    lines.append("\\end{itemize}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def histogram_records(
    plot: str,
    series: str,
    values: np.ndarray,
) -> list[dict[str, object]]:
    edges = fixed_bins()
    counts, _ = np.histogram(values, bins=edges)
    density = counts / (values.size * np.diff(edges))
    return [
        {
            "plot": plot,
            "series": series,
            "bin_left": float(left),
            "bin_right": float(right),
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    policy_path = args.policy_artifact.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    plots_dir = output_dir / "plots"
    preview_dir = args.preview_dir.expanduser().resolve() if args.preview_dir else None

    historical_full, optimized_train, artifact = load_distributions(policy_path)
    historical_range = bounded(historical_full)
    optimized_range = bounded(optimized_train)

    plot_single(
        historical_range,
        color=HISTORICAL_COLOR,
        title="Historical Price Changes",
        pdf_path=plots_dir / "historical_price_changes_decimal_minus0.1_to_0.2.pdf",
        preview_path=preview_dir / "historical_price_changes_decimal_minus0.1_to_0.2.png"
        if preview_dir
        else None,
    )
    plot_single(
        optimized_range,
        color=OPTIMIZED_COLOR,
        title="Optimized Price Changes",
        pdf_path=plots_dir / "optimized_price_changes_decimal_minus0.1_to_0.2.pdf",
        preview_path=preview_dir / "optimized_price_changes_decimal_minus0.1_to_0.2.png"
        if preview_dir
        else None,
    )
    plot_overlay(
        historical_range,
        optimized_range,
        pdf_path=plots_dir / "historical_vs_optimized_decimal_minus0.1_to_0.2_overlay.pdf",
        preview_path=preview_dir
        / "historical_vs_optimized_decimal_minus0.1_to_0.2_overlay.png"
        if preview_dir
        else None,
    )

    full_summaries = [
        decimal_summary(
            "Historical pricing",
            "full available historical population",
            historical_full,
        ),
        decimal_summary(
            "Optimized pricing",
            "train-fitted policy evaluated on training customers",
            optimized_train,
        ),
    ]
    range_summaries = [
        decimal_summary(
            "Historical pricing",
            "full historical population restricted to $[-0.1,0.2]$",
            historical_range,
        ),
        decimal_summary(
            "Optimized pricing",
            "training population restricted to $[-0.1,0.2]$",
            optimized_range,
        ),
    ]
    base_dir = output_dir.parent
    pd.DataFrame(full_summaries).to_csv(
        base_dir / "pricing_summary_statistics_decimal.csv",
        index=False,
    )
    write_itemize(base_dir / "pricing_summary_statistics.tex", full_summaries)
    pd.DataFrame(range_summaries).to_csv(
        output_dir / "pricing_summary_statistics_decimal_bounded.csv",
        index=False,
    )
    write_itemize(
        output_dir / "pricing_summary_statistics_decimal_bounded.tex",
        range_summaries,
    )

    records: list[dict[str, object]] = []
    records.extend(histogram_records("historical", "Historical", historical_range))
    records.extend(histogram_records("optimized", "Optimized", optimized_range))
    records.extend(histogram_records("overlay", "Historical", historical_range))
    records.extend(histogram_records("overlay", "Optimized", optimized_range))
    pd.DataFrame(records).to_csv(output_dir / "histogram_bin_data_decimal.csv", index=False)

    config = {
        "policy_artifact": str(policy_path),
        "dataset": str(dataset_csv_path().resolve()),
        "estimator": artifact.estimator,
        "units": "raw decimal price change U",
        "range": [RANGE_LOW, RANGE_HIGH],
        "range_semantics": (
            "Rows outside the requested range are excluded before density normalization."
        ),
        "bin_width": BIN_WIDTH,
        "historical_full_n": int(historical_full.size),
        "historical_range_n": int(historical_range.size),
        "optimized_train_full_n": int(optimized_train.size),
        "optimized_train_range_n": int(optimized_range.size),
        "colors": {
            "historical_draw_black_55": HISTORICAL_COLOR,
            "historical_text_black_60": HISTORICAL_TEXT_COLOR,
            "optimized_draw_and_text_purple_70_black": OPTIMIZED_COLOR,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(pd.DataFrame(full_summaries).to_string(index=False))
    print(pd.DataFrame(range_summaries).to_string(index=False))
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
