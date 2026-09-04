"""Tests for the profit-support and optimized-price overlay."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts import plot_spline_support_cloud_with_optimized_prices as script


def _support_frame() -> pd.DataFrame:
    u = np.linspace(-0.1, 0.2, 301)
    mean = 100.0 + 50.0 * u
    width = 2.0 + np.square(u - 0.08)
    return pd.DataFrame(
        {
            "u": u,
            "smoothed_mean_profit": mean,
            "support_cloud_lower_profit": mean - width,
            "support_cloud_upper_profit": mean + width,
        }
    )


def _histogram_frame() -> pd.DataFrame:
    edges = np.linspace(-0.1, 0.2, 31)
    return pd.DataFrame(
        {
            "plot": "optimized",
            "series": "Optimized",
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": np.arange(1, 31),
            "density": np.linspace(1.0, 3.0, 30),
        }
    )


def _write_inputs(tmp_path):
    support_csv = tmp_path / "support.csv"
    histogram_csv = tmp_path / "histogram.csv"
    support_manifest = tmp_path / "support_manifest.json"
    histogram_config = tmp_path / "histogram_config.json"
    source_pdf = tmp_path / "source.pdf"
    _support_frame().to_csv(support_csv, index=False)
    _histogram_frame().to_csv(histogram_csv, index=False)
    support_manifest.write_text(
        json.dumps(
            {
                "model": {
                    "acceptance": "monotone_spline_xgb",
                    "loss": "xgb",
                },
                "sample": {"n_customers": 20_000, "seed": 20260831},
                "support": {"baseline_subtracted": False},
            }
        ),
        encoding="utf-8",
    )
    histogram_config.write_text(
        json.dumps(
            {
                "estimator": "first_order",
                "range": [-0.1, 0.2],
                "policy_artifact": "/results/policy.json",
            }
        ),
        encoding="utf-8",
    )
    source_pdf.write_bytes(b"%PDF-1.4\n")
    return (
        support_csv,
        support_manifest,
        histogram_csv,
        histogram_config,
        source_pdf,
    )


def test_loaders_require_wide_support_and_optimized_bins(tmp_path) -> None:
    support_csv, _, histogram_csv, _, _ = _write_inputs(tmp_path)

    support = script._load_support(support_csv)
    histogram = script._load_optimized_histogram(histogram_csv)

    assert len(support) == 301
    assert len(histogram) == 30
    assert np.isclose(histogram["bin_left"].iloc[0], -0.1)
    assert np.isclose(histogram["bin_right"].iloc[-1], 0.2)


def test_overlay_labels_blue_profit_and_pink_optimized_density(
    tmp_path,
    monkeypatch,
) -> None:
    captured = {}
    monkeypatch.setattr(
        script.plt,
        "close",
        lambda figure: captured.setdefault("figure", figure),
    )

    script._plot_overlay(
        _support_frame(),
        _histogram_frame(),
        tmp_path / "overlay.pdf",
    )

    figure = captured["figure"]
    profit_ax, density_ax = figure.axes
    assert profit_ax.get_title() == script.PLOT_TITLE
    assert profit_ax.get_xlabel() == "Price Change"
    assert profit_ax.get_ylabel() == "Mean Predicted Profit Per Customer"
    assert density_ax.get_ylabel() == "Optimized Price-Change Density"
    legend_text = profit_ax.get_legend().get_texts()
    assert [text.get_text() for text in legend_text] == [
        "Mean profit with local-support cloud",
        "Optimized price-change density",
    ]
    assert legend_text[0].get_color() == script.SUPPORT_COLOR
    assert legend_text[1].get_color() == script.OPTIMIZED_COLOR


def test_run_analysis_writes_vector_pdf_and_manifest(tmp_path) -> None:
    (
        support_csv,
        support_manifest,
        histogram_csv,
        histogram_config,
        source_pdf,
    ) = _write_inputs(tmp_path)
    output_dir = tmp_path / "output"
    args = script._build_parser().parse_args(
        [
            "--support-csv",
            str(support_csv),
            "--support-manifest",
            str(support_manifest),
            "--histogram-csv",
            str(histogram_csv),
            "--histogram-config",
            str(histogram_config),
            "--source-pdf",
            str(source_pdf),
            "--output-dir",
            str(output_dir),
        ]
    )

    pdf_path, manifest_path = script.run_analysis(args)

    assert pdf_path.read_bytes().startswith(b"%PDF-")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["display"]["blue"].startswith("monotone-spline/XGBoost")
    assert manifest["display"]["pink"].startswith("saved first-order GLM-policy")
    assert manifest["optimizer"].startswith("not rerun")
