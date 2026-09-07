"""Tests for the historical and saved-policy histogram overlay."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from scripts import plot_historical_with_lower_bound_policy as script


def _write_policy(tmp_path, n: int = 200):
    rows = np.arange(n, dtype=np.int64)
    actions = np.linspace(-0.09, 0.19, n)
    artifact = tmp_path / "policy.npz"
    summary = tmp_path / "summary.json"
    np.savez_compressed(artifact, row_indices=rows, actions=actions)
    summary.write_text(
        json.dumps(
            {
                "full_application": {
                    "n_customers": n,
                    "row_indices_sha256": hashlib.sha256(rows.tobytes()).hexdigest(),
                },
                "optimizer": {
                    "entry_point": "optimization.solvers.run_first_order_minimize",
                    "success": True,
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact, summary, actions


def test_histogram_uses_30_shared_decimal_bins() -> None:
    historical = np.linspace(-0.1, 0.2, 300)
    optimized = np.linspace(-0.09, 0.19, 200)

    frame = script._histogram_frame(historical, optimized)

    assert len(frame) == 60
    assert set(frame["series"]) == {"Historical", "Optimized"}
    assert np.isclose(frame["bin_left"].min(), -0.1)
    assert np.isclose(frame["bin_right"].max(), 0.2)


def test_run_replaces_optimized_series_and_preserves_labels(tmp_path, monkeypatch) -> None:
    artifact, summary, actions = _write_policy(tmp_path)
    reference = tmp_path / "reference.pdf"
    reference.write_bytes(b"%PDF-1.4\n")
    historical = np.linspace(-0.1, 0.2, len(actions))
    monkeypatch.setattr(script, "load_observed_u_array", lambda *_args, **_kwargs: historical)
    output_dir = tmp_path / "output"
    args = script._build_parser().parse_args(
        [
            "--policy-artifact",
            str(artifact),
            "--policy-summary",
            str(summary),
            "--reference-pdf",
            str(reference),
            "--output-dir",
            str(output_dir),
        ]
    )

    pdf_path, csv_path, manifest_path = script.run_analysis(args)

    assert pdf_path.read_bytes().startswith(b"%PDF-")
    histogram = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding=None)
    assert len(histogram) == 60
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["display"]["title"] == "Historical and Optimized Price Changes"
    assert manifest["display"]["x_axis"] == "Price Change"
    assert manifest["display"]["y_axis"] == "Density"
    assert manifest["display"]["optimized"]["color"] == "#86002d"
    assert manifest["population"]["n_customers"] == len(actions)
