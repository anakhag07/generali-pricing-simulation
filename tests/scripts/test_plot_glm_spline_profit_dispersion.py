"""Tests for the GLM-versus-spline profit-dispersion analysis script."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from reporting.profit_dispersion import ProfitDispersion
from scripts import plot_glm_spline_profit_dispersion as script


def _summaries(u_grid: np.ndarray) -> dict[str, ProfitDispersion]:
    offset = np.arange(u_grid.size, dtype=float)
    return {
        "glm": ProfitDispersion(
            mean=100.0 + offset,
            std=10.0 + 0.1 * offset,
            median=90.0 + offset,
            mad=5.0 + 0.1 * offset,
        ),
        "spline": ProfitDispersion(
            mean=110.0 + offset,
            std=12.0 + 0.1 * offset,
            median=95.0 + offset,
            mad=6.0 + 0.1 * offset,
        ),
    }


def test_parser_uses_agreed_sample_defaults() -> None:
    args = script._build_parser().parse_args([])

    assert args.sample_size == 20_000
    assert args.seed == 20260831
    assert args.u_min == -0.10
    assert args.u_max == 0.20
    assert args.u_count == 301
    np.testing.assert_allclose(
        script._resolve_u_grid(args.u_min, args.u_max, args.u_count),
        np.linspace(-0.10, 0.20, 301),
    )


def test_load_canonical_model_pairs_uses_glm_and_xgb_loss(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_load(acceptance: str, risk: str):
        calls.append((acceptance, risk))
        return (acceptance, risk)

    monkeypatch.setattr(script, "load_model_artifact_pair", fake_load)

    glm_pair, spline_source_pair = script._load_canonical_model_pairs()

    assert calls == [("linear", "linear"), ("xgb", "xgb")]
    assert glm_pair == ("linear", "linear")
    assert spline_source_pair == ("xgb", "xgb")


def test_curve_rows_have_long_form_schema_and_unclipped_bands() -> None:
    u_grid = np.linspace(-0.1, 0.2, 4)
    rows = script._curve_rows(_summaries(u_grid), u_grid, n_customers=20_000)

    assert len(rows) == 2 * 2 * 4
    assert list(rows[0]) == [
        "model_family",
        "acceptance_model",
        "loss_model",
        "u",
        "center_statistic",
        "center",
        "dispersion_statistic",
        "dispersion",
        "lower",
        "upper",
        "n_customers",
    ]
    first = rows[0]
    assert first["model_family"] == "glm"
    assert first["acceptance_model"] == "linear"
    assert first["loss_model"] == "linear"
    assert first["center_statistic"] == "mean"
    assert first["dispersion_statistic"] == "std"
    assert first["lower"] == 90.0
    assert first["upper"] == 110.0


def test_plot_comparison_writes_vector_pdf(tmp_path) -> None:
    output = tmp_path / "curve.pdf"
    u_grid = np.linspace(-0.1, 0.2, 4)

    script._plot_comparison(
        _summaries(u_grid),
        u_grid,
        center_attr="median",
        dispersion_attr="mad",
        title="Median profit",
        band_label="1 MAD",
        output_path=output,
    )

    assert output.read_bytes().startswith(b"%PDF-")


def test_manifest_records_formula_models_and_generated_files(
    monkeypatch,
    tmp_path,
) -> None:
    artifacts = {}
    for name in ("dataset", "glm_acceptance", "glm_loss", "xgb_acceptance", "xgb_loss"):
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        artifacts[name] = path
    monkeypatch.setattr(script, "DATASET_PATH", artifacts["dataset"])
    monkeypatch.setitem(script.ACCEPTANCE_MODEL_ARTIFACTS["linear"], "path", artifacts["glm_acceptance"])
    monkeypatch.setitem(script.LOSS_MODEL_ARTIFACTS["linear"], "path", artifacts["glm_loss"])
    monkeypatch.setitem(script.ACCEPTANCE_MODEL_ARTIFACTS["xgb"], "path", artifacts["xgb_acceptance"])
    monkeypatch.setitem(script.LOSS_MODEL_ARTIFACTS["xgb"], "path", artifacts["xgb_loss"])
    generated = tmp_path / "curve.csv"
    generated.write_text("u,profit\n0,1\n", encoding="utf-8")
    diagnostics = tmp_path / "diagnostics.npz"
    diagnostics.write_bytes(b"diagnostics")
    args = SimpleNamespace(diagnostics=diagnostics, seed=20260831)
    manifest_path = tmp_path / "manifest.json"

    script._write_manifest(
        manifest_path,
        args=args,
        eligible_rows=np.arange(10),
        sample_rows=np.arange(3),
        spline_weights=np.ones(17),
        u_grid=np.linspace(-0.1, 0.2, 301),
        generated_files=[generated],
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model_pairs"]["spline"]["loss"] == "xgb"
    assert manifest["spline"]["raw_xgboost_fallback"] is False
    assert manifest["spline"]["boundary_behavior"]["below_0"].startswith("constant")
    assert manifest["objective"]["class"] == "ModelBasedObjective"
    assert manifest["objective"]["plot_transform"] == "profit_i(u) = -objective_i(u)"
    assert manifest["statistics"]["mad_scale_factor"] == 1.0
    assert manifest["statistics"]["smoothing"] == "none"
    assert manifest["optimizer"].startswith("not used")
    assert "curve.csv" in manifest["generated_files"]


def test_run_analysis_writes_csv_two_pdfs_and_manifest(monkeypatch, tmp_path) -> None:
    sample_rows = np.array([1, 2, 3])
    frame = pd.DataFrame({"X_policy_premium": [100.0, 200.0, 300.0]})
    artifact = SimpleNamespace(model=SimpleNamespace())
    monkeypatch.setattr(script, "eligible_csv_row_indices", lambda family: np.arange(10))
    monkeypatch.setattr(script, "load_deterministic_sample_rows", lambda *args, **kwargs: sample_rows)
    monkeypatch.setattr(script, "load_x_frame", lambda *args, **kwargs: frame)
    monkeypatch.setattr(script, "_spline_weights", lambda eligible: np.ones(17))
    monkeypatch.setattr(
        script,
        "_load_canonical_model_pairs",
        lambda: ((artifact, artifact), (artifact, artifact)),
    )
    monkeypatch.setattr(
        script,
        "exact_spline_acceptance_matrix",
        lambda artifact, frame, u_grid, *args, **kwargs: np.full((3, len(u_grid)), 0.4),
    )
    monkeypatch.setattr(
        script,
        "model_based_objective_matrix",
        lambda objective, frame, u_grid, **kwargs: -np.tile(
            np.linspace(1.0, 2.0, len(u_grid)),
            (len(frame), 1),
        ),
    )

    def fake_manifest(path: Path, **kwargs) -> None:
        path.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(script, "_write_manifest", fake_manifest)
    args = script._build_parser().parse_args(
        [
            "--diagnostics",
            str(tmp_path / "diagnostics.npz"),
            "--output-dir",
            str(tmp_path / "output"),
            "--sample-size",
            "3",
            "--n-jobs",
            "1",
        ]
    )

    outputs = script.run_analysis(args)

    assert len(outputs) == 4
    assert all(path.exists() for path in outputs)
    assert outputs[1].read_bytes().startswith(b"%PDF-")
    assert outputs[2].read_bytes().startswith(b"%PDF-")
    frame_out = pd.read_csv(outputs[0])
    assert len(frame_out) == 1204
    assert set(frame_out["center_statistic"]) == {"mean", "median"}
    assert set(frame_out["dispersion_statistic"]) == {"std", "mad"}
    assert set(frame_out["loss_model"]) == {"linear", "xgb"}
