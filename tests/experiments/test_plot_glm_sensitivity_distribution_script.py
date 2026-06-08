"""Tests for the GLM sensitivity distribution plotting script."""

from __future__ import annotations

import numpy as np
import pytest

from scripts import plot_glm_sensitivity_distribution as script


def test_resolve_u_grid_defaults_range() -> None:
    values = script._resolve_u_grid(-0.3, 0.3, 7)

    np.testing.assert_allclose(values, [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])


def test_resolve_u_grid_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="u_min"):
        script._resolve_u_grid(0.3, -0.3, 7)
    with pytest.raises(ValueError, match="positive"):
        script._resolve_u_grid(-0.3, 0.3, 0)
    with pytest.raises(ValueError, match="finite"):
        script._resolve_u_grid(float("nan"), 0.3, 7)


def test_summary_rows_reports_per_u_customer_summaries() -> None:
    matrix = np.array(
        [
            [0.1, 0.2],
            [0.2, 0.4],
            [0.3, 0.6],
        ],
        dtype=float,
    )

    rows = script._summary_rows([-0.1, 0.1], matrix)

    assert rows[0]["u"] == -0.1
    assert rows[0]["n_rows"] == 3
    assert rows[0]["mean"] == pytest.approx(0.2)
    assert rows[0]["median"] == pytest.approx(0.2)
    assert rows[1]["mean"] == pytest.approx(0.4)


def test_main_writes_summary_csvs_and_plots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    def fake_sensitivity_matrix(acceptance_model, x_frame, *, u_values, u_coef=None):
        u_arr = np.asarray(u_values, dtype=float).reshape(-1)
        customer_scale = np.array([0.1, 0.2, 0.3], dtype=float).reshape(-1, 1)
        return customer_scale + np.square(u_arr).reshape(1, -1)

    monkeypatch.setattr(
        script,
        "_resolve_row_indices",
        lambda n_rows, seed: np.array([10, 20, 30]),
    )
    monkeypatch.setattr(
        script,
        "load_model_artifacts",
        lambda model_type: (object(), object()),
    )
    monkeypatch.setattr(script, "load_x_frame", lambda model_type, row_indices: object())
    monkeypatch.setattr(script, "glm_price_sensitivity_matrix", fake_sensitivity_matrix)

    script.main(
        [
            "--u-min",
            "-0.1",
            "--u-max",
            "0.1",
            "--u-count",
            "3",
            "--hist-u",
            "-0.1",
            "0.0",
            "0.1",
            "--bins",
            "5",
            "--output-root",
            str(tmp_path),
            "--output-subdir",
            "run",
        ]
    )

    output = capsys.readouterr().out
    output_dir = tmp_path / "run"
    assert "Peak average sensitivity" in output
    assert (output_dir / "glm_sensitivity_by_u.csv").exists()
    assert (output_dir / "glm_selected_u_sensitivity_summary.csv").exists()
    assert (output_dir / "mean_sensitivity_by_u.png").exists()
    assert (output_dir / "sensitivity_histograms_by_u.png").exists()
    header = (
        output_dir.joinpath("glm_sensitivity_by_u.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert header == "u,n_rows,mean,median,q05,q25,q75,q95,min,max"
