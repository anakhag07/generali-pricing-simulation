from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import query_acceptance_at_u
from scripts.query_acceptance_at_u import (
    _resolve_preset_and_model_type,
    _resolve_u_values,
    query_mean_acceptance,
)


class _MeanAcceptanceObjective:
    def mean_acceptance_at_u(self, x_batch: np.ndarray, u: float) -> float:
        return float(np.mean(x_batch[:, 0]) + u)


def _make_config(x_fixed: np.ndarray | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        objective=_MeanAcceptanceObjective(),
        x_fixed=x_fixed,
        seed=7,
        n_samples=4,
        state_dim=2,
    )


def test_query_mean_acceptance_uses_first_n_fixed_rows() -> None:
    config = _make_config(np.array([[0.0], [2.0], [4.0]], dtype=float))

    rows = query_mean_acceptance(config, [-0.5, 0.0], n_rows=2)

    assert [row.n for row in rows] == [2, 2]
    assert rows[0].u == pytest.approx(-0.5)
    assert rows[0].mean_acceptance == pytest.approx(0.5)
    assert rows[1].mean_acceptance == pytest.approx(1.0)


def test_query_mean_acceptance_requires_supported_objective() -> None:
    config = SimpleNamespace(
        objective=object(),
        x_fixed=np.zeros((2, 1), dtype=float),
        seed=7,
        n_samples=2,
        state_dim=1,
    )

    with pytest.raises(ValueError, match="mean_acceptance_at_u"):
        query_mean_acceptance(config, [0.0])


def test_resolve_u_values_from_count_uses_default_bounds() -> None:
    values = _resolve_u_values(None, 5, -0.5, 0.5)

    np.testing.assert_allclose(values, [-0.5, -0.25, 0.0, 0.25, 0.5])


def test_resolve_u_values_requires_one_source() -> None:
    with pytest.raises(ValueError, match="either explicit --u values or --u-count"):
        _resolve_u_values([0.0], 5, -0.5, 0.5)

    with pytest.raises(ValueError, match="either explicit --u values or --u-count"):
        _resolve_u_values(None, None, -0.5, 0.5)


def test_model_type_maps_to_default_preset() -> None:
    preset, model_type = _resolve_preset_and_model_type(None, "xgb")

    assert preset == "real_data_xgb_base"
    assert model_type == "xgb"


def test_main_writes_optional_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    config = _make_config(np.array([[0.0], [2.0]], dtype=float))
    monkeypatch.setattr(query_acceptance_at_u, "get_config", lambda preset: config)
    monkeypatch.setattr(
        query_acceptance_at_u,
        "load_observed_u_array",
        lambda model_type, n_rows: np.asarray([0.1, 0.2], dtype=float),
    )
    csv_path = tmp_path / "acceptance.csv"
    output_root = tmp_path / "plots"

    query_acceptance_at_u.main(
        [
            "--model-type",
            "glm",
            "--u",
            "0.0",
            "0.5",
            "--csv",
            str(csv_path),
            "--output-root",
            str(output_root),
        ]
    )

    output = capsys.readouterr().out
    assert "mean_acceptance" in output
    assert "constant_u_histograms.png" in output
    assert "constant_u_acceptance_curve.png" in output
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "u,n,mean_acceptance",
        "0.0,2,1.0",
        "0.5,2,1.5",
    ]
    assert (output_root / "glm" / "constant_u_histograms.png").exists()
    assert (output_root / "glm" / "constant_u_acceptance_curve.png").exists()


def test_main_uses_custom_output_subdir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = _make_config(np.array([[0.0], [2.0]], dtype=float))
    monkeypatch.setattr(query_acceptance_at_u, "get_config", lambda preset: config)
    monkeypatch.setattr(
        query_acceptance_at_u,
        "load_observed_u_array",
        lambda model_type, n_rows: np.asarray([0.1, 0.2], dtype=float),
    )
    output_root = tmp_path / "acceptance_queries"

    query_acceptance_at_u.main(
        [
            "--model-type",
            "xgb",
            "--u-count",
            "3",
            "--output-root",
            str(output_root),
            "--output-subdir",
            "custom_xgb",
        ]
    )

    assert (output_root / "custom_xgb" / "constant_u_histograms.png").exists()
    assert (output_root / "custom_xgb" / "constant_u_acceptance_curve.png").exists()
