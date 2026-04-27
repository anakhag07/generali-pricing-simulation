from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import query_acceptance_at_u
from scripts.query_acceptance_at_u import query_mean_acceptance


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


def test_main_writes_optional_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys,
) -> None:
    config = _make_config(np.array([[0.0], [2.0]], dtype=float))
    monkeypatch.setattr(query_acceptance_at_u, "get_config", lambda preset: config)
    csv_path = tmp_path / "acceptance.csv"

    query_acceptance_at_u.main(
        [
            "--preset",
            "fake_preset",
            "--u",
            "0.0",
            "0.5",
            "--csv",
            str(csv_path),
        ]
    )

    output = capsys.readouterr().out
    assert "mean_acceptance" in output
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "u,n,mean_acceptance",
        "0.0,2,1.0",
        "0.5,2,1.5",
    ]
