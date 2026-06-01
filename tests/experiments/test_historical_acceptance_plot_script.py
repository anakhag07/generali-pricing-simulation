from __future__ import annotations

import numpy as np
import pytest

from scripts.plot_historical_acceptance import (
    _sample_indices,
    load_historical_acceptance_columns,
    plot_historical_acceptance_csv,
)


def _write_acceptance_csv(path) -> None:
    path.write_text(
        "idx;U;prob_acceptance\n"
        "0;-0.2;0.90\n"
        "1;-0.1;0.85\n"
        "2;0.0;0.80\n"
        "3;0.1;0.70\n"
        "4;0.2;0.60\n",
        encoding="utf-8",
    )


def test_load_historical_acceptance_columns_reads_u_and_acceptance(tmp_path) -> None:
    csv_path = tmp_path / "acceptance.csv"
    _write_acceptance_csv(csv_path)

    u_values, acceptance_values = load_historical_acceptance_columns(csv_path)

    np.testing.assert_allclose(u_values, [-0.2, -0.1, 0.0, 0.1, 0.2])
    np.testing.assert_allclose(acceptance_values, [0.90, 0.85, 0.80, 0.70, 0.60])


def test_plot_historical_acceptance_csv_writes_expected_png(tmp_path) -> None:
    csv_path = tmp_path / "acceptance.csv"
    output_dir = tmp_path / "plots"
    _write_acceptance_csv(csv_path)

    output_path = plot_historical_acceptance_csv(csv_path, output_dir, max_points=None)

    assert output_path == output_dir / "historical_u_acceptance_histogram.png"
    assert output_path.exists()


def test_sample_indices_requires_positive_max_points() -> None:
    with pytest.raises(ValueError, match="max_points must be positive"):
        _sample_indices(5, 0, seed=0)
