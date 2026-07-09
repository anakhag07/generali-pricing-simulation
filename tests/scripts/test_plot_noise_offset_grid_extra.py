"""Extra noise-offset grid plot script: family constants and seed aggregation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "plot_noise_offset_grid_extra.py"
_SPEC = importlib.util.spec_from_file_location("plot_noise_offset_grid_extra", _SCRIPT)
extra = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = extra
_SPEC.loader.exec_module(extra)


def test_family_axis_keys_and_projects() -> None:
    assert extra.HOMO.axis_key == "noise_std"
    assert extra.HETERO.axis_key == "noise_growth"
    assert extra.HOMO.project_name.endswith("-optax")
    assert extra.HETERO.project_name.endswith("-optax")
    assert set(extra.FAMILIES) == {"homoskedastic", "heteroskedastic", "all"}


def test_aggregate_means_and_stds_over_seeds() -> None:
    rows = [
        {"noise_std": "0.1", "theta_offset": "0.5", "estimator": "stein_difference",
         "run_seed": s, "theta_distance_to_truth": str(d), "clean_objective_gap": str(g)}
        for s, d, g in [(7, 1.0, 0.2), (8, 3.0, 0.4), (9, 2.0, 0.6)]
    ]
    stats = extra._aggregate(extra.HOMO, rows)
    entry = stats[(0.1, 0.5, "stein_difference")]
    assert abs(entry["theta_distance_to_truth_mean"] - 2.0) < 1e-12
    assert abs(entry["clean_objective_gap_mean"] - 0.4) < 1e-12
    # population std (ddof=0) of {1,2,3} is sqrt(2/3)
    assert abs(entry["theta_distance_to_truth_std"] - (2.0 / 3.0) ** 0.5) < 1e-12


def test_levels_are_sorted_and_estimator_scoped() -> None:
    rows = [
        {"noise_std": n, "theta_offset": o, "estimator": e, "run_seed": "7",
         "theta_distance_to_truth": "0.1", "clean_objective_gap": "0.01"}
        for n, o, e in [
            ("2.0", "5.0", "finite_difference"),
            ("0.0", "0.0", "finite_difference"),
            ("0.1", "1.0", "stein_difference"),
        ]
    ]
    stats = extra._aggregate(extra.HOMO, rows)
    noise, offsets = extra._levels(stats, "finite_difference")
    assert noise == [0.0, 2.0]
    assert offsets == [0.0, 5.0]
