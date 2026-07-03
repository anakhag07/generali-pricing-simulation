from __future__ import annotations

import json

import numpy as np

from experiments.sweep_utils import SweepResult, run_sweep

_FAST_OVERRIDES = {
    "n_samples": 8,
    "t_steps": 1,
    "enabled_estimators": ("first_order",),
    "plot": False,
    "verbose": False,
    "wandb_enabled": False,
}


def _resolved_seed_setup(summary_path) -> dict:
    payload = json.loads(summary_path.read_text())
    return payload["config"]["resolved_seed_setup"]


def test_run_sweep_seed_only_shares_one_variant_folder(tmp_path) -> None:
    result = run_sweep(
        base_preset="planted_logistic_base",
        run_seeds=(1, 2),
        override_list=[{"_run_name": "solo", **_FAST_OVERRIDES}],
        runs_root=str(tmp_path),
        project_name="seed-only",
    )

    assert isinstance(result, SweepResult)
    assert len(result.run_results) == 2
    assert {record.run_seed for record in result.run_results} == {1, 2}

    variant_dir = result.project_dir / "solo"
    summaries = sorted(p.name for p in variant_dir.glob("summary-seed-*.json"))
    assert summaries == ["summary-seed-1.json", "summary-seed-2.json"]
    seed_dirs = sorted(p.name for p in (variant_dir / "seeds").glob("seed-*"))
    assert seed_dirs == ["seed-1", "seed-2"]
    assert (variant_dir / "seed_grid_summary.csv").exists()


def test_run_sweep_axis_times_seeds_yields_full_grid(tmp_path) -> None:
    result = run_sweep(
        base_preset="planted_logistic_base",
        run_seeds=(1, 2, 3),
        override_list=[
            {"_run_name": "a", **_FAST_OVERRIDES},
            {"_run_name": "b", "step_size": 0.002, **_FAST_OVERRIDES},
        ],
        runs_root=str(tmp_path),
        project_name="grid",
    )

    assert len(result.run_results) == 6  # 2 variants x 3 seeds
    variants = {record.run_name for record in result.run_results}
    assert variants == {"a", "b"}
    # cross-variant aggregate written at project root
    assert (result.project_dir / "seed_grid_summary.csv").exists()


def test_run_sweep_default_fixes_data_split_noise_and_varies_theta(tmp_path) -> None:
    result = run_sweep(
        base_preset="planted_logistic_base",
        run_seeds=(1, 2),
        override_list=[{"_run_name": "v", **_FAST_OVERRIDES}],
        runs_root=str(tmp_path),
        project_name="fixed-data",
    )

    variant_dir = result.project_dir / "v"
    seed1 = _resolved_seed_setup(variant_dir / "summary-seed-1.json")
    seed2 = _resolved_seed_setup(variant_dir / "summary-seed-2.json")

    assert seed1["data_seed"] == seed2["data_seed"]
    assert seed1["split_seed"] == seed2["split_seed"]
    assert seed1["noise_seed"] == seed2["noise_seed"]
    assert seed1["theta_seed"] != seed2["theta_seed"]


def test_run_sweep_theta_variation_gives_non_degenerate_error_bars(tmp_path) -> None:
    # With random theta0 (theta0=None), varying theta across seeds must move the
    # deterministic first-order result -- otherwise error bars would be degenerate.
    result = run_sweep(
        base_preset="planted_logistic_base",
        run_seeds=(1, 2),
        override_list=[{"_run_name": "v", "theta0": None, **_FAST_OVERRIDES}],
        vary=("theta",),
        runs_root=str(tmp_path),
        project_name="non-degenerate",
    )

    by_seed = {record.run_seed: record for record in result.run_results}
    theta_1 = by_seed[1].result.results["first_order"].theta
    theta_2 = by_seed[2].result.results["first_order"].theta
    assert not np.allclose(theta_1, theta_2)
