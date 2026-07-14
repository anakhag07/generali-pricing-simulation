from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from objective.noise import HomoskedasticGaussianNoise, NoisyObjective
from objective.objectives import QuadraticObjective
from scratch import run_quadratic_homoskedastic_sweep as script


def test_full_and_pilot_grid_sizes() -> None:
    full = script._parse_args([])
    pilot = script._parse_args(["--pilot"])

    full_noise, full_radii, full_seeds = script._resolved_grid(full)
    pilot_noise, pilot_radii, pilot_seeds = script._resolved_grid(pilot)

    assert len(full_noise) * len(full_radii) * len(full_seeds) == 640
    assert len(pilot_noise) * len(pilot_radii) * len(pilot_seeds) == 60
    assert script._project_name(full) == script.PROJECT_NAME
    assert script._project_name(pilot) == script.PILOT_PROJECT_NAME


def test_variant_name_round_trip() -> None:
    name = script._variant_name(1e-6, 1e-2)

    assert name == "noise-std-1e-06__fd-radius-0.01"
    assert script._parse_variant(name) == (1e-6, 1e-2)
    assert script._parse_variant("fd-radius-0.01") is None


def test_override_list_covers_noise_by_radius_product() -> None:
    overrides = script._build_override_list(
        dimension=4,
        noise_stds=(0.0, 1e-4),
        fd_radii=(1e-2, 1e-1),
        t_steps=25,
    )

    assert len(overrides) == 4
    assert [item["_run_name"] for item in overrides] == [
        "noise-std-0__fd-radius-0.01",
        "noise-std-0__fd-radius-0.1",
        "noise-std-0.0001__fd-radius-0.01",
        "noise-std-0.0001__fd-radius-0.1",
    ]
    for item in overrides:
        objective = item["objective"]
        assert isinstance(objective, NoisyObjective)
        assert isinstance(objective.base_objective, QuadraticObjective)
        assert isinstance(objective.noise, HomoskedasticGaussianNoise)
        assert item["enabled_estimators"] == (script.ESTIMATOR,)
        assert item["perturbation_space"] == "theta"
        assert item["step_rule"] == "l-bfgs-b"
        assert np.linalg.norm(item["theta0"]) == pytest.approx(1.0)


def test_run_grid_varies_only_noise_seed(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    def fake_run_sweep(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(project_dir=tmp_path / "project")

    monkeypatch.setattr(script, "run_sweep", fake_run_sweep)

    project_dir = script._run_grid(
        project_name="project",
        dimension=2,
        noise_stds=(0.0,),
        fd_radii=(0.1,),
        run_seeds=(7, 8),
        t_steps=10,
    )

    assert project_dir == tmp_path / "project"
    assert calls[0]["base_preset"] == script.BASE_PRESET
    assert calls[0]["run_seeds"] == (7, 8)
    assert calls[0]["vary"] == ("noise",)
    assert calls[0]["anchor_seed"] == 7
    assert calls[0]["per_seed_plots"] is False


def test_summary_row_separates_clean_and_noisy_values(tmp_path) -> None:
    summary_path = tmp_path / "summary-seed-7.json"
    summary = _summary_payload(success=False)

    row = script._summary_row(
        summary,
        summary_path,
        noise_std=1e-4,
        fd_radius=1e-2,
    )

    assert row["noise_to_radius"] == pytest.approx(0.01)
    assert row["final_theta_norm"] == pytest.approx(0.5)
    assert row["clean_final_objective"] == pytest.approx(0.125)
    assert row["noisy_final_objective"] == pytest.approx(0.1)
    assert row["exploitation_gap"] == pytest.approx(-0.025)
    assert row["clean_improvement"] == pytest.approx(0.375)
    assert row["optimizer_success"] is False


def test_aggregation_includes_optimizer_failures_and_writes_csvs(tmp_path) -> None:
    rows = [
        script._summary_row(
            _summary_payload(success=success, clean=clean, noisy=noisy, run_seed=seed),
            tmp_path / f"summary-seed-{seed}.json",
            noise_std=1e-4,
            fd_radius=1e-2,
        )
        for success, clean, noisy, seed in (
            (True, 0.1, 0.08, 7),
            (False, 0.2, 0.15, 8),
        )
    ]

    summaries = script._aggregate_rows(rows)

    assert len(summaries) == 1
    assert summaries[0]["n_seeds"] == 2
    assert summaries[0]["optimizer_success_rate"] == pytest.approx(0.5)
    assert summaries[0]["clean_final_objective_median"] == pytest.approx(0.15)

    script._write_outputs(tmp_path, rows, plot=False)
    assert (tmp_path / "quadratic_homoskedastic_finals.csv").exists()
    assert (tmp_path / "quadratic_homoskedastic_summary.csv").exists()


def test_collect_rows_reads_saved_variant_summaries(tmp_path) -> None:
    variant_dir = tmp_path / "noise-std-0.0001__fd-radius-0.01"
    variant_dir.mkdir()
    path = variant_dir / "summary-seed-7.json"
    path.write_text(json.dumps(_summary_payload()), encoding="utf-8")

    rows = script._collect_rows(tmp_path)

    assert len(rows) == 1
    assert rows[0]["run_seed"] == 7
    assert rows[0]["summary_path"] == str(path)


def _summary_payload(
    *,
    success: bool = True,
    clean: float = 0.125,
    noisy: float = 0.1,
    run_seed: int = 7,
) -> dict:
    theta_norm = np.sqrt(2.0 * clean)
    return {
        "run": {"run_dir": "/tmp/run"},
        "config": {
            "resolved_seed_setup": {"run_seed": run_seed},
            "objective": {
                "type": "NoisyObjective",
                "base_objective": {"type": "QuadraticObjective", "dimension": 2},
            },
        },
        "initial_value": 0.5,
        "estimators": {
            script.ESTIMATOR: {
                "theta": [theta_norm, 0.0],
                "final_value": clean,
                "runtime_sec": 0.01,
                "optimizer_success": success,
                "optimizer_status": 0 if success else 2,
                "optimizer_message": "ok" if success else "abnormal",
            }
        },
        "trace_summary": {
            script.ESTIMATOR: {
                "steps": 4,
                "final_objective": noisy,
            }
        },
    }
