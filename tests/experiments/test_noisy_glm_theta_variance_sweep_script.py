from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import run_noisy_glm_theta_variance_sweep as script


def test_truth_reference_loads_theta_and_final_mean_u(tmp_path) -> None:
    truth_path = _write_truth_summary(tmp_path)

    reference = script._load_truth_reference(truth_path)

    np.testing.assert_allclose(reference.theta_initial, [3.0, 1.0])
    np.testing.assert_allclose(reference.theta_truth, [1.0, 1.0])
    assert reference.final_u == 0.12
    assert reference.initial_distance_to_truth == 2.0


def test_theta_start_uses_real_initialization_to_truth_line(tmp_path) -> None:
    reference = script._load_truth_reference(_write_truth_summary(tmp_path))

    theta, distance = script._theta_start(reference, 0.5)

    np.testing.assert_allclose(theta, [2.0, 1.0])
    assert distance == 1.0


def test_variants_center_heteroskedastic_noise_at_truth_final_u(tmp_path) -> None:
    reference = script._load_truth_reference(_write_truth_summary(tmp_path))
    variant_sets = script._variant_sets(reference)
    hetero_variance = [
        variant for variant in variant_sets["heteroskedastic"]
        if variant.axis == "noise_variance" and variant.noise_variance == 4.0
    ][0]
    homo_variance = [
        variant for variant in variant_sets["homoskedastic"]
        if variant.axis == "noise_variance" and variant.noise_variance == 4.0
    ][0]

    assert hetero_variance.u_center == 0.12
    assert hetero_variance.noise_growth == 2.0
    assert hetero_variance.noise_std == 0.0
    assert homo_variance.noise_std == 2.0
    assert homo_variance.noise_growth == 0.0


def test_launch_plan_uses_gpu_jax_scaffolding(tmp_path) -> None:
    truth_path = _write_truth_summary(tmp_path)
    args = script._parse_args([
        "--truth-summary",
        str(truth_path),
        "--grids",
        "heteroskedastic",
        "--run-seeds",
        "8",
    ])

    plan = script._build_launch_plan(args)

    expected_tasks = len(script._variant_sets(script._load_truth_reference(truth_path))["heteroskedastic"])
    assert plan.requires_jax is True
    assert plan.default_launch == "auto"
    assert plan.task_count == expected_tasks


def test_summary_final_rows_compute_distance_and_truth_gaps(tmp_path) -> None:
    reference = script._load_truth_reference(_write_truth_summary(tmp_path))
    payload = {
        "project": "proj",
        "variant": "theta-frac-1",
        "run_seed": 8,
        "run_dir": "run",
        "summary_json": "summary.json",
        "noise_kind": "homoskedastic",
        "axis": "theta_distance",
        "axis_value": 2.0,
        "theta_fraction": 1.0,
        "theta_start_distance_to_truth": 2.0,
        "noise_variance": 1.0,
        "noise_std": 1.0,
        "noise_growth": 0.0,
        "u_center": 0.12,
    }
    summary = {
        "estimators": {
            "finite_difference": {
                "theta": [2.0, 1.0],
                "final_value": -9.0,
                "final_u": 0.10,
                "mean_acceptance": 0.87,
                "runtime_sec": 1.5,
                "optimizer_success": True,
                "optimizer_status": 0,
            }
        }
    }

    rows = script._summary_final_rows(summary, payload, reference)

    assert rows[0]["theta_final_distance_to_truth"] == 1.0
    assert rows[0]["theta_distance_improvement"] == 1.0
    assert rows[0]["objective_gap_to_truth"] == 1.0
    assert rows[0]["mean_acceptance_gap_to_truth"] == pytest.approx(-0.01)


def test_write_outputs_creates_project_csvs_and_plots(tmp_path, monkeypatch) -> None:
    truth_path = _write_truth_summary(tmp_path)
    summary_path = tmp_path / "summary-seed-8.json"
    summary_path.write_text(
        json.dumps(
            {
                "estimators": {
                    "finite_difference": {
                        "theta": [2.0, 1.0],
                        "final_value": -9.0,
                        "final_u": 0.10,
                        "mean_acceptance": 0.87,
                        "runtime_sec": 1.5,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GENERALI_RESULTS_ROOT", str(tmp_path / "results"))
    args = script._parse_args(["--truth-summary", str(truth_path)])
    payloads = [
        {
            "project": "proj",
            "variant": "theta-frac-1",
            "run_seed": 8,
            "run_dir": "run",
            "summary_json": str(summary_path),
            "noise_kind": "homoskedastic",
            "axis": "theta_distance",
            "axis_value": 2.0,
            "theta_fraction": 1.0,
            "theta_start_distance_to_truth": 2.0,
            "noise_variance": 1.0,
            "noise_std": 1.0,
            "noise_growth": 0.0,
            "u_center": 0.12,
        }
    ]

    script._write_outputs_from_payloads(payloads, args)

    project_dir = tmp_path / "results" / "proj"
    assert (project_dir / "noisy_glm_theta_variance_finals.csv").exists()
    assert (project_dir / "noisy_glm_theta_variance_summary.csv").exists()
    assert (project_dir / "plots" / "theta_distance_to_truth.png").exists()
    assert (project_dir / "plots" / "objective_gap_to_truth.png").exists()


def _write_truth_summary(tmp_path):
    path = tmp_path / "truth_summary.json"
    path.write_text(
        json.dumps(
            {
                "config": {
                    "theta0": [3.0, 1.0],
                    "seed": 8,
                },
                "estimators": {
                    "first_order": {
                        "theta": [1.0, 1.0],
                        "final_u": 0.12,
                        "final_value": -10.0,
                        "mean_acceptance": 0.88,
                    }
                },
                "preset": {
                    "overrides": {
                        "compute_backend": "jax",
                        "constraint_mode": "trust_constr",
                        "n_samples": None,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path
