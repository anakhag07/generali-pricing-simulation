from __future__ import annotations

import json

import numpy as np
import pytest

from scratch import plot_planted_logistic_gradient_sample_noise_diagnostics as script


def test_variance_summary_rows_compute_theta_bias_variance_and_mse() -> None:
    truth = np.asarray([0.0, 0.0], dtype=float)
    rows = [
        _row(run_seed=1, theta_hat=np.asarray([1.0, 0.0], dtype=float), final_u=0.1),
        _row(run_seed=2, theta_hat=np.asarray([3.0, 0.0], dtype=float), final_u=0.3),
    ]

    summary = script.variance_summary_rows(rows, truth_theta=truth)

    assert len(summary) == 1
    row = summary[0]
    assert row["n_seeds"] == 2
    assert row["theta_bias_squared"] == pytest.approx(4.0)
    assert row["theta_variance_trace"] == pytest.approx(2.0)
    assert row["theta_mse"] == pytest.approx(5.0)
    assert row["theta_distance_to_truth_mean"] == pytest.approx(2.0)
    assert row["theta_distance_to_truth_var"] == pytest.approx(2.0)
    assert row["final_u_var"] == pytest.approx(0.02)


def test_collect_diagnostic_rows_reconstructs_clean_and_noisy_objectives(tmp_path) -> None:
    project_dir = tmp_path / "project"
    variant_dir = project_dir / "finite_difference__homoskedastic-std-0"
    variant_dir.mkdir(parents=True)
    summary_path = variant_dir / "summary-seed-7.json"
    truth_summary = tmp_path / "truth.json"
    theta_truth = [0.4054882808450241, 0.00012799868045781167, -4.657524122982136e-05, 6.221922280809605e-05]
    theta_hat = [0.40524034479431165, 0.0020662098755075643, -0.00042615032449958603, 0.0003708854846674679]
    truth_summary.write_text(
        json.dumps({"estimators": {"first_order": {"theta": theta_truth}}}),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(_summary_payload(theta_hat)),
        encoding="utf-8",
    )

    rows = script.collect_diagnostic_rows(project_dir, truth_summary)

    assert len(rows) == 1
    row = rows[0]
    assert row.estimator == script.FINITE_DIFFERENCE
    assert row.noise_family == "homoskedastic"
    assert row.noise_level == 0.0
    assert row.n_grad_samples is None
    assert row.run_seed == 7
    assert row.theta_distance_to_truth > 0.0
    assert row.clean_objective_gap == pytest.approx(row.noisy_objective_gap)
    assert row.noise_exploitation_gap == pytest.approx(0.0)


def _row(run_seed: int, theta_hat: np.ndarray, final_u: float) -> script.DiagnosticRow:
    distance = float(np.linalg.norm(theta_hat))
    return script.DiagnosticRow(
        variant="v",
        estimator=script.STEIN_DIFFERENCE,
        run_seed=run_seed,
        noise_family="homoskedastic",
        noise_level=0.5,
        n_grad_samples=64,
        theta_hat=theta_hat,
        theta_distance_to_truth=distance,
        theta_mse_to_truth=float(np.mean(theta_hat**2)),
        clean_objective_truth=1.0,
        clean_objective_hat=1.0 + distance,
        clean_objective_gap=distance,
        noisy_objective_truth=0.8,
        noisy_objective_hat=0.8 + distance,
        noisy_objective_gap=distance,
        noise_exploitation_gap=0.0,
        final_u=final_u,
        optimizer_success=True,
        optimizer_status=0,
        summary_path="summary.json",
    )


def _summary_payload(theta_hat: list[float]) -> dict[str, object]:
    return {
        "config": {
            "n_samples": 20,
            "state_dim": 3,
            "test_fraction": 0.0,
            "resolved_seed_setup": {
                "data_seed": 7,
                "split_seed": 7,
                "noise_seed": 7,
                "optimizer_seed": 7,
                "theta_seed": 7,
                "run_seed": 7,
            },
            "objective": {
                "type": "NoisyObjective",
                "base_objective": {
                    "type": "PlantedLogisticObjective",
                    "alpha": 1.0,
                    "beta": [0.5, -0.2, 0.3],
                    "bias": -0.2,
                    "u_star": 0.1,
                    "policy": {
                        "type": "SoftmaxPolicy",
                        "action_low": -0.5,
                        "action_high": 0.5,
                        "feature_map": {"kind": "identity", "type": "IdentityFeatureMap"},
                    },
                },
                "noise": {
                    "type": "HomoskedasticGaussianNoise",
                    "std": 0.0,
                    "seed": 7,
                },
            },
        },
        "estimators": {
            script.FINITE_DIFFERENCE: {
                "theta": theta_hat,
                "final_u": 0.09965926556331751,
                "optimizer_success": True,
                "optimizer_status": 0,
            }
        },
        "preset": {
            "variant_name": "finite_difference__homoskedastic-std-0",
            "noise_family": "homoskedastic",
            "noise_level": 0.0,
            "n_grad_samples": None,
        },
    }
