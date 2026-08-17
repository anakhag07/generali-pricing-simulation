from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.policy_lcb.continuous_gp import (
    ContinuousGPLandscape,
    FiniteFourierGPSpec,
    FourierGPDraw,
    GlobalReferenceSpec,
    SmoothClippedUncertaintySpec,
    UniformConfidenceSpec,
    analytic_uniform_certificate,
    certified_global_maximum,
    collect_continuous_gp_variable_lcb_outputs,
    evaluate_continuous_gp_variable_lcb_draw,
    evaluate_continuous_gp_variable_lcb_seed,
    load_continuous_gp_variable_lcb_manifest,
    parse_continuous_gp_variable_lcb_manifest,
    run_continuous_gp_variable_lcb_manifest_seed,
    smooth_clipped_uncertainty,
    smoothstep,
)


def _payload() -> dict[str, object]:
    return {
        "kind": "continuous_gp_variable_lcb",
        "name": "continuous-gp-test",
        "domain": [0.0, 1.0],
        "true_value": {"type": "concave_quadratic", "linear": 5.0, "quadratic": 5.0},
        "gp": {
            "type": "deterministic_spectral_finite_rank",
            "rank": 4,
            "lengthscale": 0.2,
        },
        "uncertainty": {
            "type": "smooth_clipped_distance_ramp",
            "centers": [0.0, 0.5],
            "minimum": 0.1,
            "maximum": 1.0,
            "ramp_radius": 0.5,
        },
        "surrogate": {"type": "scaled_nonstationary_gp", "noise_scales": [0.0, 0.5]},
        "confidence": {
            "type": "bonferroni_net_smoothness",
            "delta": 0.05,
            "net_count": 33,
        },
        "global_reference": {
            "type": "certified_branch_and_bound",
            "value_tolerance": 1e-4,
            "max_intervals": 50000,
            "initial_grid_count": 33,
        },
        "optimizer": {
            "step_rule": "projected_constant",
            "probe_domain": "analytic_real_line_extension",
            "enabled_estimators": ["first_order", "finite_difference", "stein_difference"],
            "starts": [0.1, 0.5, 0.9],
            "t_steps": 5,
            "step_size": 0.01,
            "perturbation_radius": 0.02,
            "n_stein_perturbations": 4,
            "checkpoint_every": 1,
        },
        "seeds": {
            "master_gp_seed": 20260817,
            "master_optimizer_seed": 20260818,
            "reporting_seed": 20260819,
            "run_seeds": {"type": "range", "start": 101, "count": 2},
            "optimizer_run_seeds": {"type": "range", "start": 101, "count": 1},
        },
        "launch": {"mode": "local", "array": "seed"},
    }


def test_smoothstep_and_uncertainty_are_c2_and_move_with_center() -> None:
    assert smoothstep([0.0, 0.5, 1.0]) == pytest.approx([0.0, 0.5, 1.0])
    assert smoothstep([0.0, 1.0], 1) == pytest.approx([0.0, 0.0])
    assert smoothstep([0.0, 1.0], 2) == pytest.approx([0.0, 0.0])
    uncertainty = SmoothClippedUncertaintySpec((0.0, 0.5), 0.1, 1.0, 0.5)
    x = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    assert smooth_clipped_uncertainty(x, 0.5, uncertainty) == pytest.approx(
        [1.0, 0.55, 0.1, 0.55, 1.0]
    )
    assert smooth_clipped_uncertainty(0.5, 0.5, uncertainty, 1) == 0.0
    assert smooth_clipped_uncertainty(0.5, 0.5, uncertainty, 2) == 0.0


def test_fourier_draw_is_analytic_and_derivatives_match_finite_differences() -> None:
    gp = FiniteFourierGPSpec(rank=4, lengthscale=0.2)
    draw = FourierGPDraw(gp, (0.2, -0.5, 1.1, 0.7), (-0.4, 0.3, 0.8, -0.2))
    x = 0.37
    step = 1e-5
    first = (draw.evaluate(x + step) - draw.evaluate(x - step)) / (2.0 * step)
    second = (draw.evaluate(x + step) - 2.0 * draw.evaluate(x) + draw.evaluate(x - step)) / step**2
    assert draw.evaluate(x, 1) == pytest.approx(first, rel=1e-7, abs=1e-7)
    assert draw.evaluate(x, 2) == pytest.approx(second, rel=2e-5, abs=2e-5)

    omega = gp.frequencies()
    left, right = 0.21, 0.83
    expected_kernel = np.mean(np.cos(omega * (left - right)))
    features_left = np.r_[np.cos(omega * left), np.sin(omega * left)] / np.sqrt(gp.rank)
    features_right = np.r_[np.cos(omega * right), np.sin(omega * right)] / np.sqrt(gp.rank)
    assert features_left @ features_right == pytest.approx(expected_kernel)
    assert features_left @ features_left == pytest.approx(1.0)


def test_analytic_certificate_matches_locked_continuum_calculation() -> None:
    certificate = analytic_uniform_certificate(
        FiniteFourierGPSpec(rank=32, lengthscale=0.2),
        UniformConfidenceSpec(delta=0.05, net_count=129),
    )
    assert certificate.feature_lipschitz == pytest.approx(4.950534602512078)
    assert certificate.net_quantile == pytest.approx(3.7269661820826827)
    assert certificate.remainder == pytest.approx(0.18141093727487437)
    assert certificate.quantile == pytest.approx(3.908377119357557)


def test_branch_and_bound_certifies_multimodal_global_value() -> None:
    result = certified_global_maximum(
        lambda x: np.cos(6.0 * np.pi * x) + 0.1 * x,
        second_derivative_bound=(6.0 * np.pi) ** 2,
        reference=GlobalReferenceSpec(1e-6, 200000, 65),
    )
    dense_x = np.linspace(0.0, 1.0, 200001)
    dense_max = float(np.max(np.cos(6.0 * np.pi * dense_x) + 0.1 * dense_x))
    assert result.certified
    assert result.upper_bound - result.value <= 1e-6
    assert dense_max <= result.upper_bound + 1e-12
    assert result.value == pytest.approx(dense_max, abs=1e-6)


def test_seed_replay_pairing_zero_scale_and_conditional_lcb_certificate() -> None:
    manifest = parse_continuous_gp_variable_lcb_manifest(_payload())
    first = evaluate_continuous_gp_variable_lcb_seed(manifest.spec, 101)
    replay = evaluate_continuous_gp_variable_lcb_seed(manifest.spec, 101)
    second = evaluate_continuous_gp_variable_lcb_seed(manifest.spec, 102)
    assert first == replay
    assert first.a_coefficients != second.a_coefficients
    assert len(first.conditions) == 2 * 2
    assert len(first.selectors) == 2 * 2 * 2
    assert len(first.optimizer_finals) == 2 * 2 * 2 * 3 * 3
    zero_conditions = [row for row in first.conditions if row.noise_scale == 0.0]
    zero_selectors = [row for row in first.selectors if row.noise_scale == 0.0]
    assert all(row.simultaneous_coverage and row.maximum_half_width == 0.0 for row in zero_conditions)
    assert all(row.selected_x == 0.5 and row.regret == 0.0 for row in zero_selectors)
    for row in first.selectors:
        if row.target == "variable_lcb" and row.selected_point_covered:
            assert row.certificate_slack is not None and row.certificate_slack >= -1e-6


def test_one_draw_changes_by_sigma_not_by_resampling_across_centers() -> None:
    manifest = parse_continuous_gp_variable_lcb_manifest(_payload())
    draw = FourierGPDraw(manifest.spec.gp, (0.1, 0.2, 0.3, 0.4), (-0.4, -0.3, -0.2, -0.1))
    certificate = analytic_uniform_certificate(manifest.spec.gp, manifest.spec.confidence)
    left = ContinuousGPLandscape(draw, manifest.spec.uncertainty, 0.0, 0.5, certificate.quantile, "nominal")
    middle = ContinuousGPLandscape(draw, manifest.spec.uncertainty, 0.5, 0.5, certificate.quantile, "nominal")
    x = 0.25
    standardized_left = (left.evaluate(x) - (5 * x - 5 * x**2)) / (
        0.5 * smooth_clipped_uncertainty(x, 0.0, manifest.spec.uncertainty)
    )
    standardized_middle = (middle.evaluate(x) - (5 * x - 5 * x**2)) / (
        0.5 * smooth_clipped_uncertainty(x, 0.5, manifest.spec.uncertainty)
    )
    assert standardized_left == pytest.approx(draw.evaluate(x))
    assert standardized_middle == pytest.approx(draw.evaluate(x))


def test_manifest_validation_and_committed_full_cube() -> None:
    manifest = parse_continuous_gp_variable_lcb_manifest(_payload())
    assert manifest.spec.run_seeds == (101, 102)
    assert manifest.spec.optimizer_run_seeds == (101,)

    invalid = copy.deepcopy(_payload())
    invalid["optimizer"]["probe_domain"] = "clipped"  # type: ignore[index]
    with pytest.raises(ValueError, match="real-line extension"):
        parse_continuous_gp_variable_lcb_manifest(invalid)

    path = Path(__file__).parents[2] / "manifests" / "continuous_gp_variable_lcb.json"
    committed = load_continuous_gp_variable_lcb_manifest(path)
    assert committed.spec.gp.rank == 32
    assert committed.spec.gp.lengthscale == 0.2
    assert committed.spec.uncertainty.centers == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert committed.spec.noise_scales == (0.0, 0.25, 0.5, 1.0, 2.0)
    assert len(committed.spec.run_seeds) == 2000
    assert len(committed.spec.optimizer_run_seeds) == 200
    assert committed.spec.optimizer.n_stein_perturbations == 64


def test_persistence_resume_collection_and_plots(tmp_path: Path) -> None:
    manifest = parse_continuous_gp_variable_lcb_manifest(_payload())
    first = run_continuous_gp_variable_lcb_manifest_seed(manifest, 0, runs_root=tmp_path)
    skipped = run_continuous_gp_variable_lcb_manifest_seed(manifest, 0, runs_root=tmp_path)
    run_continuous_gp_variable_lcb_manifest_seed(manifest, 1, runs_root=tmp_path)
    collected = collect_continuous_gp_variable_lcb_outputs(manifest, runs_root=tmp_path)
    assert first["skipped"] is False
    assert skipped["skipped"] is True
    assert collected["n_condition_rows"] == 2 * 2 * 2
    assert collected["n_selector_rows"] == 2 * 2 * 2 * 2
    project = manifest.project_dir(tmp_path)
    for filename in (
        "EXPERIMENT.md",
        "seed_gp_metrics.csv",
        "seed_condition_metrics.csv",
        "seed_selector_metrics.csv",
        "seed_optimizer_finals.csv",
        "condition_summary.csv",
        "selector_summary.csv",
        "optimizer_final_summary.csv",
        "optimizer_trajectory_summary.csv",
        "coverage_summary.csv",
    ):
        assert (project / filename).exists()
    for filename in (
        "realized_analytic_landscape.png",
        "simultaneous_coverage.png",
        "regret_by_center.png",
        "selected_location_by_center.png",
        "envelope_tightness.png",
        "optimizer_global_gap_heatmaps.png",
        "optimizer_convergence.png",
    ):
        assert (project / "plots" / filename).exists()
    payload = json.loads(manifest.seed_result_path(101, tmp_path).read_text())
    assert "a_coefficients" in payload
    assert "trajectories" not in payload
    assert manifest.seed_trajectory_path(101, tmp_path).exists()
    assert not manifest.seed_trajectory_path(102, tmp_path).exists()
    with np.load(manifest.seed_trajectory_path(101, tmp_path), allow_pickle=False) as data:
        assert len(data["step"]) > 0
        assert set(data.files) == {
            "run_seed",
            "uncertainty_center",
            "noise_scale",
            "target",
            "estimator",
            "start_x",
            "step",
            "x",
            "target_value",
            "true_regret",
            "optimization_gap",
        }
