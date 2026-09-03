from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.policy_lcb.continuous_gp import (
    ContinuousGPLandscape,
    continuous_gp_seed_for_run,
    load_continuous_gp_variable_lcb_manifest,
)
from experiments.policy_lcb.continuous_gp_decomposition import (
    build_decomposition_conditions,
    collect_continuous_gp_decomposition_outputs,
    continuous_gp_decomposition_seed,
    evaluate_continuous_gp_decomposition_seed,
    load_continuous_gp_decomposition_manifest,
    parse_continuous_gp_decomposition_manifest,
    run_continuous_gp_decomposition_manifest_seed,
)
from experiments.policy_lcb.continuous_gp_core import (
    DecomposedGPLandscape,
    FiniteFourierGPSpec,
    FourierGPDraw,
    GlobalReferenceSpec,
    SmoothClippedUncertaintySpec,
    UniformConfidenceSpec,
    analytic_uniform_certificate,
    certified_coverage_probability,
    certified_shape_ratio,
    certified_weighted_gp_supremum,
)
from experiments.policy_lcb.continuous_gp_decomposition_reporting import (
    _cluster_bootstrap_metrics,
)


def _draw() -> FourierGPDraw:
    gp = FiniteFourierGPSpec(rank=4, lengthscale=0.2)
    return FourierGPDraw(
        gp,
        (0.2, -0.5, 1.1, 0.7),
        (-0.4, 0.3, 0.8, -0.2),
    )


def _uncertainty() -> SmoothClippedUncertaintySpec:
    return SmoothClippedUncertaintySpec((0.0, 0.25, 0.5, 0.75, 1.0), 0.1, 1.0, 0.5)


def _payload() -> dict[str, object]:
    path = Path(__file__).parents[2] / "manifests" / "continuous_gp_regret_decomposition.json"
    payload = json.loads(path.read_text())
    payload["name"] = "decomposition-test"
    payload["gp"]["rank"] = 4  # type: ignore[index]
    payload["global_reference"]["value_tolerance"] = 1e-4  # type: ignore[index]
    payload["global_reference"]["initial_grid_count"] = 17  # type: ignore[index]
    payload["design"]["one_axis_scales"] = [0.0, 1.0]  # type: ignore[index]
    payload["design"]["factorial_scales"] = [1.0]  # type: ignore[index]
    payload["optimizer"]["checkpoint_steps"] = [0, 1, 2]  # type: ignore[index]
    payload["optimizer"]["n_stein_perturbations"] = 4  # type: ignore[index]
    payload["seeds"]["run_seeds"] = {"type": "range", "start": 101, "count": 2}  # type: ignore[index]
    payload["seeds"]["diagnostic_run_seeds"] = [101]  # type: ignore[index]
    return payload


def test_decomposed_landscape_derivatives_and_matched_legacy_parity() -> None:
    draw = _draw()
    uncertainty = _uncertainty()
    confidence = UniformConfidenceSpec(delta=0.05, net_count=33)
    quantile = analytic_uniform_certificate(draw.spec, confidence).quantile
    landscape = DecomposedGPLandscape(
        draw,
        uncertainty,
        surrogate_center=0.25,
        surrogate_scale=0.5,
        envelope_center=0.25,
        envelope_scale=0.5,
        quantile=quantile,
    )
    legacy = ContinuousGPLandscape(
        draw,
        uncertainty,
        center=0.25,
        noise_scale=0.5,
        quantile=quantile,
        target="variable_lcb",
    )
    x = 0.37
    step = 1e-5
    first = (landscape.evaluate(x + step) - landscape.evaluate(x - step)) / (2.0 * step)
    second = (
        landscape.evaluate(x + step)
        - 2.0 * landscape.evaluate(x)
        + landscape.evaluate(x - step)
    ) / step**2
    assert landscape.evaluate(x, 1) == pytest.approx(first, rel=1e-7, abs=1e-7)
    assert landscape.evaluate(x, 2) == pytest.approx(second, rel=2e-5, abs=2e-5)
    grid = np.linspace(0.0, 1.0, 101)
    assert landscape.evaluate(grid) == pytest.approx(legacy.evaluate(grid))


def test_certificate_inversion_and_shape_ratio_are_locked() -> None:
    gp = FiniteFourierGPSpec(rank=32, lengthscale=0.2)
    confidence = UniformConfidenceSpec(delta=0.05, net_count=129)
    quantile = analytic_uniform_certificate(gp, confidence).quantile
    assert certified_coverage_probability(gp, confidence, quantile) == pytest.approx(
        0.95, abs=1e-10
    )
    reference = GlobalReferenceSpec(1e-6, 200000, 65)
    matched = certified_shape_ratio(_uncertainty(), 0.5, 0.5, reference)
    mismatched = certified_shape_ratio(_uncertainty(), 0.5, 0.25, reference)
    assert matched.lower_bound == matched.upper_bound == 1.0
    assert mismatched.certified
    assert 0.0 < mismatched.lower_bound <= mismatched.value <= mismatched.upper_bound < 1.0


def test_weighted_surrogate_error_supremum_brackets_dense_evaluation() -> None:
    draw = _draw()
    uncertainty = _uncertainty()
    reference = GlobalReferenceSpec(1e-6, 200000, 65)
    result = certified_weighted_gp_supremum(draw, uncertainty, 0.25, reference)
    grid = np.linspace(0.0, 1.0, 10001)
    dense = np.max(
        np.abs(
            np.asarray(draw.evaluate(grid))
            * np.asarray(
                [
                    0.1
                    + 0.9
                    * (
                        6 * min(abs(x - 0.25) / 0.5, 1.0) ** 5
                        - 15 * min(abs(x - 0.25) / 0.5, 1.0) ** 4
                        + 10 * min(abs(x - 0.25) / 0.5, 1.0) ** 3
                    )
                    for x in grid
                ]
            )
        )
    )
    assert result.certified
    assert result.lower_bound <= dense + 1e-8
    assert dense <= result.upper_bound + 1e-8


def test_manifest_grid_is_deduplicated_and_committed_contract_is_locked() -> None:
    manifest = parse_continuous_gp_decomposition_manifest(_payload())
    conditions = build_decomposition_conditions(manifest.spec.design)
    keys = {
        (
            row.surrogate_center,
            row.surrogate_scale,
            row.envelope_center,
            row.envelope_scale,
        )
        for row in conditions
    }
    assert len(keys) == len(conditions)
    matched = next(
        row
        for row in conditions
        if (
            row.surrogate_center,
            row.surrogate_scale,
            row.envelope_center,
            row.envelope_scale,
        )
        == (0.5, 1.0, 0.5, 1.0)
    )
    assert set(matched.memberships) >= {
        "axis_1_surrogate_scale",
        "axis_2_envelope_scale",
        "axis_2_envelope_shape",
        "axis_3_optimizer",
        "combined_factorial",
        "shape_robustness",
    }
    committed = load_continuous_gp_decomposition_manifest(
        Path(__file__).parents[2] / "manifests" / "continuous_gp_regret_decomposition.json"
    )
    assert committed.spec.run_seeds == tuple(range(101, 301))
    assert committed.spec.diagnostic_run_seeds == (179, 116, 225)
    assert committed.spec.optimizer.checkpoint_steps == (0, 5, 10, 25, 50, 100, 250, 500)
    assert committed.spec.optimizer.enabled_estimators == (
        "finite_difference",
        "stein_difference",
    )


def test_seed_mapping_reuses_original_fourier_paths() -> None:
    decomposition = parse_continuous_gp_decomposition_manifest(_payload())
    old = load_continuous_gp_variable_lcb_manifest(
        Path(__file__).parents[2] / "manifests" / "continuous_gp_variable_lcb.json"
    )
    assert continuous_gp_decomposition_seed(decomposition.spec, 101) == continuous_gp_seed_for_run(
        old.spec, 101
    )


def test_small_evaluation_records_exact_and_paired_optimizer_metrics() -> None:
    manifest = parse_continuous_gp_decomposition_manifest(_payload())
    result = evaluate_continuous_gp_decomposition_seed(manifest.spec, 101)
    assert len(result.conditions) == len(manifest.spec.conditions())
    assert result.checkpoints
    raw = [row for row in result.checkpoints if not row.is_best_start]
    best = [row for row in result.checkpoints if row.is_best_start]
    expected_raw = (
        sum(condition.run_optimizer for condition in manifest.spec.conditions())
        * len(manifest.spec.optimizer.enabled_estimators)
        * len(manifest.spec.optimizer.starts)
        * len(manifest.spec.optimizer.checkpoint_steps)
    )
    assert len(raw) == expected_raw
    assert len(best) == expected_raw // len(manifest.spec.optimizer.starts)
    assert {row.oracle_queries for row in best if row.estimator == "finite_difference"} == {0, 2, 4}
    assert {row.oracle_queries for row in best if row.estimator == "stein_difference"} == {0, 8, 16}
    assert all(row.certificate_slack_upper >= -1e-4 for row in best if row.certificate_eligible)

    replay = evaluate_continuous_gp_decomposition_seed(manifest.spec, 101)
    assert result == replay


def test_manifest_rejects_missing_real_line_probe_contract() -> None:
    payload = copy.deepcopy(_payload())
    payload["optimizer"]["probe_domain"] = "clipped"  # type: ignore[index]
    with pytest.raises(ValueError, match="real-line extension"):
        parse_continuous_gp_decomposition_manifest(payload)


def test_cluster_bootstrap_sufficient_statistics_match_expanded_rows() -> None:
    y = np.asarray([0.2, 0.4, 1.0, 1.2, 0.8, 0.1, 0.3], dtype=float)
    residual = np.asarray([0.1, -0.2, 0.3, -0.1, 0.2, -0.05, 0.04], dtype=float)
    seeds = np.asarray([101, 101, 102, 102, 102, 103, 103], dtype=int)
    fast_r2, fast_mae = _cluster_bootstrap_metrics(
        y,
        residual,
        seeds,
        np.random.default_rng(77),
        n_bootstrap=40,
    )
    rng = np.random.default_rng(77)
    unique = np.unique(seeds)
    expected_r2 = []
    expected_mae = []
    for _ in range(40):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(seeds == seed) for seed in chosen])
        sample_y = y[indices]
        sample_residual = residual[indices]
        denominator = float(np.sum((sample_y - np.mean(sample_y)) ** 2))
        expected_r2.append(
            1.0 - float(np.sum(sample_residual**2)) / denominator
            if denominator > 0.0
            else float("nan")
        )
        expected_mae.append(float(np.mean(np.abs(sample_residual))))
    assert fast_r2 == pytest.approx(expected_r2, nan_ok=True)
    assert fast_mae == pytest.approx(expected_mae)


def test_persistence_resume_collection_and_five_plots(tmp_path: Path) -> None:
    manifest = parse_continuous_gp_decomposition_manifest(_payload())
    first = run_continuous_gp_decomposition_manifest_seed(manifest, 0, runs_root=tmp_path)
    skipped = run_continuous_gp_decomposition_manifest_seed(manifest, 0, runs_root=tmp_path)
    run_continuous_gp_decomposition_manifest_seed(manifest, 1, runs_root=tmp_path)
    collected = collect_continuous_gp_decomposition_outputs(manifest, runs_root=tmp_path)
    assert first["skipped"] is False
    assert skipped["skipped"] is True
    assert collected["n_seed_results"] == 2
    project = manifest.project_dir(tmp_path)
    for filename in (
        "EXPERIMENT.md",
        "seed_condition_metrics.csv",
        "seed_optimizer_checkpoints.csv",
        "seed_optimizer_best.csv",
        "axis_summary.csv",
        "optimizer_summary.csv",
        "coverage_summary.csv",
        "explanatory_model_summary.csv",
    ):
        assert (project / filename).exists()
    for filename in (
        "one_at_a_time_sweeps.pdf",
        "optimizer_error_vs_regret.pdf",
        "decomposition_bound_check.pdf",
        "factorial_tradeoff_heatmaps.pdf",
        "representative_landscapes.pdf",
    ):
        assert (project / "plots" / filename).exists()
    payload = json.loads(manifest.seed_result_path(101, tmp_path).read_text())
    assert "a_coefficients" in payload
    assert "checkpoints" not in payload
    with np.load(manifest.seed_checkpoint_path(101, tmp_path), allow_pickle=False) as data:
        assert "oracle_queries" in data.files
        assert "is_best_start" in data.files
