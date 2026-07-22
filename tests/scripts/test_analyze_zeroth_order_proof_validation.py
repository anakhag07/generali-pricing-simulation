from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.configs import get_config
from experiments.manifest import ExperimentManifest, load_experiment_manifest
from objective import ArctanRemainderThetaBias, ArctanThetaBias, LinearThetaBias
from scripts import analyze_zeroth_order_proof_validation as script


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_population_estimator_roots_have_expected_sigma_scaling() -> None:
    sigma = np.asarray([0.05, 0.075, 0.1125, 0.16875, 0.253125])
    fd = np.asarray([abs(script.estimator_root("finite_difference", value, None)) for value in sigma])
    stein = np.asarray([abs(script.estimator_root("stein_difference", value, None)) for value in sigma])

    assert np.polyfit(np.log(sigma), np.log(fd), 1)[0] == pytest.approx(2.0, abs=0.01)
    assert np.polyfit(np.log(sigma), np.log(stein**2), 1)[0] == pytest.approx(4.0, abs=0.03)


def test_bias_optima_distinguish_linear_and_cubic_near_zero_forms() -> None:
    linear = script.biased_optimum(LinearThetaBias(0.2))
    arctan = script.biased_optimum(ArctanThetaBias(0.2))
    remainder = script.biased_optimum(ArctanRemainderThetaBias(0.2))

    assert linear < 0.0
    assert arctan < 0.0
    assert remainder == pytest.approx(0.0, abs=1e-12)
    assert script.estimator_root("finite_difference", 0.15, ArctanRemainderThetaBias(0.2)) != pytest.approx(0.0)


def test_aggregate_mse_decomposition_is_exact() -> None:
    rows = [_minimal_run_row(seed, x_k) for seed, x_k in enumerate([0.1, 0.2, 0.3], start=1)]
    aggregate = script.aggregate_rows(rows)[0]

    assert aggregate["mean_x_k"] == pytest.approx(0.2)
    assert aggregate["total_mse"] == pytest.approx(np.mean(np.square([0.1, 0.2, 0.3])))
    assert aggregate["mse_decomposition_residual"] == pytest.approx(0.0, abs=1e-15)


def test_completed_manifest_outputs_produce_all_tables_and_plots(tmp_path: Path) -> None:
    baseline = load_experiment_manifest(REPO_ROOT / "manifests" / "zeroth_order_baseline.json")
    bias = load_experiment_manifest(REPO_ROOT / "manifests" / "zeroth_order_functional_bias.json")
    _write_fake_summaries(baseline, tmp_path)
    _write_fake_summaries(bias, tmp_path)

    run_rows = [
        *script.collect_run_rows(baseline, runs_root=tmp_path),
        *script.collect_run_rows(bias, runs_root=tmp_path),
    ]
    aggregates = script.aggregate_rows(run_rows)
    fits = script.scaling_fits(aggregates)
    output = tmp_path / "analysis"
    script.write_outputs(run_rows, aggregates, fits, output)

    expected = {
        "per_run_metrics.csv",
        "aggregate_metrics.csv",
        "scaling_fits.csv",
        "theorem_checks.csv",
        "validation_summary.md",
        "sigma_landmarks.png",
        "sigma_displacement_decomposition.png",
        "m_landmarks.png",
        "m_mse_decomposition.png",
        "bias_landmarks.png",
        "bias_displacement_decomposition.png",
        "bias_proof_bounds.png",
        "scaling_summary.png",
    }
    assert expected <= {path.name for path in output.iterdir()}


def _write_fake_summaries(manifest: ExperimentManifest, root: Path) -> None:
    for variant in manifest.variants:
        config = get_config(manifest.base_preset, overrides=variant.overrides)
        bias, *_ = script._bias_metadata(config)
        for seed_index, seed in enumerate(manifest.seeds.run_seeds):
            estimators = {}
            for estimator in config.enabled_estimators:
                target = script.estimator_root(estimator, config.sigma, bias)
                jitter = 0.0 if estimator == "finite_difference" else (seed_index - 7.5) * 1e-5 / np.sqrt(config.n_grad_samples)
                estimators[estimator] = {"theta": [target + jitter]}
            path = manifest.variant_dir(variant, root) / f"summary-seed-{seed}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"estimators": estimators}), encoding="utf-8")


def _minimal_run_row(seed: int, x_k: float) -> dict[str, object]:
    return {
        "project": "test",
        "variant": "base",
        "sweep": "sigma",
        "run_seed": seed,
        "estimator": "stein_difference",
        "bias_form": "none",
        "alpha": 0.0,
        "sigma": 0.1,
        "m": 8,
        "eta": 0.01,
        "k_steps": 10,
        "beta": 0.0,
        "kappa_minus": 0.0,
        "kappa_plus": 0.0,
        "rho_bias": 0.0,
        "strong_convexity_retained": True,
        "stein_step_valid": True,
        "x0": 1.0,
        "x_k": x_k,
        "x_star": 0.0,
        "x_b_star": 0.0,
        "x_estimator_star": 0.15,
        "functional_bias_signed": 0.0,
        "functional_bias_abs": 0.0,
        "smoothing_signed": 0.15,
        "smoothing_abs": 0.15,
        "finite_run_signed": x_k - 0.15,
        "finite_run_abs": abs(x_k - 0.15),
        "truth_error_signed": x_k,
        "truth_error_abs": abs(x_k),
        "truth_error_squared": x_k**2,
        "theorem_bound": 1.0,
        "theorem_metric": "squared_error",
        "summary_path": "summary.json",
        "paired_xk_delta": "",
    }
