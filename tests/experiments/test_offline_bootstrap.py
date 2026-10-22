"""Behavioral checks for fixed observations, bootstrap sampling, and replay."""

import json
from math import comb
from pathlib import Path

import numpy as np
import pytest

from experiments.bootstrap_band import features
from experiments.offline_bootstrap import (
    _calibration,
    load_manifest,
    prepare_snapshot,
    quantile_spec,
    run_experiment,
    sample_pairs_indices,
)
from experiments.provenance import file_record


ROOT = Path(__file__).resolve().parents[2]


def _manifest(tmp_path):
    payload = json.loads((ROOT / "scratch/retired_experiments/bootstrap_ols/manifests/offline_bootstrap.json").read_text())
    x = np.array([-1.2, -.9, -.6, -.3, 0., .3, .6, .9, 1.2, 1.4, -1.05, .45])
    errors = np.array([.3, -.1, .2, -.4, .5, -.2, .15, -.25, .35, -.05, .45, -.3])
    y = features(x) @ np.array([0., 5., -5.]) + errors
    candidate = np.array([-.4, 5.1, -5.1])
    observations = tmp_path / "observations.npz"
    predictor = tmp_path / "predictor.npz"
    np.savez_compressed(observations, saved_x=x, saved_y=y)
    np.savez_compressed(predictor, saved_beta=candidate)
    payload.update(
        name="offline-test",
        offline_data={"path": observations.name, "sha256": file_record(observations)["sha256"],
                      "x_key": "saved_x", "y_key": "saved_y", "source_sigma": 1.},
        predictor={"path": predictor.name, "sha256": file_record(predictor)["sha256"],
                   "beta_key": "saved_beta"},
        baseline={"N": 8, "B": 12, "sigma": 1.},
        axes={"N": [4, 8, 12], "B": [8, 12], "sigma": [.5, 1.]},
        delta=.2,
        gp={"rank": 4, "lengthscale": .3},
    )
    payload["bootstrap"]["mc_failure_probability"] = .2
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return load_manifest(path), x, y, candidate


def _forbid_normal_draws(monkeypatch):
    default_rng = np.random.default_rng

    class ObservedRowsOnly:
        def __init__(self, rng):
            self.rng = rng

        def normal(self, *args, **kwargs):
            raise AssertionError("Calibration drew fresh population/observation noise.")

        def __getattr__(self, name):
            return getattr(self.rng, name)

    monkeypatch.setattr(np.random, "default_rng", lambda *a, **kw: ObservedRowsOnly(default_rng(*a, **kw)))


def test_binomial_quantile_guarantee_and_insufficient_draws():
    bootstrap = {"quantile": "mc_upper", "mc_failure_probability": .05}
    with pytest.raises(ValueError, match="too small"):
        quantile_spec(58, .05, bootstrap)
    assert quantile_spec(59, .05, bootstrap)["rank"] == 59
    result = quantile_spec(100, .05, bootstrap)
    assert result["rank"] == 99
    rank = result["rank"]
    exact = sum(comb(100, j) * .95**j * .05**(100-j) for j in range(rank))
    preceding = exact - comb(100, rank-1) * .95**(rank-1) * .05**(101-rank)
    assert preceding < .95 <= exact
    assert result["target_mass_confidence"] == pytest.approx(exact)
    assert result["bootstrap_mass_lower"] >= .95
    empirical = quantile_spec(8, .05, {**bootstrap, "quantile": "higher"})
    assert empirical["rank"] == 8
    assert empirical["bootstrap_mass_lower"] < .95


def test_snapshot_preserves_original_pairs_fixed_candidate_and_gp_normalization(tmp_path):
    manifest, x, y, candidate = _manifest(tmp_path)
    project = tmp_path / "results" / manifest.name
    metadata, arrays = prepare_snapshot(manifest, project)
    np.testing.assert_array_equal(arrays["x"], x)
    np.testing.assert_array_equal(arrays["source_y"], y)
    np.testing.assert_array_equal(arrays["homoskedastic_sigma1_y"], y)
    np.testing.assert_array_equal(arrays["candidate_beta"], candidate)
    mean = features(x) @ arrays["truth_beta"]
    for profile in manifest.payload["profiles"]:
        name = profile["name"]
        h = arrays[f"{name}_h"]
        assert np.all(h > 0)
        assert np.mean(h*h) == pytest.approx(1.)
        np.testing.assert_allclose(arrays[f"{name}_sigma0.5_y"]-mean,
                                   .5*(arrays[f"{name}_sigma1_y"]-mean), atol=1e-14)
    assert np.ptp(arrays["heteroskedastic_gp_h"]) > .01
    assert metadata["reservoir_size"] == len(x)
    assert metadata["inference"]["sampling_coverage_lower_bound"] is None
    # The complete reservoir, not the largest requested N, defines normalization.
    reduced = json.loads(manifest.source_path.read_text())
    reduced["name"] = "offline-reduced"
    reduced["axes"]["N"] = [4, 8]
    path = tmp_path / "reduced.json"
    path.write_text(json.dumps(reduced))
    _, fewer = prepare_snapshot(load_manifest(path), tmp_path / "reduced")
    np.testing.assert_array_equal(fewer["heteroskedastic_gp_h"], arrays["heteroskedastic_gp_h"])


def test_rank_redraws_keep_b_prefix_and_do_not_change_with_profile_seed():
    # Full rank initially, with many singular bootstrap samples to exercise retries.
    x = np.array([0., 0., 0., 0., 1., 2.])
    seeds = {"bootstrap": 3301, "bootstrap_redraw": 3302, "gp_profile": 4301}
    first, rejected = sample_pairs_indices(x, 12, seeds, 1000)
    larger, more_rejected = sample_pairs_indices(x, 31, seeds, 1000)
    np.testing.assert_array_equal(first, larger[:12])
    np.testing.assert_array_equal(rejected, more_rejected[:12])
    assert rejected.sum() > 0
    assert all(np.linalg.matrix_rank(features(x[rows])) == 3 for rows in larger)
    changed, changed_rejected = sample_pairs_indices(x, 12, {**seeds, "gp_profile": 99}, 1000)
    np.testing.assert_array_equal(first, changed)
    np.testing.assert_array_equal(rejected, changed_rejected)
    with pytest.raises(RuntimeError, match="retry limit"):
        sample_pairs_indices(x, 31, seeds, 0)


def test_calibration_refits_only_saved_pairs_and_pairs_profiles(tmp_path, monkeypatch):
    manifest, x, _, candidate = _manifest(tmp_path)
    project = tmp_path / "results" / manifest.name
    snapshot, arrays = prepare_snapshot(manifest, project)
    _forbid_normal_draws(monkeypatch)
    previous_indices = None
    for profile in ("homoskedastic", "heteroskedastic_gp"):
        _, cal = _calibration(manifest, project, snapshot, arrays, profile, 8, 12)
        design = features(x[:8])
        observed = arrays[f"{profile}_sigma1_y"][:8]
        fitted = np.linalg.lstsq(design, observed, rcond=None)[0]
        direct = np.array([np.linalg.lstsq(design[rows], observed[rows], rcond=None)[0]
                           for rows in cal["indices"]])
        np.testing.assert_allclose(cal["beta"], fitted, atol=1e-12)
        np.testing.assert_allclose(cal["beta"]+cal["sigma_hat"]*cal["standardized_perturbations"],
                                   direct, atol=1e-12)
        assert not np.allclose(cal["beta"], candidate)
        if previous_indices is not None:
            np.testing.assert_array_equal(cal["indices"], previous_indices)
        previous_indices = cal["indices"]


def test_single_snapshot_sweeps_regret_replay_and_tamper_detection(tmp_path, monkeypatch):
    manifest, _, _, candidate = _manifest(tmp_path)
    runs_root = tmp_path / "results"
    project = runs_root / manifest.name
    prepare_snapshot(manifest, project)
    _forbid_normal_draws(monkeypatch)
    summary = run_experiment(manifest, runs_root)
    assert summary["inference"]["outer_datasets"] == 0
    assert summary["inference"]["sampling_coverage_lower_bound"] is None
    assert len(summary["rows"]) == 2*sum(map(len, manifest.payload["axes"].values()))
    assert all(row["plugin_action"] == summary["plugin"]["action"] for row in summary["rows"])
    certified = 0
    for row in summary["rows"]:
        for axis in ("N", "B", "sigma"):
            assert row[axis] == (row["axis_value"] if row["axis"] == axis else manifest.payload["baseline"][axis])
        with np.load(project / "bands" / f"{row['artifact_key']}.npz") as band:
            np.testing.assert_array_equal(band["candidate_beta"], candidate)
        assert row["radius_at_reference"] == pytest.approx(row["width_at_reference"]+row["discrepancy_at_reference"])
        assert row["bootstrap_mass_lower"] >= 1-manifest.payload["delta"]
        if row["optimization_state"] == "certified":
            certified += 1
            assert row["lcb_regret_lower"] <= row["lcb_regret_upper"]
            if row["regret_bound_applicable"]:
                assert row["envelope_covered"]
                assert row["lcb_regret_upper"] <= row["lcb_regret_bound"]+1e-9
        else:
            assert row["lcb_regret_upper"] is None
            assert not row["regret_bound_applicable"]
    assert certified > 0
    for profile in ("homoskedastic", "heteroskedastic_gp"):
        rows = [r for r in summary["rows"] if r["profile"] == profile and r["axis"] == "sigma"]
        assert len({r["auxiliary_covered"] for r in rows}) == 1
        normalized_width = [r["width_at_reference"]/r["sigma"] for r in rows]
        assert normalized_width[0] == pytest.approx(normalized_width[1], rel=1e-12)
    assert summary["plots"]
    for record in summary["plots"]:
        assert Path(record["path"]).suffix == ".pdf"
        assert Path(record["path"]).read_bytes().startswith(b"%PDF")
    assert run_experiment(manifest, runs_root) == summary
    checkpoint = project / "calibration" / "homoskedastic" / "N8" / "draws.npz"
    with checkpoint.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Saved artifact changed"):
        run_experiment(manifest, runs_root)


@pytest.mark.parametrize("target", ["source", "snapshot"])
def test_offline_source_and_snapshot_checksums_are_enforced(tmp_path, target):
    manifest, _, _, _ = _manifest(tmp_path)
    project = tmp_path / "results" / manifest.name
    prepare_snapshot(manifest, project)
    path = tmp_path / "observations.npz" if target == "source" else project / "offline_snapshot.npz"
    with path.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum changed|Saved artifact changed"):
        prepare_snapshot(manifest, project)


def test_regret_endpoints_round_outward_for_represented_coefficients():
    import sympy as sp
    from experiments.offline_bootstrap import _regret_interval

    coefficients, action = [0.1, 0.7, -0.3], 0.2
    reference = {"candidate_value_lower": 0.8, "global_value_upper": 0.8000001}
    exact_value = sum(sp.Rational(c)*sp.Rational(action)**i for i, c in enumerate(coefficients))
    lower, upper = _regret_interval(coefficients, reference, action)
    assert sp.Rational(lower) <= sp.Rational(reference["candidate_value_lower"])-exact_value
    assert sp.Rational(upper) >= sp.Rational(reference["global_value_upper"])-exact_value


def test_force_rebuilds_snapshot_but_never_bypasses_source_hash(tmp_path):
    manifest, _, _, _ = _manifest(tmp_path)
    project = tmp_path/"results"/manifest.name
    _, arrays = prepare_snapshot(manifest, project)
    (project/"offline_snapshot.npz").write_bytes(b"damaged output")
    _, rebuilt = prepare_snapshot(manifest, project, force=True)
    for key in arrays:
        np.testing.assert_array_equal(arrays[key], rebuilt[key])
    with (tmp_path/"observations.npz").open("ab") as stream:
        stream.write(b"changed input")
    with pytest.raises(ValueError, match="checksum changed"):
        prepare_snapshot(manifest, project, force=True)
