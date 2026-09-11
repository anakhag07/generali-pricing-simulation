from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.bootstrap_band import (
    bootstrap_band, collect_outputs, evaluate_band, features,
    load_bootstrap_band_manifest, run_case,
)

ROOT = Path(__file__).resolve().parents[2]


def dataset():
    rng = np.random.default_rng(71)
    x = rng.normal(size=40)
    y = features(x) @ [0, 5, -5] + rng.normal(size=len(x))
    return x, y, np.linspace(0, 1, 51)


def test_ols_se_and_each_bootstrap_refit_match_independent_lstsq():
    x, y, grid = dataset()
    result = bootstrap_band(x, y, grid, draws=29, delta=0.05, bootstrap_seed=18)
    p, evaluation = features(x), features(grid)
    beta = np.linalg.lstsq(p, y, rcond=None)[0]
    sigma = np.sqrt(np.sum((y - p @ beta)**2) / (len(x) - 3))
    cov = np.linalg.inv(p.T @ p)
    se = sigma * np.sqrt(np.einsum("ij,jk,ik->i", evaluation, cov, evaluation))
    np.testing.assert_allclose(result["beta_hat"], beta, atol=1e-12)
    np.testing.assert_allclose(result["standard_error"], se, atol=1e-12)
    z = np.random.default_rng(18).normal(size=(29, len(x)))
    independent_fits = np.stack([
        np.linalg.lstsq(p, p @ beta + sigma * draw, rcond=None)[0] for draw in z
    ])
    np.testing.assert_allclose(result["bootstrap_beta"], independent_fits, atol=1e-12)
    maxima = np.max(np.abs((independent_fits - beta) @ evaluation.T) / se, axis=1)
    np.testing.assert_allclose(result["bootstrap_maxima"], maxima, atol=1e-11)
    assert result["critical_value"] == pytest.approx(np.quantile(maxima, .95, method="higher"))


def test_band_uses_original_scale_and_is_equivariant_to_response_scale():
    x, y, grid = dataset()
    first = bootstrap_band(x, y, grid, draws=31, delta=.05, bootstrap_seed=18)
    second = bootstrap_band(x, 3*y, grid, draws=31, delta=.05, bootstrap_seed=18)
    np.testing.assert_allclose(second["bootstrap_maxima"], first["bootstrap_maxima"], atol=1e-11)
    np.testing.assert_allclose(second["radius"], 3*first["radius"], atol=1e-11)
    np.testing.assert_allclose(second["sigma2_hat"], 9*first["sigma2_hat"])
    np.testing.assert_allclose(first["radius"], first["critical_value"] * first["standard_error"])


def test_coverage_checks_every_point_and_truth_cannot_change_calibration():
    x, y, grid = dataset()
    band = bootstrap_band(x, y, grid, draws=31, delta=.05, bootstrap_seed=18)
    original = band["bootstrap_maxima"].copy()
    assert evaluate_band(band, band["fitted"])["simultaneous_covered"]
    truth = band["fitted"].copy()
    truth[-1] += 2 * band["radius"][-1]
    evaluated = evaluate_band(band, truth)
    assert evaluated["simultaneous_covered"] is False
    assert evaluated["fraction_grid_covered"] == pytest.approx(50/51)
    np.testing.assert_array_equal(original, band["bootstrap_maxima"])


def test_nested_grid_maxima_cannot_decrease():
    x, y, grid = dataset()
    coarse = bootstrap_band(x, y, grid[::5], draws=29, delta=.05, bootstrap_seed=18)
    dense = bootstrap_band(x, y, grid, draws=29, delta=.05, bootstrap_seed=18)
    assert np.all(dense["bootstrap_maxima"] >= coarse["bootstrap_maxima"] - 1e-12)
    assert dense["critical_value"] >= coarse["critical_value"] - 1e-12


@pytest.mark.parametrize("x,y", [
    (np.ones(5), np.arange(5.)),
    (np.arange(3.), np.arange(3.)),
    (np.arange(6.), np.arange(6.)**2),
])
def test_rejects_unidentified_or_zero_variance_fits(x, y):
    with pytest.raises(ValueError):
        bootstrap_band(x, y, np.linspace(0, 1, 5), draws=10, delta=.05, bootstrap_seed=0)


def test_manifest_runner_replay_and_pdf_collection(tmp_path):
    payload = json.loads((ROOT / "manifests/bootstrap_ols_grid_band.json").read_text())
    payload["stage1"] = {"n": 20, "noise_std": 1.0}
    payload["stage2"] = {"sample_sizes": [10, 20], "noise_stds": [1.0], "datasets": 3}
    payload["bootstrap"]["draws"] = 19
    payload["grid"]["count"] = 21
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload))
    manifest = load_bootstrap_band_manifest(manifest_path)
    for i in range(len(manifest.cases())):
        assert not run_case(manifest, i, runs_root=tmp_path, force=False)["skipped"]
    assert run_case(manifest, 0, runs_root=tmp_path, force=False)["skipped"]
    collect_outputs(manifest, runs_root=tmp_path)
    project = tmp_path / manifest.name
    summary = json.loads((project / "summary.json").read_text())
    assert isinstance(summary["stage1"]["simultaneous_covered"], bool)
    assert len(summary["stage2"]) == 2
    assert all(row["datasets"] == 3 for row in summary["stage2"])
    assert len(summary["plots"]) == 5
    for record in summary["plots"]:
        assert Path(record["path"]).read_bytes().startswith(b"%PDF-")
    assert not list(project.rglob("*.png"))
    raw = project / "stage1-n20-sd1/draws.npz"
    with np.load(raw) as data:
        original = data["bootstrap_maxima"].copy()
        assert data["bootstrap_beta"].shape == (1, 19, 3)
    run_case(manifest, 0, runs_root=tmp_path, force=True)
    with np.load(raw) as data:
        np.testing.assert_array_equal(original, data["bootstrap_maxima"])
    seeds = []
    for case in project.glob("stage*/summary.json"):
        for row in json.loads(case.read_text())["rows"]:
            seeds.extend(row[key] for key in ("design_seed", "observation_noise_seed", "bootstrap_seed"))
    assert len(seeds) == len(set(seeds))
    raw.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Existing outputs differ"):
        run_case(manifest, 0, runs_root=tmp_path, force=False)


def test_shared_cli_dispatch(monkeypatch, tmp_path):
    from scripts import run_experiment_manifest as cli
    captured = []
    monkeypatch.setattr(cli, "run_launch_plan", lambda plan, **kwargs: captured.append(plan))
    cli.main([str(ROOT / "manifests/bootstrap_ols_grid_band.json"), "--launch", "local", "--runs-root", str(tmp_path)])
    assert captured[0].task_count == 10
    assert captured[0].default_launch == "local"


@pytest.mark.parametrize("section,key,value", [
    ("stage1", "n", 3), ("bootstrap", "draws", 0), ("bootstrap", "delta", 1),
    ("grid", "upper", 2), ("stage2", "noise_stds", [-1]),
])
def test_manifest_rejects_invalid_inputs(tmp_path, section, key, value):
    payload = json.loads((ROOT / "manifests/bootstrap_ols_grid_band.json").read_text())
    payload[section][key] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_bootstrap_band_manifest(path)
