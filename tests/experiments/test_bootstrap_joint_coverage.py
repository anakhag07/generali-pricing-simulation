import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import sympy as sp

from experiments.bootstrap_band import features
from experiments.bootstrap_band_continuous import VARIABLE, nonnegative_on_real_line
from experiments.bootstrap_band_sweep import dataset_streams
from experiments.bootstrap_joint_coverage import load_manifest, run_dataset, collect, build_launch_plan
from experiments.provenance import array_sha256

ROOT = Path(__file__).resolve().parents[2]


def payload():
    return json.loads((ROOT/"manifests/bootstrap_ols_joint_coverage.json").read_text())


def small_manifest(tmp_path):
    p = payload()
    p.update(datasets=2, datasets_per_task=1, settings=[{"N": 12, "B": 9}, {"N": 20, "B": 19}])
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(p))
    return load_manifest(path)


def test_direct_pairs_refits_exact_coverage_replay_and_collection(tmp_path):
    manifest = small_manifest(tmp_path)
    for index in range(2):
        saved = run_dataset(manifest, index, tmp_path)
        assert run_dataset(manifest, index, tmp_path) == saved
        stream_spec = {"seeds": manifest.payload["seeds"], "axes": {}, "baseline": {"N": 20, "B": 19}}
        seeds, x, epsilon, uniforms = dataset_streams(stream_spec, index)
        assert seeds == saved["seeds"]
        assert saved["uniform_shape"] == [19, 20]
        with np.load(saved["arrays"]["path"]) as arrays:
            np.testing.assert_array_equal(arrays["training_actions"], x)
            np.testing.assert_array_equal(arrays["standardized_observation_errors"], epsilon)
            for row in saved["rows"]:
                n, b, prefix = row["N"], row["B"], row["array_prefix"]
                design = features(x[:n])
                y = design @ [0., 5., -5.] + epsilon[:n]
                np.testing.assert_array_equal(arrays[prefix+"_y"], y)
                indices = (n*uniforms[:b,:n]).astype(np.int32)
                assert array_sha256(indices) == row["bootstrap_indices_sha256"]
                direct = np.array([np.linalg.lstsq(design[ids], y[ids], rcond=None)[0] for ids in indices])
                np.testing.assert_allclose(arrays[prefix+"_bootstrap_beta"], direct, atol=1e-11)
                bounds = arrays[prefix+"_bootstrap_supremum_bounds"]
                assert row["critical_value"] == np.quantile(bounds[:,1], .95, method="higher")
                poly = sp.Poly.from_list([sp.Rational(c) for c in arrays[prefix+"_containment_exact_coefficients"][::-1]], VARIABLE, domain=sp.QQ)
                assert nonnegative_on_real_line(poly) == row["covered"]
    collect(manifest, tmp_path)
    project = tmp_path/manifest.name
    summary = json.loads((project/"summary.json").read_text())
    assert len(summary["metrics"]) == 2
    for metric in summary["metrics"]:
        assert metric["datasets"] == 2
        assert metric["coverage"] == metric["covered_count"]/2
        assert metric["coverage_lower"] <= metric["coverage"] <= metric["coverage_upper"]
        assert "regret" not in metric and "pareto" not in metric
    assert Path(summary["plots"][0]["path"]).read_bytes().startswith(b"%PDF-")
    # Changed seeds cannot silently reuse a completed dataset.
    manifest.payload["seeds"]["bootstrap"] += 1
    with pytest.raises(ValueError, match="Cached joint sweep"):
        run_dataset(manifest, 0, tmp_path)


def test_grouped_array_tasks_cover_each_dataset_once(monkeypatch, tmp_path):
    manifest = small_manifest(tmp_path)
    manifest.payload.update(datasets=5, datasets_per_task=2)
    called = []
    monkeypatch.setattr("experiments.bootstrap_joint_coverage.run_dataset",
                        lambda m, i, root, force: called.append(i))
    plan = build_launch_plan(manifest, runs_root=tmp_path)
    assert plan.task_count == 3
    for task in range(plan.task_count):
        plan.run_task(task, SimpleNamespace(runs_root=tmp_path))
    assert called == list(range(5))
    assert plan.slurm_profile.cpus_per_task == 1 and plan.slurm_profile.gres is None
    with pytest.raises(IndexError):
        plan.run_task(3, SimpleNamespace(runs_root=tmp_path))


def test_shared_cli_and_production_joint_path(monkeypatch, tmp_path):
    from scripts import run_experiment_manifest as cli
    captured = []
    monkeypatch.setattr(cli, "run_launch_plan", lambda plan, **kwargs: captured.append(plan))
    cli.main([str(ROOT/"manifests/bootstrap_ols_joint_coverage.json"), "--runs-root", str(tmp_path)])
    assert captured[0].task_count == 200
    assert captured[0].default_array
    p = payload()
    assert p["datasets"] == 2000
    assert [s["N"] for s in p["settings"]] == [20,50,100,200,500,1000,2000,3000,5000]
    assert all(s["N"] == s["B"] for s in p["settings"])


@pytest.mark.parametrize("key,value", [
    ("settings", [{"N":20,"B":20},{"N":50,"B":10}]), ("datasets", 1),
    ("datasets_per_task", 0), ("bootstrap_method", "parametric"), ("delta", 1),
])
def test_manifest_rejects_invalid_contract(tmp_path, key, value):
    p = payload()
    p[key] = value
    path = tmp_path/"bad.json"
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        load_manifest(path)
