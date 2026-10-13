import json
from pathlib import Path

import numpy as np
import pytest

from experiments.envelopes import wild_study as study
from experiments.seeds import derive_seed


@pytest.fixture
def manifest_path(tmp_path):
    original = Path(__file__).parents[2] / "manifests/uncertainty-envelopes-wild-bootstrap.json"
    payload = json.loads(original.read_text())
    payload.update(name="wild-test", n_values=[12, 20], B_values=[9, 19],
                   deltas=[.2, .1], baseline={"n": 20, "B": 19, "delta": .1},
                   repetitions=4, repetitions_per_task=2)
    payload["diagnostics"]["actions"]["count"] = 7
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_profile_has_global_cap_and_exact_deterministic_offgrid_evaluation(manifest_path):
    manifest = study.load_manifest(manifest_path)
    p = manifest.payload
    snapshot, _, _ = study._environment(json.dumps(p, sort_keys=True))
    X = np.array([[0., 0., 0.], [1e6, -2e6, 3e6], [-.25, .37, .66]])
    a = np.array([-1e9, .123456789, 1e9])
    scale = study.noise_scale(X, a, p["noise"], snapshot["gp_cos"], snapshot["gp_sin"])
    assert np.all(scale >= p["noise"]["sigma_min"])
    assert np.all(scale <= p["noise"]["sigma_max"])
    np.testing.assert_array_equal(scale, study.noise_scale(X, a, p["noise"], snapshot["gp_cos"], snapshot["gp_sin"]))
    np.testing.assert_array_equal(snapshot["sigma"], study.noise_scale(snapshot["X"], snapshot["actions"],
                                  p["noise"], snapshot["gp_cos"], snapshot["gp_sin"]))


def test_repetition_replays_paired_prefixes_and_quantiles(manifest_path, tmp_path):
    manifest = study.load_manifest(manifest_path)
    record = study.run_repetition(manifest, 0, tmp_path / "runs")
    p = manifest.payload
    with np.load(record["arrays"]["path"]) as arrays:
        snapshot, designs, _ = study._environment(json.dumps(p, sort_keys=True))
        innovations = np.random.default_rng(derive_seed(p["seeds"]["observations"], "repetition:0")).normal(size=20)
        multipliers = 2*np.random.default_rng(derive_seed(p["seeds"]["method"], "repetition:0")).integers(0, 2, size=(19, 20), dtype=np.int8)-1
        np.testing.assert_array_equal(arrays["innovations"], innovations)
        for ni, n in enumerate(p["n_values"]):
            y = designs[ni].design @ p["truth"]["coefficients"] + snapshot["sigma"][:n]*innovations[:n]
            beta, errors = designs[ni].bootstrap_errors(y, multipliers[:, :n])
            np.testing.assert_array_equal(arrays["beta"][ni], beta)
            np.testing.assert_array_equal(arrays["bootstrap_statistics"][ni], designs[ni].statistics(errors))
            for bi, b in enumerate(p["B_values"]):
                for di, delta in enumerate(p["deltas"]):
                    assert arrays["critical"][ni, bi, di, 1] == np.quantile(arrays["bootstrap_statistics"][ni, :b], 1-delta, method="higher")
        assert np.all(~arrays["auxiliary_covered"] | arrays["transferred_covered"])
    assert study.run_repetition(manifest, 0, tmp_path / "runs") == record
    other = study.run_repetition(manifest, 1, tmp_path / "runs")
    assert other["seeds"] != record["seeds"]
    assert other["multiplier_sha256"] != record["multiplier_sha256"]


def test_collect_uses_dataset_joint_events_and_saves_first_example(manifest_path, tmp_path, monkeypatch):
    from experiments.envelopes import wild_reporting
    monkeypatch.setattr(wild_reporting, "write_reports", lambda *args: [])
    manifest = study.load_manifest(manifest_path)
    root = tmp_path / "runs"
    with pytest.raises(FileNotFoundError):
        study.collect(manifest, root)
    arrays = []
    for index in range(4):
        record = study.run_repetition(manifest, index, root)
        with np.load(record["arrays"]["path"]) as saved:
            arrays.append({key: saved[key] for key in saved.files})
    summary = study.collect(manifest, root)
    row = next(r for r in summary["metrics"] if r["n"] == 20 and r["B"] == 19 and r["delta"] == .1 and r["method"] == "wild_bootstrap")
    expected = sum(a["auxiliary_covered"][1, 1, 1, 1] for a in arrays)/4
    assert row["auxiliary_coverage"] == expected
    assert row["auxiliary_lower"] <= expected <= row["auxiliary_upper"]
    assert row["width_ratio"] == pytest.approx(np.mean([a["critical"][1, 1, 1, 1]/a["critical"][1, 1, 1, 0] for a in arrays]))
    assert Path(summary["example"]["path"]).exists()
    assert len(summary["repetitions"]) == 4
    assert Path(summary["documentation"]["path"]).exists()


def test_changed_contract_and_corrupt_artifacts_are_rejected(manifest_path, tmp_path):
    manifest = study.load_manifest(manifest_path)
    root = tmp_path / "runs"
    record = study.run_repetition(manifest, 0, root)
    Path(record["arrays"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Stale repetition"):
        study.run_repetition(manifest, 0, root)
    payload = json.loads(manifest_path.read_text())
    payload["noise"]["amplitude"] = 3.
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="contract changed"):
        study.run_repetition(study.load_manifest(manifest_path), 0, root)


@pytest.mark.parametrize("key,value", [
    ("n_values", [5, 20]), ("B_values", [19, 9]), ("deltas", [0, .1]),
    ("repetitions", True), ("target", {"contexts": "calibration", "actions": "grid"}),
    ("bootstrap", {"multipliers": "rademacher", "residuals": "HC2", "quantile": "higher"}),
    ("seeds", {"contexts": 1, "actions": 1, "observations": 2, "method": 3, "profile": 4}),
])
def test_invalid_contracts_fail_before_sampling(manifest_path, key, value):
    payload = json.loads(manifest_path.read_text())
    payload[key] = value
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        study.load_manifest(manifest_path)


def test_launcher_groups_repetitions_and_rejects_invalid_task(manifest_path, tmp_path):
    manifest = study.load_manifest(manifest_path)
    plan = study.build_launch_plan(manifest, runs_root=tmp_path)
    assert plan.task_count == 2
    with pytest.raises(IndexError):
        plan.run_task(2, None)


def test_manifest_entrypoint_routes_to_wild_study(manifest_path, monkeypatch):
    from scripts import run_experiment_manifest as runner
    seen = {}
    def launch(plan, **kwargs):
        seen.update(plan=plan, **kwargs)
    monkeypatch.setattr(runner, "run_launch_plan", launch)
    runner.main([str(manifest_path), "--launch", "local"])
    assert seen["plan"].name == "wild-test"
    assert seen["plan"].task_count == 2
