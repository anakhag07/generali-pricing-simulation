import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import norm, truncnorm

from experiments.envelopes import wild_study as study
from experiments.envelopes.core import TargetDomain
from experiments.envelopes.linear import LinearModel, check_linear_containment
from experiments.envelopes.wild import WildLinearDesign
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


def _truncated_payload(path, bound, context_dim=0):
    payload = json.loads(path.read_text())
    payload["design"] = {"context_dim": context_dim,
                         "distribution": "independent_truncated_standard_normal",
                         "truncation_bound": bound}
    payload["truth"]["coefficients"] = [1., *([.25] * context_dim), 1.]
    payload["predictor"]["coefficients"] = [1.2, *([.3] * context_dim), .8]
    payload["noise"] = {"type": "constant", "sigma_max": .5}
    return payload


@pytest.mark.parametrize("context_dim", [0, 2])
def test_truncated_bounds_share_uniforms_and_constant_noise(manifest_path, context_dim):
    snapshots = []
    for bound in (.5, 3.):
        payload = _truncated_payload(manifest_path, bound, context_dim)
        manifest_path.write_text(json.dumps(payload))
        manifest = study.load_manifest(manifest_path)
        snapshot, _, _ = study._environment(json.dumps(manifest.payload, sort_keys=True))
        snapshots.append(snapshot)
        for name in ("X", "actions"):
            assert np.all(np.abs(snapshot[name]) < bound)
        assert snapshot["X"].shape == (20, context_dim)
        np.testing.assert_array_equal(snapshot["sigma"], np.full(20, .5))
        assert snapshot["gp_cos"].shape == snapshot["gp_sin"].shape == (0,)
        np.testing.assert_array_equal(study.noise_scale(
            np.zeros((3, context_dim)), np.array([-1e8, .123456, 1e8]),
            payload["noise"], snapshot["gp_cos"], snapshot["gp_sin"]), [.5, .5, .5])
    for name, seed_name in (("X", "contexts"), ("actions", "actions")):
        uniforms = np.random.default_rng(payload["seeds"][seed_name]).random(snapshots[0][name].shape)
        for bound, snapshot in zip((.5, 3.), snapshots):
            np.testing.assert_allclose(truncnorm.cdf(snapshot[name], -bound, bound), uniforms,
                                       rtol=0, atol=1e-14)
    payload["seeds"]["profile"] += 100
    changed, _, _ = study._environment(json.dumps(payload, sort_keys=True))
    for name in ("X", "actions", "sigma", "gp_cos", "gp_sin"):
        np.testing.assert_array_equal(changed[name], snapshots[-1][name])


def test_gaussian_environment_preserves_original_draws(manifest_path):
    payload = study.load_manifest(manifest_path).payload
    snapshot, _, _ = study._environment(json.dumps(payload, sort_keys=True))
    seeds = payload["seeds"]
    np.testing.assert_array_equal(snapshot["X"], np.random.default_rng(seeds["contexts"]).normal(size=(20, 3)))
    np.testing.assert_array_equal(snapshot["actions"], np.random.default_rng(seeds["actions"]).normal(size=20))
    coefficients = np.random.default_rng(seeds["profile"]).normal(size=(2, payload["noise"]["rank"]))
    np.testing.assert_array_equal(snapshot["gp_cos"], coefficients[0])
    np.testing.assert_array_equal(snapshot["gp_sin"], coefficients[1])


@pytest.mark.parametrize("context_dim", [0, 2])
@pytest.mark.parametrize("bound", [10., 100., 1000., None])
def test_wide_and_unbounded_designs_match_paired_normal_quantiles(manifest_path, context_dim, bound):
    payload = _truncated_payload(manifest_path, bound, context_dim)
    manifest_path.write_text(json.dumps(payload))
    manifest = study.load_manifest(manifest_path)
    snapshot, _, _ = study._environment(json.dumps(manifest.payload, sort_keys=True))
    for name, seed_name in (("X", "contexts"), ("actions", "actions")):
        uniforms = np.random.default_rng(payload["seeds"][seed_name]).random(snapshot[name].shape)
        np.testing.assert_allclose(snapshot[name], norm.ppf(uniforms), rtol=0, atol=1e-14)
        assert np.isfinite(snapshot[name]).all()
    np.testing.assert_array_equal(snapshot["sigma"], np.full(20, .5))


def test_missing_truncation_bound_is_not_treated_as_unbounded(manifest_path):
    payload = _truncated_payload(manifest_path, None)
    del payload["design"]["truncation_bound"]
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="explicit truncation_bound"):
        study.load_manifest(manifest_path)


def test_scalar_leverage_and_bootstrap_are_affine_invariant():
    actions = np.array([-2., -.9, -.3, 0., .2, .4, .8, 1., 3.])
    contexts = np.empty((len(actions), 0))
    original = WildLinearDesign(contexts, actions)
    transformed = WildLinearDesign(contexts, 7. - 2.5*actions)
    centered = actions-actions.mean()
    leverage = 1/len(actions) + centered**2 / (centered @ centered)
    np.testing.assert_allclose(original.leverage, leverage, atol=1e-15)
    np.testing.assert_allclose(transformed.leverage, leverage, atol=1e-15)
    errors = np.array([[.1, -.2], [-.3, .5], [0., 0.]])
    expected = np.sum((errors @ original.design.T)**2, axis=1)
    np.testing.assert_allclose(original.statistics(errors)**2, expected, atol=1e-14)
    # Full bootstrap statistics remain unchanged under the same affine basis
    # change; centering is therefore unnecessary for a coverage comparison.
    y = np.array([1., .4, .2, .6, .9, .1, 2., 1.5, 3.])
    signs = np.random.default_rng(713).choice([-1, 1], size=(19, len(actions)))
    _, original_errors = original.bootstrap_errors(y, signs)
    _, transformed_errors = transformed.bootstrap_errors(y, signs)
    np.testing.assert_allclose(original.statistics(original_errors),
                               transformed.statistics(transformed_errors), atol=1e-14)
    for beta, critical in (([1.1, 1.1], .3), ([3., -.5], .1), ([1., 1.], 1.)):
        truth, predictor = LinearModel([1., 1.]), LinearModel([1.2, .8])
        envelope = original.envelope(predictor, beta, critical)
        exact = check_linear_containment(envelope, truth, TargetDomain(contexts[:1]))
        assert original.transferred_contains(beta, predictor.coefficients, truth.coefficients,
                                              critical) == (exact.status == "covered")


def test_zero_context_repetitions_keep_pairing_and_all_real_statistic(manifest_path, tmp_path):
    records, saved = [], []
    for bound in (.5, 3.):
        payload = _truncated_payload(manifest_path, bound)
        payload["name"] = f"truncated-{bound}".replace(".", "p")
        manifest_path.write_text(json.dumps(payload))
        manifest = study.load_manifest(manifest_path)
        record = study.run_repetition(manifest, 0, tmp_path / "runs")
        records.append(record)
        snapshot, designs, _ = study._environment(json.dumps(payload, sort_keys=True))
        with np.load(record["arrays"]["path"]) as arrays:
            saved.append(arrays["innovations"])
            for ni, design in enumerate(designs):
                error = arrays["beta"][ni] - payload["truth"]["coefficients"]
                independent = np.linalg.norm(design.design @ error)
                assert arrays["truth_statistics"][ni] == pytest.approx(independent)
                for bi, b in enumerate(payload["B_values"]):
                    for di, delta in enumerate(payload["deltas"]):
                        critical = np.quantile(arrays["bootstrap_statistics"][ni, :b], 1-delta, method="higher")
                        assert arrays["critical"][ni, bi, di, 1] == critical
                        assert arrays["auxiliary_covered"][ni, bi, di, 1] == (independent <= critical)
            example = study._example(payload, snapshot, designs, arrays)
            assert example["truth"].shape == (20, 7)
            np.testing.assert_allclose(example["truth"], 1. + np.broadcast_to(example["actions_grid"], (20, 7)))
    assert records[0]["seeds"] == records[1]["seeds"]
    assert records[0]["multiplier_sha256"] == records[1]["multiplier_sha256"]
    np.testing.assert_array_equal(saved[0], saved[1])


@pytest.mark.parametrize("bound", [0., -1., float("inf"), float("nan"), True])
def test_invalid_truncation_bounds_fail_before_sampling(manifest_path, bound):
    manifest_path.write_text(json.dumps(_truncated_payload(manifest_path, bound)))
    with pytest.raises(ValueError, match="truncation_bound"):
        study.load_manifest(manifest_path)


@pytest.mark.parametrize("sigma", [0., -1., float("inf"), True])
def test_invalid_constant_noise_fails_before_sampling(manifest_path, sigma):
    payload = _truncated_payload(manifest_path, 1.)
    payload["noise"]["sigma_max"] = sigma
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="sigma_max"):
        study.load_manifest(manifest_path)
