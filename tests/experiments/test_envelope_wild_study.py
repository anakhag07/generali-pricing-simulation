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
    original = Path(__file__).parents[2] / "scratch/retired_experiments/wild_bootstrap_diagnostics/manifests/uncertainty-envelopes-wild-bootstrap.json"
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


@pytest.mark.parametrize("variant", ["centered_exponential", "independent_standardized_t3"])
def test_distribution_perturbations_preserve_pairing(manifest_path, tmp_path, variant):
    from scipy.special import log_ndtr
    from scipy.stats import t
    baseline = study.load_manifest(manifest_path)
    original = study.run_repetition(baseline, 0, tmp_path / "runs")
    base_snapshot, _, _ = study._environment(json.dumps(baseline.payload, sort_keys=True))
    p = json.loads(manifest_path.read_text())
    p["name"] += "-variant"
    if variant == "centered_exponential":
        p["noise"]["innovation_distribution"] = variant
    else:
        p["design"]["distribution"] = variant
    manifest_path.write_text(json.dumps(p))
    manifest = study.load_manifest(manifest_path)
    changed = study.run_repetition(manifest, 0, tmp_path / "runs")
    snapshot, designs, _ = study._environment(json.dumps(manifest.payload, sort_keys=True))
    assert changed["multiplier_sha256"] == original["multiplier_sha256"]
    assert changed["seeds"] == original["seeds"]
    for key in ("gp_cos", "gp_sin", "actions_grid"):
        np.testing.assert_array_equal(snapshot[key], base_snapshot[key])
    with np.load(changed["arrays"]["path"]) as arrays, np.load(original["arrays"]["path"]) as base_arrays:
        if variant == "centered_exponential":
            np.testing.assert_array_equal(arrays["innovations"], -log_ndtr(-base_arrays["innovations"])-1)
            for key in ("X", "actions", "sigma"):
                np.testing.assert_array_equal(snapshot[key], base_snapshot[key])
        else:
            np.testing.assert_array_equal(arrays["innovations"], base_arrays["innovations"])
            for key in ("X", "actions"):
                np.testing.assert_allclose(snapshot[key], t.ppf(norm.cdf(base_snapshot[key]), 3)/np.sqrt(3), rtol=1e-10)
            assert not np.array_equal(snapshot["sigma"], base_snapshot["sigma"])
            np.testing.assert_array_equal(snapshot["sigma"], study.noise_scale(snapshot["X"], snapshot["actions"],
                                         p["noise"], snapshot["gp_cos"], snapshot["gp_sin"]))
        assert np.all(~arrays["auxiliary_covered"] | arrays["transferred_covered"])
    for design, n in zip(designs, p["n_values"]):
        assert np.linalg.matrix_rank(design.design) == design.design.shape[1]
        np.testing.assert_array_equal(design.design[:, 1:-1], snapshot["X"][:n])
    assert study.run_repetition(manifest, 0, tmp_path / "runs") == changed


def test_t3_transform_retains_extreme_tails_without_clipping():
    z = np.array([-12., -9., -1., 0., 1., 9., 12.])
    actual = study._standardized_t3(z)
    assert np.isfinite(actual).all()
    assert np.all(np.diff(actual) > 0)
    np.testing.assert_array_equal(actual, -actual[::-1])
    assert actual[-1] > actual[-2] * 1000


def test_unknown_innovation_law_fails_before_sampling(manifest_path):
    p = json.loads(manifest_path.read_text())
    p["noise"]["innovation_distribution"] = "cauchy"
    manifest_path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="innovations"):
        study.load_manifest(manifest_path)


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
            beta, errors = designs[ni].bootstrap_errors(y, multipliers[:, :n],
                         predictor=designs[ni].design @ p["predictor"]["coefficients"])
            np.testing.assert_array_equal(arrays["beta"][ni], beta)
            np.testing.assert_array_equal(arrays["bootstrap_statistics"][ni], designs[ni].statistics(errors))
            for bi, b in enumerate(p["B_values"]):
                for di, delta in enumerate(p["deltas"]):
                    assert arrays["critical"][ni, bi, di, 1] == np.sort(arrays["bootstrap_statistics"][ni, :b])[int(np.ceil(b*(1-delta)))-1]
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
                error = arrays["beta"][ni] - (np.asarray(payload["truth"]["coefficients"])-payload["predictor"]["coefficients"])
                independent = np.linalg.norm(design.design @ error)
                assert arrays["truth_statistics"][ni] == pytest.approx(independent)
                for bi, b in enumerate(payload["B_values"]):
                    for di, delta in enumerate(payload["deltas"]):
                        critical = np.sort(arrays["bootstrap_statistics"][ni, :b])[int(np.ceil(b*(1-delta)))-1]
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


@pytest.mark.parametrize("override", [
    None, [], {"index": 0}, {"index": 0, "value": 10, "extra": 1},
    {"index": True, "value": 10}, {"index": 0., "value": 10},
    {"index": -1, "value": 10}, {"index": 12, "value": 10},
    {"index": 0, "value": True}, {"index": 0, "value": "10"},
    {"index": 0, "value": float("inf")}, {"index": 0, "value": float("nan")},
])
def test_invalid_action_override_fails_before_sampling(manifest_path, override):
    payload = _truncated_payload(manifest_path, 3.)
    payload["design"]["action_override"] = override
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="action_override"):
        study.load_manifest(manifest_path)


def test_moved_observation_preserves_bulk_and_matches_leverage_formula(manifest_path):
    payload = _truncated_payload(manifest_path, 3.)
    payload["n_values"] = [300]
    payload["baseline"]["n"] = 300
    original, _, _ = study._environment(json.dumps(payload, sort_keys=True))
    bulk = original["actions"][1:]
    m, n = len(bulk), len(original["actions"])
    bulk_ss = np.sum((bulk-bulk.mean())**2)
    maxima = []
    for location in (3., 10., 20., 40., 100., 1000.):
        payload["design"]["action_override"] = {"index": 0, "value": location}
        manifest_path.write_text(json.dumps(payload))
        manifest = study.load_manifest(manifest_path)
        snapshot, (design,), _ = study._environment(json.dumps(manifest.payload, sort_keys=True))
        assert snapshot["actions"][0] == location
        np.testing.assert_array_equal(snapshot["actions"][1:], bulk)
        assert np.all(np.abs(bulk) < 3.)
        np.testing.assert_array_equal(snapshot["X"], original["X"])
        np.testing.assert_array_equal(snapshot["sigma"], original["sigma"])
        squared_distance = (location-bulk.mean())**2
        expected = 1/n + (m/n)**2*squared_distance/(bulk_ss+(m/n)*squared_distance)
        assert design.leverage[0] == pytest.approx(expected, abs=1e-14)
        assert np.all((design.leverage >= 0) & (design.leverage <= 1))
        maxima.append(np.max(design.leverage))
        errors = np.array([[.1, -.2], [-.3, .5], [0., 0.]])
        np.testing.assert_allclose(design.statistics(errors)**2,
                                   np.sum((errors @ design.design.T)**2, axis=1), rtol=1e-12)
    assert np.all(np.diff(maxima) > 0)
    assert maxima[-1] > .999


def test_action_override_precedes_location_dependent_noise(manifest_path):
    payload = study.load_manifest(manifest_path).payload
    original, _, _ = study._environment(json.dumps(payload, sort_keys=True))
    payload["design"]["action_override"] = {"index": 0, "value": 10.}
    manifest_path.write_text(json.dumps(payload))
    manifest = study.load_manifest(manifest_path)
    snapshot, _, _ = study._environment(json.dumps(manifest.payload, sort_keys=True))
    np.testing.assert_array_equal(snapshot["X"], original["X"])
    np.testing.assert_array_equal(snapshot["actions"][1:], original["actions"][1:])
    np.testing.assert_array_equal(snapshot["sigma"][1:], original["sigma"][1:])
    np.testing.assert_array_equal(snapshot["sigma"], study.noise_scale(
        snapshot["X"], snapshot["actions"], payload["noise"], snapshot["gp_cos"], snapshot["gp_sin"]))
    assert snapshot["sigma"][0] != original["sigma"][0]


@pytest.mark.parametrize("sigma", [0., -1., 1.01, float("inf"), float("nan"), True, "0.7"])
def test_moved_observation_scale_rejects_invalid_or_unbounded_values(manifest_path, sigma):
    payload = json.loads(manifest_path.read_text())
    payload["design"]["action_override"] = {"index": 0, "value": 40.}
    payload["noise"]["moved_observation_sigma"] = sigma
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="moved_observation_sigma"):
        study.load_manifest(manifest_path)


def test_moved_observation_scale_requires_a_moved_row(manifest_path):
    payload = json.loads(manifest_path.read_text())
    payload["noise"]["moved_observation_sigma"] = .7
    manifest_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="requires design.action_override"):
        study.load_manifest(manifest_path)


def test_fixed_moved_scale_changes_only_noise_at_selected_row(manifest_path):
    payload = study.load_manifest(manifest_path).payload
    original, _, _ = study._environment(json.dumps(payload, sort_keys=True))
    index = 2  # Check the configured row, rather than accidentally hard-coding row 0.
    frozen_sigma = float(original["sigma"][index])
    for location in (3., 40., 1000.):
        payload["design"]["action_override"] = {"index": index, "value": location}
        natural, natural_designs, _ = study._environment(json.dumps(payload, sort_keys=True))
        fixed = json.loads(json.dumps(payload))
        fixed["noise"]["moved_observation_sigma"] = frozen_sigma
        manifest_path.write_text(json.dumps(fixed))
        manifest = study.load_manifest(manifest_path)
        snapshot, designs, _ = study._environment(json.dumps(manifest.payload, sort_keys=True))
        for key in ("X", "actions", "gp_cos", "gp_sin", "actions_grid"):
            np.testing.assert_array_equal(snapshot[key], natural[key])
        np.testing.assert_array_equal(snapshot["sigma"], original["sigma"])
        assert snapshot["sigma"][index] != natural["sigma"][index]
        for design, natural_design in zip(designs, natural_designs):
            np.testing.assert_array_equal(design.leverage, natural_design.leverage)
        np.testing.assert_array_equal(study.noise_scale(
            snapshot["X"], snapshot["actions"], fixed["noise"], snapshot["gp_cos"], snapshot["gp_sin"]),
            natural["sigma"])


def test_fixed_moved_scale_pairs_locations_but_draws_fresh_repetitions(manifest_path, tmp_path):
    payload = study.load_manifest(manifest_path).payload
    payload["noise"]["moved_observation_sigma"] = .7
    saved = {}
    for location in (3., 1000.):
        payload["name"] = f"fixed-noise-{int(location)}"
        payload["design"]["action_override"] = {"index": 0, "value": location}
        manifest_path.write_text(json.dumps(payload))
        manifest = study.load_manifest(manifest_path)
        snapshot, designs, _ = study._environment(json.dumps(manifest.payload, sort_keys=True))
        for repetition in (0, 1):
            record = study.run_repetition(manifest, repetition, tmp_path / "runs")
            with np.load(record["arrays"]["path"]) as arrays:
                eps = snapshot["sigma"] * arrays["innovations"]
                for ni, (n, design) in enumerate(zip(payload["n_values"], designs)):
                    y = design.design @ payload["truth"]["coefficients"] + eps[:n]
                    fitted = design.fit(y-design.design @ payload["predictor"]["coefficients"])
                    np.testing.assert_allclose(arrays["beta"][ni], fitted, rtol=0, atol=1e-12)
                saved[location, repetition] = (eps.copy(), record)
    for repetition in (0, 1):
        left, right = saved[3., repetition], saved[1000., repetition]
        np.testing.assert_array_equal(left[0], right[0])
        assert left[1]["seeds"] == right[1]["seeds"]
        assert left[1]["multiplier_sha256"] == right[1]["multiplier_sha256"]
    assert not np.array_equal(saved[3., 0][0], saved[3., 1][0])
    assert saved[3., 0][0][0] != saved[3., 1][0][0]


def test_moved_observation_preserves_response_and_multiplier_pairing(manifest_path, tmp_path):
    records, saved = [], []
    for location in (3., 1000.):
        payload = _truncated_payload(manifest_path, 3.)
        payload["name"] = f"moved-{int(location)}"
        payload["design"]["action_override"] = {"index": 0, "value": location}
        manifest_path.write_text(json.dumps(payload))
        manifest = study.load_manifest(manifest_path)
        record = study.run_repetition(manifest, 0, tmp_path / "runs")
        records.append(record)
        snapshot, designs, _ = study._environment(json.dumps(payload, sort_keys=True))
        with np.load(record["arrays"]["path"]) as arrays:
            saved.append(arrays["innovations"])
            for ni, (n, design) in enumerate(zip(payload["n_values"], designs)):
                y = 1. + snapshot["actions"][:n] + .5*arrays["innovations"][:n]
                np.testing.assert_allclose(arrays["beta"][ni], design.fit(y-design.design @ payload["predictor"]["coefficients"]), rtol=0, atol=1e-14)
                error = arrays["beta"][ni] - (np.asarray(payload["truth"]["coefficients"])-payload["predictor"]["coefficients"])
                assert arrays["truth_statistics"][ni] == pytest.approx(np.linalg.norm(design.design @ error))
    assert records[0]["seeds"] == records[1]["seeds"]
    assert records[0]["multiplier_sha256"] == records[1]["multiplier_sha256"]
    np.testing.assert_array_equal(saved[0], saved[1])


def test_quadratic_residual_end_to_end_and_exact_empirical_rank(tmp_path):
    original = Path(__file__).parents[2] / 'scratch/retired_experiments/wild_bootstrap_diagnostics/manifests/uncertainty-envelopes-quadratic-residual-outlier.json'
    payload = json.loads(original.read_text())
    payload.update(name='quadratic-smoke', n_values=[12], B_values=[10], deltas=[.2],
                   baseline={'n': 12, 'B': 10, 'delta': .2}, repetitions=2)
    payload['diagnostics']['actions']['count'] = 7
    path = tmp_path / 'quadratic.json'
    path.write_text(json.dumps(payload))
    manifest = study.load_manifest(path)
    for index in range(2):
        record = study.run_repetition(manifest, index, tmp_path)
        with np.load(record['arrays']['path']) as arrays:
            assert arrays['beta'].shape == (1, 3)
            assert arrays['critical'][0, 0, 0, 1] == np.sort(arrays['bootstrap_statistics'][0])[7]
            assert np.all(arrays['bootstrap_statistics_lower'] <= arrays['bootstrap_statistics'])
            assert np.all(~arrays['auxiliary_covered'] | arrays['transferred_covered'])
    summary = study.collect(manifest, tmp_path)
    assert summary['band_target'] == 'residual'
    assert len(summary['plots']) == 7
    with np.load(summary['example']['path']) as example:
        np.testing.assert_allclose(example['residual_truth'], example['truth']-example['predictor'])
        np.testing.assert_allclose(example['bootstrap_lower'], np.minimum(example['predictor'], example['auxiliary']-example['bootstrap_width']))
        np.testing.assert_allclose(example['bootstrap_upper'], np.maximum(example['predictor'], example['auxiliary']+example['bootstrap_width']))
    assert all('moved_radius_lower' in row for row in summary['metrics'])
    for record in summary['plots']:
        assert Path(record['path']).read_bytes().startswith(b'%PDF-')


@pytest.mark.parametrize('mutation', ['unequal_axes', 'crossed_baseline', 'unknown_mode'])
def test_paired_schedule_rejects_inconsistent_settings(manifest_path, mutation):
    p = json.loads(manifest_path.read_text())
    p['sweep_mode'] = 'paired'
    if mutation == 'unequal_axes':
        p['B_values'] = [9, 19, 29]
    elif mutation == 'crossed_baseline':
        p['baseline']['B'] = 9
    else:
        p['sweep_mode'] = 'unknown'
    manifest_path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        study.load_manifest(manifest_path)


def test_paired_sweep_replays_only_paired_prefixes_and_fixed_context_slices(manifest_path, tmp_path):
    from scratch.retired_experiments.studies.envelopes.diagnostics import _slice_indices
    p = json.loads(manifest_path.read_text())
    p['sweep_mode'] = 'paired'
    manifest_path.write_text(json.dumps(p))
    manifest = study.load_manifest(manifest_path)
    p = manifest.payload
    root = tmp_path / 'runs'
    first = None
    for index in range(p['repetitions']):
        record = study.run_repetition(manifest, index, root)
        with np.load(record['arrays']['path']) as arrays:
            np.testing.assert_array_equal(arrays['active_conditions'], np.eye(2, dtype=bool))
            assert np.isnan(arrays['critical'][0, 1]).all()
            assert np.isnan(arrays['critical'][1, 0]).all()
            assert np.isnan(arrays['bootstrap_statistics'][0, 9:]).all()
            if index == 0:
                first = {key: arrays[key] for key in arrays.files}
    snapshot, designs, _ = study._environment(json.dumps(p, sort_keys=True))
    signs = 2*np.random.default_rng(derive_seed(p['seeds']['method'], 'repetition:0')).integers(0, 2, (19, 20), dtype=np.int8)-1
    for ni, (n, b) in enumerate(zip(p['n_values'], p['B_values'])):
        design = designs[ni]
        y = design.design @ p['truth']['coefficients'] + snapshot['sigma'][:n]*first['innovations'][:n]
        beta, errors = design.bootstrap_errors(y, signs[:b, :n], predictor=design.design@p['predictor']['coefficients'])
        np.testing.assert_array_equal(beta, first['beta'][ni])
        np.testing.assert_array_equal(design.statistics(errors), first['bootstrap_statistics'][ni, :b])
    summary = study.collect(manifest, root)
    assert {(r['n'], r['B']) for r in summary['metrics']} == {(12, 9), (20, 19)}
    assert len(summary['metrics']) == 8
    with np.load(summary['paired_examples']['path']) as examples:
        indices = _slice_indices(snapshot['X'][:12])
        np.testing.assert_array_equal(examples['context_indices'], indices)
        np.testing.assert_array_equal(examples['X'], snapshot['X'][indices])
        np.testing.assert_array_equal(examples['truth'][0], examples['truth'][1])
        for ni in range(2):
            expected = study._example(p, snapshot, designs, first,
                       baseline={'n': p['n_values'][ni], 'B': p['B_values'][ni], 'delta': .1}, context_indices=indices)
            np.testing.assert_allclose(examples['bootstrap_lower'][ni], expected['bootstrap_lower'])
            np.testing.assert_allclose(examples['bootstrap_upper'][ni], expected['bootstrap_upper'])
    assert len(summary['plots']) == 6
    assert all(Path(record['path']).read_bytes().startswith(b'%PDF-') for record in summary['plots'])


def test_shared_setting_reproduces_existing_study_draws(manifest_path, tmp_path):
    from experiments.settings import prepare_setting
    manifest = study.load_manifest(manifest_path)
    p = manifest.payload
    prepared = prepare_setting(dict(source='wild_study', manifest=str(manifest_path), n=12, repetition=0))
    record = study.run_repetition(manifest, 0, tmp_path/'study')
    snapshot, designs, _ = study._environment(json.dumps(p, sort_keys=True))
    with np.load(record['arrays']['path']) as saved:
        expected = designs[0].design @ np.asarray(p['truth']['coefficients']) + snapshot['sigma'][:12]*saved['innovations'][:12]
        np.testing.assert_array_equal(prepared.observations.y, expected)
    full = 2*np.random.default_rng(record['seeds']['method']).integers(0, 2, size=(max(p['B_values']), max(p['n_values'])), dtype=np.int8)-1
    np.testing.assert_array_equal(prepared.multipliers, full[:,:12])
    assert prepared.provenance['multiplier_sha256'] == record['multiplier_sha256']
