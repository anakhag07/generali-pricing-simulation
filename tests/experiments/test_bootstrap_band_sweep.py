from pathlib import Path
import json

import numpy as np
import pytest
import sympy as sp

from experiments.bootstrap_band_sweep import (
    VARIABLE, QuadraticLCBObjective, lcb_below_level, optimize_lcb,
    load_manifest, dataset_streams, calibrate_design, run_dataset, collect,
)
from experiments.bootstrap_band import features
from experiments.bootstrap_band_continuous import nonnegative_on_real_line

ROOT = Path(__file__).resolve().parents[2]


def payload():
    return json.loads((ROOT/"manifests/bootstrap_ols_controlled_sweep.json").read_text())


def dense_n_payload():
    return json.loads((ROOT/"manifests/bootstrap_ols_dense_n_sweep.json").read_text())


def test_lcb_gradient_matches_finite_difference():
    v = np.array([[2., .1, -.2], [.1, 1., .15], [-.2, .15, 1.]])
    objective = QuadraticLCBObjective([.1, 5.2, -4.9], v, 2.3)
    for a in (-10., -.5, 0., .5, 1., 10.):
        step = 1e-5
        expected = (objective.value([a+step], None)-objective.value([a-step], None))/(2*step)
        assert objective.grad([a], None)[0] == pytest.approx(expected, rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("beta,q,level,expected", [
    ([0, 5, -5], 0, 1.25, True), ([0, 5, -5], 0, 1.249999, False),
    ([0, 0, 1], 1+VARIABLE**2+VARIABLE**4, 0, True),
    ([0, 0, 2], 1+VARIABLE**2+VARIABLE**4, 0, False),
    ([1, 0, 0], 1+VARIABLE**2+VARIABLE**4, 0, True),
    ([1.000000001, 0, 0], 1+VARIABLE**2+VARIABLE**4, 0, False),
])
def test_exact_global_level_certification(beta, q, level, expected):
    assert lcb_below_level(beta, sp.Poly(q, VARIABLE, domain=sp.QQ), level) is expected


def test_sign_cells_detect_narrow_violation_and_ignore_irrelevant_negative_square():
    a = VARIABLE
    q = sp.Poly((a-2)**2+1-sp.Rational(1, 10**20), a, domain=sp.QQ)
    assert not lcb_below_level([1, 0, 0], q, 0)
    assert lcb_below_level([-1, 0, 0], q, 0)


def test_sign_cells_refine_intervals_touching_an_exact_root():
    a = VARIABLE
    # Default root intervals for S are (-1,0), (0,0): zero cannot be used as
    # the test point between them, because the negative open interval is missed.
    q = sp.Poly(1+a*(a+sp.Rational(1, 10**20)), a, domain=sp.QQ)
    assert not lcb_below_level([1, 0, 0], q, 0)


def test_repository_optimizer_is_unconstrained_and_tail_failures_are_explicit():
    cfg = payload()["optimizer"]
    result = optimize_lcb([0, 20, -5], np.eye(3), 1., 0., cfg)
    assert result["optimization_state"] == "certified"
    assert result["action"] == pytest.approx(2., abs=1e-8)  # Outside initialization interval.
    assert result["global_gap_upper"] <= cfg["global_gap_tolerance"]*1.001
    assert len(result["attempts"]) == len(cfg["starts"])
    assert optimize_lcb([0, 0, 2], np.eye(3), 1., 1., cfg)["optimization_state"] == "unbounded"
    assert optimize_lcb([0, 0, 1], np.eye(3), 1., 1., cfg)["optimization_state"] == "degenerate_tail"


def test_solver_success_alone_cannot_certify_a_nonoptimal_action(monkeypatch):
    from types import SimpleNamespace
    trace = SimpleNamespace(optimizer_success=True, optimizer_status=0, optimizer_message="mock success")
    monkeypatch.setattr("experiments.bootstrap_band_sweep.run_first_order_minimize",
                        lambda *args, **kwargs: (np.array([0.]), trace))
    result = optimize_lcb([0, 5, -5], np.eye(3), 1., 0., payload()["optimizer"])
    assert result["solver_success"]
    assert result["optimization_state"] == "uncertified"
    assert result["global_gap_upper"] is None


def test_level_certificate_agrees_with_polynomial_band_special_case():
    # For v=(1+a^2)^2, sqrt(v)=1+a^2 everywhere; the global inequality
    # reduces independently to polynomial nonnegativity, without selecting an action.
    rng = np.random.default_rng(15)
    for _ in range(40):
        beta = rng.normal(size=3)
        scale, level = float(rng.uniform(.1, 3)), float(rng.normal())
        radius = sp.Rational(scale)*(1+VARIABLE**2)
        fitted = sp.Poly.from_list([sp.Rational(float(c)) for c in beta[::-1]], VARIABLE, domain=sp.QQ)
        expected = nonnegative_on_real_line(sp.Poly(sp.Rational(level)+radius, VARIABLE)-fitted)
        assert lcb_below_level(beta, sp.Poly(radius**2, VARIABLE), level) == expected


def test_standardized_bootstrap_is_an_actual_refit_without_truth_calibration():
    rng = np.random.default_rng(81)
    x = rng.normal(size=25)
    # Arbitrary observed responses ensure calibration cannot reconstruct noise
    # from a presumed truth or replace it by freshly generated normal draws.
    y = 4-2*x+3*x*x + (1+x*x)*rng.uniform(-1, 1, size=len(x))
    indices = rng.integers(0, len(x), size=(9, len(x)))
    p, q, triangular, v, d, bounds, timings = calibrate_design(x, y, indices, 1e-8)
    beta = np.linalg.lstsq(p, y, rcond=None)[0]
    sigma_hat = np.linalg.norm(y-p@beta)/np.sqrt(len(x)-3)
    direct = np.array([np.linalg.lstsq(p[rows], y[rows], rcond=None)[0] for rows in indices])
    np.testing.assert_allclose(beta+sigma_hat*d, direct, atol=1e-12)
    assert np.all(bounds[:, 1] >= bounds[:, 0])
    assert all(value >= 0 for value in timings.values())


def test_nested_independent_streams():
    p = payload()
    first = dataset_streams(p, 0)
    replay = dataset_streams(p, 0)
    second = dataset_streams(p, 1)
    assert len(set(first[0].values())) == 3
    for a, b in zip(first[1:], replay[1:]):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(first[1], second[1])
    for key, values in zip(("design", "observation"), first[1:3]):
        np.testing.assert_array_equal(values, np.random.default_rng(first[0][key]).normal(size=len(values)))
    assert np.all((0 <= first[3]) & (first[3] < 1))
    p["axes"]["B"] = [5, 2000, 2001]
    larger = dataset_streams(p, 0)
    np.testing.assert_array_equal(first[3], larger[3][:len(first[3])])
    p["seeds"]["bootstrap"] += 1
    changed = dataset_streams(p, 0)
    np.testing.assert_array_equal(first[1], changed[1])
    np.testing.assert_array_equal(first[2], changed[2])
    assert not np.array_equal(larger[3], changed[3])


def test_paired_sweep_end_to_end_certificates_scaling_replay_and_pdfs(tmp_path):
    p = payload()
    p["datasets"] = 2
    p["baseline"] = {"N": 12, "sigma": 1., "B": 9}
    p["axes"] = {"sigma": [.5, 1., 2.], "N": [8, 12, 20], "B": [5, 9, 19]}
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(p))
    manifest = load_manifest(path)
    for index in range(2):
        saved = run_dataset(manifest, index, tmp_path)
        assert run_dataset(manifest, index, tmp_path) == saved
        noise_rows = [r for r in saved["rows"] if r["axis"] == "sigma"]
        assert len({r["covered"] for r in noise_rows}) == 1
        np.testing.assert_allclose([r["width_at_true_optimum"]/r["sigma"] for r in noise_rows],
                                   noise_rows[1]["width_at_true_optimum"], rtol=1e-12)
        np.testing.assert_allclose([r["sigma_hat"]/r["sigma"] for r in noise_rows], noise_rows[1]["sigma_hat"], rtol=1e-12)
        with np.load(saved["arrays"]["path"]) as arrays:
            for r in saved["rows"]:
                indices = arrays[f"N{r['N']}_bootstrap_indices"][:r["B"]]
                design = features(arrays["training_actions"][:r["N"]])
                y = arrays[r["array_prefix"]+"_y"]
                direct = np.array([np.linalg.lstsq(design[rows], y[rows], rcond=None)[0] for rows in indices])
                np.testing.assert_allclose(arrays[r["array_prefix"]+"_bootstrap_beta"], direct, atol=1e-11)
                coefficients = arrays[r["array_prefix"]+"_containment_exact_coefficients"]
                poly = sp.Poly.from_list([sp.Rational(v) for v in coefficients[::-1]], VARIABLE, domain=sp.QQ)
                assert nonnegative_on_real_line(poly) == r["covered"]
                if r["covered"] and r["optimization_state"] == "certified":
                    assert r["regret"] <= r["regret_bound"]+r["global_gap_upper"]+1e-12
            for r in noise_rows:
                np.testing.assert_allclose(arrays[r["array_prefix"]+"_error_coefficients"]/r["sigma"],
                                           arrays[noise_rows[1]["array_prefix"]+"_error_coefficients"], atol=1e-12)
            b_rows = [r for r in saved["rows"] if r["axis"] == "B"]
            for r in b_rows:
                np.testing.assert_array_equal(arrays[r["array_prefix"]+"_beta_hat"],
                                               arrays[b_rows[0]["array_prefix"]+"_beta_hat"])
    collect(manifest, tmp_path)
    project = tmp_path/manifest.name
    summary = json.loads((project/"summary.json").read_text())
    assert len(summary["metrics"]) == 9
    assert len(summary["plots"]) == 3
    for record in summary["plots"]:
        assert Path(record["path"]).read_bytes().startswith(b"%PDF-")
    assert not list(project.rglob("*.png"))
    p["seeds"]["bootstrap"] += 1
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="Cached sweep contract"):
        run_dataset(load_manifest(path), 0, tmp_path)


def test_n_only_sweep_uses_fixed_bootstrap_count_and_single_panel_pdfs(tmp_path):
    p = dense_n_payload()
    p["datasets"] = 2
    p["baseline"] = {"N": 12, "sigma": 1., "B": 9}
    p["axes"] = {"N": [8, 12, 20]}
    path = tmp_path/"dense-n.json"
    path.write_text(json.dumps(p))
    manifest = load_manifest(path)
    _, x, epsilon, z = dataset_streams(p, 0)
    assert x.shape == epsilon.shape == (20,)
    assert z.shape == (9, 20)
    for index in range(2):
        saved = run_dataset(manifest, index, tmp_path)
        assert [(row["axis"], row["N"], row["sigma"], row["B"]) for row in saved["rows"]] == [
            ("N", 8, 1., 9), ("N", 12, 1., 9), ("N", 20, 1., 9),
        ]
    collect(manifest, tmp_path)
    project = tmp_path/manifest.name
    summary = json.loads((project/"summary.json").read_text())
    assert len(summary["metrics"]) == 3
    assert len(summary["plots"]) == 3
    for record in summary["plots"]:
        assert Path(record["path"]).read_bytes().startswith(b"%PDF-")
    assert not list(project.rglob("*.png"))


@pytest.mark.parametrize("key,value", [("domain", "interval"), ("delta", 1.), ("datasets", 1)])
def test_manifest_rejects_invalid_contract(tmp_path, key, value):
    p = payload()
    p[key] = value
    path = tmp_path/"bad.json"
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_manifest_rejects_empty_axes_and_invalid_fixed_baseline(tmp_path):
    for axes, baseline in (({}, {"N": 100, "sigma": 1., "B": 500}),
                           ({"N": [100]}, {"N": 100, "sigma": 1., "B": 1})):
        p = dense_n_payload()
        p["axes"], p["baseline"] = axes, baseline
        path = tmp_path/f"bad-{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps(p))
        with pytest.raises(ValueError):
            load_manifest(path)


def test_shared_launcher_has_one_cpu_task_per_dataset(tmp_path):
    from experiments.bootstrap_band_sweep import build_launch_plan
    manifest = load_manifest(ROOT/"manifests/bootstrap_ols_controlled_sweep.json")
    plan = build_launch_plan(manifest, runs_root=tmp_path)
    assert plan.task_count == 100
    assert plan.default_array
    assert plan.slurm_profile.cpus_per_task == 1
    assert plan.slurm_profile.gres is None


def test_original_parametric_manifest_cannot_be_silently_reinterpreted(tmp_path):
    p = payload()
    del p["bootstrap_method"]
    path = tmp_path/"old.json"
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError, match="pairs"):
        load_manifest(path)


def test_approved_sweep_is_not_a_cartesian_product():
    from experiments.bootstrap_band_sweep import settings
    p = payload()
    rows = list(settings(p))
    assert p["baseline"] == {"N": 100, "sigma": 1., "B": 2000}
    assert len(rows) == 37
    assert len({(r["N"], r["sigma"], r["B"]) for _, _, r in rows}) == 35
    for axis, value, setting in rows:
        assert all(setting[key] == base for key, base in p["baseline"].items() if key != axis)
    assert p["axes"]["N"][-1] == 5000
    assert p["axes"]["B"][0] == 5
