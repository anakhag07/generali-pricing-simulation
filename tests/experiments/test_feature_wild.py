"""Independent refits and exact feature/action identities for the basis study."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from experiments.envelopes.feature_wild import WildFeatureDesign, feature_band_components
from scratch import generali_residual_basis_study as study
from scratch.generali_residual_basis_study import _Basis, _FeatureFibers, _split, _solve, _MLPLoss
from scratch.generali_residual_basis_study import _filter_rows, _fit, _subset_positions


def test_exclusion_preserves_split_identities_and_actual_bootstrap_calibration(tmp_path):
    rng = np.random.default_rng(414)
    inputs = rng.normal(size=(160, 4))
    arrays = {"row_indices": np.arange(160)*3, "glm_residual": rng.normal(size=160),
              "is_churn": np.arange(160) % 2}
    split = {"representation": np.arange(60), "beta": np.arange(60, 130),
             "holdout": np.arange(130, 160)}
    filtered, retained, pools = _filter_rows(arrays, inputs, split, [3, 12, 405])
    np.testing.assert_array_equal(filtered["row_indices"][pools["beta"]], arrays["row_indices"][split["beta"]])
    for key in split:
        np.testing.assert_array_equal(filtered["row_indices"][pools[key]],
            np.setdiff1d(arrays["row_indices"][split[key]], [3, 12, 405]))
    config = {"mlp": {"hidden": 16}, "bootstrap": {"count": 31, "batch_size": 7, "delta": .05},
              "evaluation_batch_size": 32}
    seeds = rng.integers(0, 2**32, size=31)
    full_dir, trimmed_dir = tmp_path/"full", tmp_path/"trimmed"
    full_dir.mkdir(); trimmed_dir.mkdir()
    basis, state, info = _fit(config, "affine", "glm", arrays, inputs, split, seeds, full_dir)
    other, changed, details = _fit(config, "affine", "glm", filtered, retained, pools, seeds, trimmed_dir)
    assert info["full_row_multiplier_sha256"] == details["full_row_multiplier_sha256"]
    assert details["multiplier_domain_size"] == 160
    np.testing.assert_allclose(state["statistics"], changed["statistics"], atol=1e-13)
    np.testing.assert_allclose(basis.evaluate(retained)@state["beta"],
                               other.evaluate(retained)@changed["beta"], atol=1e-13)
    with pytest.raises(ValueError, match="unique and present"):
        _filter_rows(arrays, inputs, split, [3, 3])
    with pytest.raises(ValueError, match="unique and present"):
        _filter_rows(arrays, inputs, split, [999])


def test_policy_cohort_cannot_silently_lose_a_filtered_row():
    rows = np.array([1, 2, 5, 7])
    np.testing.assert_array_equal(_subset_positions(rows, np.array([2, 7])), [1, 3])
    for missing in [np.array([2, 8]), np.array([3])]:
        with pytest.raises(ValueError, match="not a subset"):
            _subset_positions(rows, missing)


def test_coefficient_bootstrap_matches_independent_refits_and_bounds_new_queries():
    rng = np.random.default_rng(410)
    matrix = np.column_stack((np.ones(73), rng.normal(size=(73, 8))))
    y = rng.normal(size=73)
    design = WildFeatureDesign(matrix)
    beta = design.fit(y)
    signs = rng.choice([-1, 1], size=(19, len(y)))
    errors, statistics = design.bootstrap_errors(y-matrix@beta, signs)
    for j, sign in enumerate(signs):
        refit = np.linalg.lstsq(matrix, matrix@beta+(y-matrix@beta)*sign, rcond=None)[0]
        np.testing.assert_allclose(refit-beta, errors[j], atol=1e-14)
    query = rng.normal(size=(101, matrix.shape[1]))
    for error, critical in zip(errors, statistics):
        parts = feature_band_components(query, beta, design.triangular, critical)
        assert np.all(np.abs(query@error) <= parts["width"]+1e-12)
    np.testing.assert_allclose(design.leverage.sum(), matrix.shape[1], atol=1e-12)


@pytest.mark.parametrize("kind", ["affine", "quadratic", "cubic", "mlp"])
def test_basis_action_derivative_and_exact_polynomial_cache(kind):
    rng = np.random.default_rng(411)
    x = rng.normal(size=(240, 4))
    weights = (rng.normal(size=(4, 5)), rng.normal(size=5), rng.normal(size=(5, 5)), rng.normal(size=5))
    basis = _Basis(kind, x.mean(0), x.std(0), hidden=5, weights=weights if kind == "mlp" else None)
    raw = basis.raw(x)
    basis.feature_mean, basis.feature_scale = raw.mean(0), raw.std(0)
    basis.feature_mean[0], basis.feature_scale[0] = 0., 1.
    design = WildFeatureDesign(basis.evaluate(x))
    beta = design.fit(rng.normal(size=len(x)))
    _, derivative = basis.evaluate(x, True)
    plus, minus = x.copy(), x.copy()
    plus[:, -1] += 1e-5
    minus[:, -1] -= 1e-5
    np.testing.assert_allclose(derivative, (basis.evaluate(plus)-basis.evaluate(minus))/2e-5, rtol=2e-7, atol=2e-8)
    fibers = _FeatureFibers(x, basis, beta, design.triangular, 3.)
    positions = np.arange(30)
    fibers.prepare(positions)
    actions = np.linspace(-.1, .2, len(positions))
    actual, direct = fibers.evaluate(positions, actions), fibers.direct(positions, actions)
    for key in actual:
        np.testing.assert_allclose(actual[key], direct[key], rtol=1e-9, atol=1e-10)
    xquery = x[positions].copy()
    xquery[:, -1] = actions
    parts = feature_band_components(basis.evaluate(xquery), beta, design.triangular, 3.)
    for key in parts:
        np.testing.assert_allclose(parts[key], actual[key], rtol=1e-9, atol=1e-10)
    plus = fibers.values(positions, actions+1e-6)
    minus = fibers.values(positions, actions-1e-6)
    for side, sign in (("lower", -1), ("upper", 1)):
        finite = sign*(plus[f"radius_{side}"]-minus[f"radius_{side}"])/2e-6
        np.testing.assert_allclose(actual[f"{side}_adjustment_derivative"], finite,
                                   rtol=2e-7, atol=2e-8)
    restored = _Basis.restore(basis.state())
    np.testing.assert_array_equal(restored.evaluate(x), basis.evaluate(x))


def test_customer_split_preserves_holdout_and_repeated_id_groups():
    ids = np.repeat(np.arange(30), 2)
    pools = _split(ids, np.arange(48), np.arange(48, 60), 412)
    np.testing.assert_array_equal(pools["holdout"], np.arange(48, 60))
    assert len(pools["representation"]) == len(pools["beta"]) == 24
    assert not set(ids[pools["representation"]]) & set(ids[pools["beta"]])
    again = _split(ids, np.arange(48), np.arange(48, 60), 412)
    for key in pools:
        np.testing.assert_array_equal(pools[key], again[key])


def test_rank_failure_is_explicit():
    with pytest.raises(ValueError, match="rank-deficient"):
        WildFeatureDesign(np.ones((30, 3)))


def test_neural_residual_gradient_matches_independent_finite_differences():
    rng = np.random.default_rng(413)
    x = rng.normal(size=(31, 3))
    objective = _MLPLoss(x, rng.normal(size=len(x)), 4)
    theta = rng.normal(size=3*4+4+4*4+4+4+1)*.15
    expected = []
    for i in range(len(theta)):
        delta = np.zeros_like(theta)
        delta[i] = 1e-5
        expected.append((objective.value(theta+delta, x)-objective.value(theta-delta, x))/2e-5)
    np.testing.assert_allclose(objective.grad(theta, x), expected, atol=2e-10, rtol=1e-6)


def test_repository_solver_checkpoint_can_be_replayed_without_optimization(tmp_path):
    class Quadratic:
        def value(self, theta, inputs):
            return float(np.sum((theta-2)**2))
        def grad(self, theta, inputs):
            return 2*(theta-2)
    settings = {"wall_seconds": 30, "max_steps": 20, "grad_norm_tol": 1e-8}
    result, metadata = _solve(Quadratic(), np.ones((2, 1)), np.zeros(2), settings,
                              tmp_path/"solver", algorithm="l-bfgs-b")
    np.testing.assert_allclose(result, 2., atol=1e-8)
    assert metadata["success"]
    class NeverFit:
        def value(self, *args):
            raise AssertionError("Completed replay must not optimize.")
    replay, same = _solve(NeverFit(), np.ones((2, 1)), np.zeros(2), settings,
                          tmp_path/"solver", algorithm="l-bfgs-b")
    np.testing.assert_array_equal(replay, result)
    assert same == metadata


@pytest.mark.parametrize("algorithm, filename", [("l-bfgs-b", "network"),
                                                  ("trust-constr", "policy_optimizer")])
def test_wall_budget_stops_without_finalizing_and_resumes(tmp_path, algorithm, filename):
    class Quadratic:
        def value(self, theta, inputs):
            return float(np.sum((theta-2)**2))
        def grad(self, theta, inputs):
            return 2*(theta-2)
    settings = {"wall_seconds": 0, "max_steps": 20, "grad_norm_tol": 1e-8}
    output = tmp_path/filename
    with pytest.raises(study._Deadline, match="wall-time"):
        _solve(Quadratic(), np.ones((2, 1)), np.zeros(2), settings,
               output, algorithm=algorithm)
    assert not output.with_suffix(".json").exists()
    saved = study._load(output.with_suffix(".npz"))["theta"]
    np.testing.assert_array_equal(saved, np.zeros(2))
    info = json.loads(output.with_suffix(".interrupted.json").read_text())
    assert info["success"] is False
    assert info["status"] == "wall_time_limit"
    theta, result = _solve(Quadratic(), np.ones((2, 1)), np.full(2, -10.),
                           {**settings, "wall_seconds": 30}, output, algorithm=algorithm)
    assert result["success"]
    assert result["restarted_from_saved_iterate"]
    np.testing.assert_allclose(theta, 2., atol=1e-8)
    np.testing.assert_array_equal(study._load(output.with_suffix(".npz"))["theta_trace"][0], saved)


def test_legacy_wall_time_result_resumes_instead_of_replaying(tmp_path):
    class Quadratic:
        def value(self, theta, inputs):
            return float(np.sum((theta-2)**2))
        def grad(self, theta, inputs):
            return 2*(theta-2)
    output = tmp_path/"network"
    study.save_arrays_atomic(output.with_suffix(".npz"), theta=np.ones(2))
    legacy = {"success": False, "status": "wall_time_limit",
              "checkpoint": study.file_record(output.with_suffix(".npz"))}
    study.write_json_atomic(output.with_suffix(".json"), legacy)
    theta, info = _solve(Quadratic(), np.ones((2, 1)), np.zeros(2),
        {"wall_seconds": 30, "max_steps": 20, "grad_norm_tol": 1e-8}, output, algorithm="l-bfgs-b")
    assert info["success"] and info["restarted_from_saved_iterate"]
    assert json.loads(output.with_suffix(".interrupted.json").read_text()) == legacy
    np.testing.assert_allclose(theta, 2., atol=1e-8)
    np.testing.assert_array_equal(study._load(output.with_suffix(".npz"))["theta_trace"][0], np.ones(2))


@pytest.mark.parametrize("algorithm", ["l-bfgs-b", "trust-constr"])
def test_deadline_checkpoint_uses_callback_iterate_after_rejected_trial(tmp_path, monkeypatch, algorithm):
    from optimization import solvers

    clock = [0.]
    monkeypatch.setattr(study, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    accepted, rejected = np.array([1., 1.]), np.array([-4., -3.])

    class Quadratic:
        def value(self, theta, inputs):
            return float(np.sum((theta-2)**2))
        def grad(self, theta, inputs):
            return 2*(theta-2)

    def trial_then_callback(fun, *, x0, jac, callback, **kwargs):
        # Inject evaluations into the repository's optimizer callback adapter;
        # no fitted solution is supplied by this solver test double.
        fun(accepted)
        jac(accepted)
        fun(rejected)
        clock[0] = 2.
        if algorithm == "trust-constr":
            callback(accepted, SimpleNamespace(x=accepted))
        else:
            callback(accepted)
        raise AssertionError("The deadline must stop this callback.")

    monkeypatch.setattr(solvers, "minimize", trial_then_callback)
    output = tmp_path/"callback"
    with pytest.raises(study._Deadline):
        _solve(Quadratic(), np.ones((2, 1)), np.zeros(2),
            {"wall_seconds": 1, "max_steps": 20, "grad_norm_tol": 1e-8}, output, algorithm=algorithm)
    np.testing.assert_array_equal(study._load(output.with_suffix(".npz"))["theta"], accepted)
    info = json.loads(output.with_suffix(".interrupted.json").read_text())
    assert info["final_objective"] == Quadratic().value(accepted, None)


@pytest.mark.parametrize("stage", ["network", "policy_optimizer"])
def test_task_deadline_does_not_publish_a_completed_fit_or_summary(tmp_path, monkeypatch, stage):
    inputs = np.arange(120, dtype=float).reshape(40, 3)/120
    arrays = {"glm_residual": np.sin(np.arange(40)), "row_indices": np.arange(40),
              "is_churn": np.arange(40) % 2}
    split = {"representation": np.arange(16), "beta": np.arange(16, 32), "holdout": np.arange(32, 40)}
    settings = {"wall_seconds": 0, "max_steps": 20, "grad_norm_tol": 1e-8}
    config = {"bases": ["mlp"], "models": ["glm"], "seed_root": 20261026,
              "mlp": {**settings, "hidden": 2}, "policy": settings}
    monkeypatch.setattr(study, "_contract", lambda *args: {"test": "deadline"})
    monkeypatch.setattr(study, "_inputs", lambda *args: (arrays, inputs, split, {}, np.arange(3)))
    if stage == "policy_optimizer":
        monkeypatch.setattr(study, "_fit", lambda *args: (None, {}, {}))
        def policy_stage(config, kind, name, arrays, inputs, basis, state, info, provenance, output):
            objective = _MLPLoss(inputs, arrays["glm_residual"], 2)
            _solve(objective, inputs, np.zeros(17), settings, output/"policy_optimizer",
                   algorithm="trust-constr")
            raise AssertionError("An interrupted policy must not finish its stage.")
        monkeypatch.setattr(study, "_policies", policy_stage)
    with pytest.raises(study._Deadline):
        study._run_task(config, tmp_path, tmp_path/"config.json", tmp_path, 0)
    folder = tmp_path/"mlp/glm"
    assert (folder/f"{stage}.npz").is_file()
    assert (folder/f"{stage}.interrupted.json").is_file()
    for name in [f"{stage}.json", "fit.json", "band.npz", "policy.npz", "summary.json"]:
        assert not (folder/name).exists()
