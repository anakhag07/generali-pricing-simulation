"""Independent refits and exact feature/action identities for the basis study."""
import numpy as np
import pytest
from experiments.envelopes.feature_wild import WildFeatureDesign, feature_band_components
from scratch.generali_residual_basis_study import _Basis, _FeatureFibers, _split, _solve, _MLPLoss


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


def test_wall_budget_retains_an_iterate_without_claiming_convergence(tmp_path):
    class Quadratic:
        def value(self, theta, inputs):
            return float(np.sum((theta-2)**2))
        def grad(self, theta, inputs):
            return 2*(theta-2)
    settings = {"wall_seconds": 0, "max_steps": 20, "grad_norm_tol": 1e-8}
    theta, info = _solve(Quadratic(), np.ones((2, 1)), np.zeros(2), settings,
                         tmp_path/"deadline", algorithm="l-bfgs-b")
    np.testing.assert_array_equal(theta, np.zeros(2))
    assert info["success"] is False
    assert info["status"] == "wall_time_limit"
    assert (tmp_path/"deadline.npz").is_file()
