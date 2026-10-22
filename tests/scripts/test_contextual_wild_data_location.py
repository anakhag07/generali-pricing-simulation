"""Independent refits, simultaneous bounds, and native contextual Stein scaling."""

from types import SimpleNamespace

import numpy as np
import pytest

from optimization.gradients.methods import SteinDifferenceGradient
from scratch import run_contextual_wild_data_location as script


@pytest.fixture
def example():
    index = np.arange(19)
    contexts = np.column_stack((np.sin(index), np.cos(.7*index), np.sin(.3*index)))
    actions = np.linspace(-1., 2., len(index))
    design = script.features(contexts, actions)
    responses = design @ script.CONFIG["truth_coefficients"] + np.cos(1.3*index)
    signs = np.array([[1. if (i+b) % (b+2) else -1. for i in index]
                      for b in range(5)])
    return script.fit_band(contexts, actions, responses, signs), design, signs


def test_contextual_wild_errors_equal_independent_ols_refits(example):
    band, design, signs = example
    beta = np.linalg.lstsq(design, band["responses"], rcond=None)[0]
    residuals = band["responses"] - design @ beta
    np.testing.assert_allclose(band["beta"], beta, atol=1e-13)
    for sign, error, statistic in zip(signs, band["bootstrap_errors"],
                                     band["bootstrap_statistics"]):
        refit = np.linalg.lstsq(design, design @ beta + residuals*sign, rcond=None)[0]
        np.testing.assert_allclose(error, refit-beta, atol=1e-13)
        assert statistic == pytest.approx(np.linalg.norm(design @ error))
        fiber_error, _, _ = script.fibers(band["contexts"], error, band["inverse"])
        for coefficients, covariance in zip(fiber_error, band["fiber_covariance"]):
            certificate = script.QuadraticErrorCertificate(covariance, 1.)
            assert certificate.contains(certificate.difference(coefficients, np.zeros(3)),
                                        statistic + 1e-12)


def test_personalized_off_grid_values_match_full_feature_formula(example):
    band, design, _ = example
    actions = np.linspace(-2.718281828, 10.123456789, len(design))
    phi = script.features(band["contexts"], actions)
    covariance = np.linalg.inv(design.T @ design)
    expected_fit = phi @ band["beta"]
    expected_radius = band["critical"] * np.sqrt(np.einsum("ni,ij,nj->n", phi, covariance, phi))
    fitted, radius = script.band_at_actions(band, actions)
    np.testing.assert_allclose(fitted, expected_fit)
    np.testing.assert_allclose(radius, expected_radius)
    oracle = script.QueryOnlyContextLCB(band)
    assert oracle.value(actions, band["contexts"]) == pytest.approx(np.mean(expected_radius-expected_fit))
    with pytest.raises(AssertionError, match="function values only"):
        oracle.grad(actions, band["contexts"])


def test_native_u_stein_uses_mean_gradient_and_identity_policy(example):
    band, design, _ = example
    n = len(design)
    theta = np.linspace(-.4, 1.4, n)
    seed = script.CONFIG["run_seed"]  # Fixed test-only replay, not a simulation stream.
    sigma, m = .01, 7
    oracle = script.QueryOnlyContextLCB(band)
    optimizer = SimpleNamespace(objective=oracle, x_array=band["contexts"], n_total=n,
                                perturbation_space="u", sigma=sigma, n_grad_samples=m,
                                rng=np.random.default_rng(seed))
    actual = SteinDifferenceGradient().theta_grad(optimizer, theta, np.arange(n))
    directions = np.random.default_rng(seed).normal(size=m)
    values = []
    for sign in (1., -1.):
        rows = []
        for direction in directions:
            phi = script.features(band["contexts"], theta + sign*sigma*direction)
            radius = band["critical"] * np.linalg.norm(phi @ band["inverse"], axis=1)
            rows.append(radius - phi @ band["beta"])
        values.append(np.asarray(rows))
    action_gradient = np.mean((values[0]-values[1])*directions[:, None]/(2*sigma), axis=0)
    np.testing.assert_allclose(actual*n, action_gradient, atol=1e-12)
    assert oracle.query_count == 2*m*n
