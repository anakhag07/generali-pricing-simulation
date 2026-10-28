"""Independent refit and function-only access checks for the 10/21 illustration."""

import numpy as np
import pytest

from scratch import run_wild_data_location as script


@pytest.fixture
def example():
    actions = np.linspace(-1., 2., 17)
    design = script.features(actions)
    responses = design @ [0., 5., -5.] + np.sin(np.arange(17))
    # Explicit deterministic multipliers keep this a numerical identity test.
    signs = np.array([[1. if (i + b) % (b + 2) else -1. for i in range(17)]
                      for b in range(5)])
    return script.fit_band(actions, responses, signs), design, signs


def test_wild_draws_equal_independent_refits(example):
    band, design, signs = example
    beta = np.linalg.lstsq(design, band["responses"], rcond=None)[0]
    residuals = band["responses"] - design @ beta
    np.testing.assert_allclose(band["beta"], beta, atol=1e-13)
    for sign, error in zip(signs, band["bootstrap_errors"]):
        refit = np.linalg.lstsq(design, design @ beta + residuals * sign, rcond=None)[0]
        np.testing.assert_allclose(error, refit-beta, atol=1e-13)


def test_frozen_oracle_evaluates_off_grid_and_rejects_gradients(example):
    band, design, _ = example
    oracle = script.QueryOnlyLowerBound(band)
    action = 2.718281828
    p = np.array([1., action, action**2])
    expected = -p @ band["beta"] + band["critical"] * np.sqrt(
        p @ np.linalg.inv(design.T @ design) @ p)
    first = oracle.value([action], None)
    assert first == pytest.approx(expected)
    assert oracle.value([action], None) == first
    with pytest.raises(AssertionError, match="function queries only"):
        oracle.grad([action], None)


def test_statistic_upper_bounds_cover_every_bootstrap_curve(example):
    band, _, _ = example
    certificate = script.QuadraticErrorCertificate(band["covariance"], 1.)
    for error, (low, high) in zip(band["bootstrap_errors"], band["bootstrap_statistic_bounds"]):
        difference = certificate.difference(error, np.zeros(3))
        assert certificate.contains(difference, high)
        assert high-low <= 2e-8 * max(1., high)


def test_steepness_signal_shift_matches_independent_wild_refits(example):
    band, design, signs = example
    shifted = script._shift_signal(band, 4.)
    independently_fitted = script.fit_band(band["actions"], shifted["responses"], signs)
    for key in ("beta", "residuals", "bootstrap_errors", "critical"):
        np.testing.assert_allclose(shifted[key], independently_fitted[key], atol=2e-12)
    queries = np.array([-10., -.17, .5, 20.])
    _, radius = script.band_values(band, queries)
    _, shifted_radius = script.band_values(shifted, queries)
    np.testing.assert_array_equal(radius, shifted_radius)


def test_sweep_extension_preserves_original_draws_and_nested_prefixes():
    spec = {"n_values": [75, 300, 600], "baseline_n": 300}
    seeds, bootstrap_seed, draws = script._paired_sweep_draws(spec, 0)
    np.testing.assert_array_equal(draws["design_innovations"][:300],
                                 np.random.default_rng(seeds.data_seed).normal(size=300))
    np.testing.assert_array_equal(draws["noise_innovations"][:300],
                                 np.random.default_rng(seeds.noise_seed).normal(size=300))
    original = np.random.default_rng(bootstrap_seed).choice(
        [-1., 1.], size=(script.CONFIG["bootstrap_draws"], 300))
    np.testing.assert_array_equal(draws["multipliers"][:, :300], original)
    repeated = script._paired_sweep_draws(spec, 0)[2]
    for key in draws:
        np.testing.assert_array_equal(draws[key], repeated[key])
    independent = script._paired_sweep_draws(spec, 1)[2]
    assert not np.array_equal(draws["noise_innovations"], independent["noise_innovations"])
