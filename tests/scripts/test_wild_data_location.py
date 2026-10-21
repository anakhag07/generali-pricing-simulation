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
