"""Check paired bootstrap batching and honest residual-fit diagnostics."""

import numpy as np

from experiments.envelopes.wild import WildLinearDesign
from scratch.generali_residual_wild import _bootstrap_batches, _metrics


def test_batching_preserves_draws_and_matches_independent_ols_refits():
    rng = np.random.default_rng(261027)
    design = WildLinearDesign(rng.normal(size=(31, 2)), rng.normal(size=31))
    residual = design.design @ np.array([.5, 1., -.2, .3]) + rng.normal(size=31)
    responses = {"glm": residual, "spline_xgb": 2*residual}
    small = list(_bootstrap_batches(design, responses, count=9, batch_size=2, method_seed=123))
    large = list(_bootstrap_batches(design, responses, count=9, batch_size=5, method_seed=123))
    signs = np.concatenate([item[1] for item in small])
    np.testing.assert_array_equal(signs, np.concatenate([item[1] for item in large]))
    np.testing.assert_array_equal(np.concatenate([item[0] for item in small]),
                                  np.concatenate([item[0] for item in large]))
    beta = np.linalg.lstsq(design.design, residual, rcond=None)[0]
    fitted = design.design @ beta
    errors = np.concatenate([item[2]["glm"]["errors"] for item in small])
    reference = np.stack([np.linalg.lstsq(design.design, fitted+s*(residual-fitted), rcond=None)[0]-beta
                          for s in signs])
    np.testing.assert_allclose(errors, reference, atol=2e-14)
    np.testing.assert_allclose(errors, np.concatenate([item[2]["glm"]["errors"] for item in large]), atol=2e-14)
    np.testing.assert_allclose(2*errors, np.concatenate([item[2]["spline_xgb"]["errors"] for item in small]), atol=2e-14)
    np.testing.assert_allclose(np.concatenate([item[2]["glm"]["statistics"] for item in small]),
                               design.statistics(reference), atol=2e-13)


def test_metrics_keep_zero_baseline_distinct_from_residual_mean_baseline():
    residual = np.array([1., 3., -2., -4.])
    fitted = np.array([1., 2., -1., -3.])
    result = _metrics(residual, fitted, np.array([0, 0, 1, 1]), model="example", stage="holdout")
    all_rows, renewed, churned = result
    assert all_rows["n"] == 4
    np.testing.assert_allclose(all_rows["raw_rmse"], np.sqrt(7.5))
    np.testing.assert_allclose(all_rows["fit_rmse"], np.sqrt(.75))
    np.testing.assert_allclose(all_rows["residual_r2"], 1-3/29)
    assert renewed["n"] + churned["n"] == 4
    assert renewed["raw_residual_mean"] == 2
    assert churned["raw_residual_mean"] == -3
