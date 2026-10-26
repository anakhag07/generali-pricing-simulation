"""Cohort selection and support diagnostics must preserve observation identity."""
import numpy as np
import pandas as pd

from scratch.generali_wild_slices import _action_groups, _support_diagnostics


def test_inclusive_logged_action_groups_choose_their_own_median_norm():
    a = np.array([-.005, 0., .005, .095, .1, .105, .2, .3, .4])
    z = np.array([3., 1., 2., 6., 4., 5., 9., 7., 8.])[:, None]
    groups = _action_groups(a, z, {"centers": [0., .1], "half_width": .005, "high_quantile": .7})
    for group, positions, representative in zip(groups, [range(3), range(3, 6), range(6, 9)], [2, 5, 8]):
        np.testing.assert_array_equal(group["positions"], list(positions))
        assert group["representative"] == representative


def test_support_neighbours_exclude_all_records_of_the_query_customer():
    x = np.array([0., .1, 3., 10.])
    arrays = {"customer_id": np.array(["A", "A", "B", "C"]), "row_indices": np.array([4, 5, 7, 9]),
              "a": np.zeros(4)}
    state = {"X": x[:, None], "leverage": np.full(4, .25),
             "feature_names": np.array(["intercept", "z:x", "action"])}
    records, _ = _support_diagnostics(arrays, state, pd.DataFrame({"x": x}), [("example", 0)], np.array([7]))
    assert records[0]["nearest_other_customer_row"] == 7
    assert records[0]["nearest_other_customer_distance"] == 3.
    assert records[0]["in_calibration_cohort"]
    assert not records[0]["in_optimization_cohort"]
