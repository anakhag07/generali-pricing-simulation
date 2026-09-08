from __future__ import annotations

import numpy as np

from data.coverage import local_support_matrix, normalized_coverage_widths


def test_local_support_is_deterministic_for_fixed_inputs() -> None:
    embedding = np.asarray([[0.0], [0.5], [1.0], [1.5]], dtype=float)
    observed_u = np.asarray([0.00, 0.05, 0.10, 0.15])
    grid = np.asarray([0.0, 0.08, 0.16])

    first = local_support_matrix(
        embedding, observed_u, grid, n_neighbors=2, n_jobs=1
    )
    second = local_support_matrix(
        embedding, observed_u, grid, n_neighbors=2, n_jobs=1
    )

    np.testing.assert_array_equal(first, second)
    widths = normalized_coverage_widths(first, scale=10.0)
    assert widths.shape == first.shape
    assert np.all((widths >= 0.0) & (widths <= 10.0))
