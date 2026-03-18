from __future__ import annotations

import numpy as np

from objective.base import default_rng, sample_states


def test_sample_states_shape() -> None:
    rng = default_rng(123)
    x_samples = sample_states(rng, n=10, dim=5)
    assert x_samples.shape == (10, 5)


def test_sample_states_dtype() -> None:
    rng = default_rng(456)
    x_samples = sample_states(rng, n=3, dim=2)
    assert x_samples.dtype == float
