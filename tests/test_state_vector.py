from __future__ import annotations

from data.models import StateVector, default_rng


def test_state_vector_sample_dim() -> None:
    rng = default_rng(123)
    vector = StateVector.sample(rng, dim=5)
    assert vector.as_array().shape == (5,)
