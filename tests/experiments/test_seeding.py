from __future__ import annotations

import numpy as np
import pytest

from experiments.configs import get_config
from experiments.seeding import (
    SeedSetup,
    derive_seed,
    optimizer_rngs,
    resolve_seed_setup,
    seed_setup_from_mapping,
)


def test_resolve_seed_setup_is_deterministic() -> None:
    first = resolve_seed_setup(SeedSetup(run_seed=42), legacy_seed=7)
    second = resolve_seed_setup(SeedSetup(run_seed=42), legacy_seed=7)

    assert first == second
    assert first.run_seed == 42
    assert len(set(first.to_dict().values())) == 5


def test_resolve_seed_setup_preserves_explicit_overrides() -> None:
    resolved = resolve_seed_setup(
        SeedSetup(
            run_seed=3,
            data_seed=100,
            split_seed=101,
            theta_seed=102,
            optimizer_seed=103,
        ),
        legacy_seed=7,
    )

    assert resolved.to_dict() == {
        "run_seed": 3,
        "data_seed": 100,
        "split_seed": 101,
        "theta_seed": 102,
        "optimizer_seed": 103,
    }


def test_resolve_seed_setup_uses_legacy_seed_when_setup_absent() -> None:
    resolved = resolve_seed_setup(None, legacy_seed=11)

    assert resolved.run_seed == 11
    assert resolved.data_seed == derive_seed(11, "data")


def test_seed_setup_from_mapping_normalizes_payload() -> None:
    setup = seed_setup_from_mapping(
        {
            "run_seed": 5,
            "data_seed": None,
            "split_seed": 6,
            "theta_seed": None,
            "optimizer_seed": 7,
        }
    )

    assert setup == SeedSetup(run_seed=5, split_seed=6, optimizer_seed=7)


def test_seed_setup_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="run_seed"):
        SeedSetup(run_seed=-1)


def test_optimizer_rngs_are_estimator_specific() -> None:
    seeds = resolve_seed_setup(SeedSetup(run_seed=9, optimizer_seed=123), legacy_seed=7)
    first_batch, first_gradient = optimizer_rngs(seeds, "first_order")
    stein_batch, stein_gradient = optimizer_rngs(seeds, "stein_difference")

    first_values = (
        first_batch.normal(size=3),
        first_gradient.normal(size=3),
    )
    stein_values = (
        stein_batch.normal(size=3),
        stein_gradient.normal(size=3),
    )

    assert not np.allclose(first_values[0], stein_values[0])
    assert not np.allclose(first_values[1], stein_values[1])


def test_experiment_config_serializes_seed_setup() -> None:
    config = get_config(
        "planted_logistic_base",
        overrides={"seed_setup": SeedSetup(run_seed=13, optimizer_seed=99)},
    )

    payload = config.to_dict()

    assert payload["seed"] == 7
    assert payload["seed_setup"] == {
        "run_seed": 13,
        "data_seed": None,
        "split_seed": None,
        "theta_seed": None,
        "optimizer_seed": 99,
    }
    assert payload["resolved_seed_setup"]["run_seed"] == 13
    assert payload["resolved_seed_setup"]["optimizer_seed"] == 99
