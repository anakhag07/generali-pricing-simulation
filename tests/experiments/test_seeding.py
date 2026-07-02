from __future__ import annotations

import numpy as np
import pytest

from experiments.configs import get_config
from experiments.run import run_experiment
from experiments.seeding import (
    SeedSetup,
    optimizer_rngs,
    resolve_seed_setup,
    seed_setup_from_mapping,
)


def test_resolve_seed_setup_is_deterministic() -> None:
    first = resolve_seed_setup(SeedSetup(run_seed=42), legacy_seed=7)
    second = resolve_seed_setup(SeedSetup(run_seed=42), legacy_seed=7)

    assert first == second
    assert first.run_seed == 42
    assert len(set(first.to_dict().values())) == 6


def test_resolve_seed_setup_preserves_explicit_overrides() -> None:
    resolved = resolve_seed_setup(
        SeedSetup(
            run_seed=3,
            data_seed=100,
            split_seed=101,
            theta_seed=102,
            noise_seed=103,
            optimizer_seed=104,
        ),
        legacy_seed=7,
    )

    assert resolved.to_dict() == {
        "run_seed": 3,
        "data_seed": 100,
        "split_seed": 101,
        "theta_seed": 102,
        "noise_seed": 103,
        "optimizer_seed": 104,
    }


def test_resolve_seed_setup_uses_legacy_seed_when_setup_absent() -> None:
    resolved = resolve_seed_setup(None, legacy_seed=11)

    assert resolved.to_dict() == {
        "run_seed": 11,
        "data_seed": 11,
        "split_seed": 11,
        "theta_seed": 11,
        "noise_seed": 11,
        "optimizer_seed": 11,
    }


def test_seed_setup_from_mapping_normalizes_payload() -> None:
    setup = seed_setup_from_mapping(
        {
            "run_seed": 5,
            "data_seed": None,
            "split_seed": 6,
            "theta_seed": None,
            "noise_seed": 8,
            "optimizer_seed": 7,
        }
    )

    assert setup == SeedSetup(run_seed=5, split_seed=6, noise_seed=8, optimizer_seed=7)


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
        "noise_seed": None,
        "optimizer_seed": 99,
    }
    assert payload["resolved_seed_setup"]["run_seed"] == 13
    assert payload["resolved_seed_setup"]["optimizer_seed"] == 99


def test_run_experiment_uses_data_seed_for_synthetic_samples() -> None:
    base_overrides = {
        "n_samples": 6,
        "t_steps": 1,
        "test_fraction": 0.0,
        "train_fraction": 1.0,
        "enabled_estimators": ("first_order",),
        "plot": False,
        "verbose": False,
    }
    first = run_experiment(
        get_config(
            "planted_logistic_base",
            overrides={
                **base_overrides,
                "seed_setup": SeedSetup(run_seed=1, data_seed=10, split_seed=20, optimizer_seed=30),
            },
        )
    )
    second = run_experiment(
        get_config(
            "planted_logistic_base",
            overrides={
                **base_overrides,
                "seed_setup": SeedSetup(run_seed=1, data_seed=11, split_seed=20, optimizer_seed=30),
            },
        )
    )

    assert not np.allclose(first.x_samples, second.x_samples)


def test_run_experiment_uses_split_seed_for_train_test_split() -> None:
    config = get_config(
        "planted_logistic_base",
        overrides={
            "n_samples": 10,
            "t_steps": 1,
            "test_fraction": 0.3,
            "train_fraction": 0.7,
            "enabled_estimators": ("first_order",),
            "plot": False,
            "verbose": False,
            "seed_setup": SeedSetup(run_seed=1, data_seed=10, split_seed=123, optimizer_seed=30),
        },
    )

    result = run_experiment(config)
    shuffled = np.random.default_rng(123).permutation(10).astype(int)

    np.testing.assert_array_equal(result.test_indices, shuffled[:3])
    np.testing.assert_array_equal(result.train_indices, shuffled[3:])


def test_run_experiment_uses_theta_seed_for_random_theta0() -> None:
    base_overrides = {
        "theta0": None,
        "n_samples": 6,
        "t_steps": 1,
        "test_fraction": 0.0,
        "train_fraction": 1.0,
        "enabled_estimators": ("first_order",),
        "plot": False,
        "verbose": False,
    }
    first = run_experiment(
        get_config(
            "planted_logistic_base",
            overrides={
                **base_overrides,
                "seed_setup": SeedSetup(run_seed=1, data_seed=10, split_seed=20, theta_seed=100, optimizer_seed=30),
            },
        )
    )
    second = run_experiment(
        get_config(
            "planted_logistic_base",
            overrides={
                **base_overrides,
                "seed_setup": SeedSetup(run_seed=1, data_seed=10, split_seed=20, theta_seed=101, optimizer_seed=30),
            },
        )
    )

    assert first.config.theta0 is not None
    assert second.config.theta0 is not None
    assert not np.allclose(first.config.theta0, second.config.theta0)


def test_run_experiment_optimizer_streams_are_estimator_order_independent() -> None:
    base_overrides = {
        "n_samples": 8,
        "t_steps": 2,
        "step_rule": "constant",
        "step_size": 0.01,
        "n_grad_samples": 3,
        "sigma": 0.05,
        "test_fraction": 0.0,
        "train_fraction": 1.0,
        "plot": False,
        "verbose": False,
        "seed_setup": SeedSetup(run_seed=1, data_seed=10, split_seed=20, optimizer_seed=30),
    }
    first = run_experiment(
        get_config(
            "planted_logistic_base",
            overrides={**base_overrides, "enabled_estimators": ("spsa", "stein_difference")},
        )
    )
    second = run_experiment(
        get_config(
            "planted_logistic_base",
            overrides={**base_overrides, "enabled_estimators": ("stein_difference", "spsa")},
        )
    )

    for estimator in ("spsa", "stein_difference"):
        np.testing.assert_allclose(
            first.results[estimator].theta,
            second.results[estimator].theta,
        )
        np.testing.assert_allclose(
            first.traces[estimator].objective_values,
            second.traces[estimator].objective_values,
        )
