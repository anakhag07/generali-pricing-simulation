"""Synthetic ladder preset construction and end-to-end smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.configs import get_config, list_configs
from experiments.configs.synthetic_ladder import (
    DEFAULT_ESTIMATORS,
    build_synthetic_ladder_config,
)
from experiments.run import run_experiment
from objective import IMPLEMENTED_SYNTHETIC_LADDER, SyntheticFunction

_PRESETS = ("synthetic_quadratic_base", "synthetic_smoothed_nonconvex_base")

_SMOKE_OVERRIDES = {
    "dimension": 4,
    "t_steps": 5,
    "n_grad_samples": 4,
    "plot": False,
}


def test_presets_registered() -> None:
    names = list_configs()
    for preset in _PRESETS:
        assert preset in names


@pytest.mark.parametrize("preset", _PRESETS)
def test_preset_defaults(preset: str) -> None:
    config = get_config(preset)
    assert isinstance(config.objective, SyntheticFunction)
    assert config.perturbation_space == "theta"
    assert config.theta0 is None
    assert config.enabled_estimators == DEFAULT_ESTIMATORS
    assert "gauss_stein" not in config.enabled_estimators


def test_dimension_and_function_params_forwarded() -> None:
    config = build_synthetic_ladder_config(
        rung="quadratic",
        dimension=6,
        function_params={"condition_number": 10.0},
        plot=False,
    )
    assert config.objective.theta_dim() == 6
    assert config.objective.eigenvalues.max() == pytest.approx(10.0)


def test_rejects_u_perturbation_space() -> None:
    with pytest.raises(ValueError, match="no action space"):
        build_synthetic_ladder_config(rung="quadratic", perturbation_space="u")


def test_rejects_stub_rung() -> None:
    with pytest.raises(ValueError, match="structural stub"):
        build_synthetic_ladder_config(rung="piecewise_convex")


def test_rejects_unknown_rung_and_override() -> None:
    with pytest.raises(ValueError, match="Unknown synthetic ladder rung"):
        build_synthetic_ladder_config(rung="nope")
    with pytest.raises(ValueError, match="override fields"):
        build_synthetic_ladder_config(rung="quadratic", not_a_field=1)


@pytest.mark.parametrize("rung", IMPLEMENTED_SYNTHETIC_LADDER)
def test_smoke_run_is_finite_and_deterministic(rung: str) -> None:
    def run():
        config = build_synthetic_ladder_config(rung=rung, **_SMOKE_OVERRIDES)
        return run_experiment(config)

    first, second = run(), run()
    assert set(first.results) == set(DEFAULT_ESTIMATORS)
    for name, estimator_result in first.results.items():
        assert np.all(np.isfinite(estimator_result.theta))
        assert np.isfinite(estimator_result.value)
        assert estimator_result.u is None
        assert np.array_equal(estimator_result.theta, second.results[name].theta)


def test_smoke_run_true_gap_is_computable() -> None:
    config = build_synthetic_ladder_config(rung="quadratic", **_SMOKE_OVERRIDES)
    result = run_experiment(config)
    objective = config.objective
    for estimator_result in result.results.values():
        gap = estimator_result.value - objective.optimal_value()
        distance = np.linalg.norm(estimator_result.theta - objective.optimal_theta())
        assert gap >= 0.0
        assert np.isfinite(distance)
