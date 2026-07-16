"""Guards on objective serialization into run summaries.

A run's `summary.json` carries `config.objective`, and post-run analysis rebuilds
the objective from it to compute true gaps. An objective that serializes to a
bare `{"type": ...}` is therefore unreproducible, so the fallback must fail loudly
rather than silently drop construction parameters.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.config import _objective_to_dict
from experiments.configs import get_config
from objective import SmoothedNonconvex, StronglyConvexQuadratic, SyntheticFunction
from objective.base import Objective

_X_DUMMY = np.zeros((1, 1), dtype=float)

_LADDER_PRESETS = ("synthetic_quadratic_base", "synthetic_smoothed_nonconvex_base")


class _UnregisteredObjective(Objective):
    """Stand-in for a newly added objective nobody wired into the serializer."""

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        return 0.0

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        return np.zeros_like(theta)


def test_unserializable_objective_raises_instead_of_silently_degrading() -> None:
    with pytest.raises(TypeError, match="no serialization branch"):
        _objective_to_dict(_UnregisteredObjective())


@pytest.mark.parametrize("preset", _LADDER_PRESETS)
def test_ladder_preset_objective_round_trips(preset: str) -> None:
    config = get_config(preset, overrides={"dimension": 4, "plot": False})
    payload = config.to_dict()["objective"]

    json.dumps(payload, allow_nan=False)
    assert set(payload) >= {"type", "rung", "spec", "w_star", "fingerprint"}

    rebuilt = SyntheticFunction.from_dict(payload)
    probe = np.asarray(payload["w_star"], dtype=float) + 0.37
    assert rebuilt.value(probe, _X_DUMMY) == config.objective.value(probe, _X_DUMMY)
    np.testing.assert_allclose(rebuilt.optimal_theta(), config.objective.optimal_theta())


def test_from_dict_rejects_a_changed_construction() -> None:
    payload = StronglyConvexQuadratic.from_seed(11, dim=4).to_dict()
    payload["spec"]["params"]["condition_number"] = 2.0

    with pytest.raises(ValueError, match="does not match the recorded fingerprint"):
        SyntheticFunction.from_dict(payload)


def test_from_dict_rejects_a_directly_built_instance() -> None:
    """Instances built outside a seeded factory carry no replay spec."""
    payload = SmoothedNonconvex(
        w_star=np.zeros(2),
        center_depth=1.0,
        center_width=1.0,
        bump_centers=np.array([[3.0, 0.0]]),
        bump_depths=np.array([0.1]),
        bump_radii=np.array([1.0]),
    ).to_dict()

    assert payload["spec"] is None
    with pytest.raises(ValueError, match="no replayable spec"):
        SyntheticFunction.from_dict(payload)
