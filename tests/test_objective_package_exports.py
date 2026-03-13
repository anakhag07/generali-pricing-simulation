from __future__ import annotations

from objective import (
    PlantedLogisticObjective,
    StateVector,
    default_rng,
)


def test_objective_package_exports_are_importable() -> None:
    rng = default_rng(7)
    x = StateVector.sample(rng, dim=2)
    objective = PlantedLogisticObjective(alpha=1.0, beta=[0.1, -0.2], bias=0.0, u_star=1.0)

    value = objective.value(x, u=1.0)
    grad_u = objective.grad_u(x, u=1.0)
    assert isinstance(value, float)
    assert isinstance(grad_u, float)
