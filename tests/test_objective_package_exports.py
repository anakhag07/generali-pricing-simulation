from __future__ import annotations

import numpy as np

from objective import (
    ConstantPolicy,
    PlantedLogisticObjective,
    StateVector,
    default_rng,
)


def test_objective_package_exports_are_importable() -> None:
    """Test that core objective module exports are importable and functional."""
    rng = default_rng(7)
    x = StateVector.sample(rng, dim=2)
    x_array = np.asarray(x, dtype=float).reshape(1, -1)
    theta = np.asarray([1.0], dtype=float)
    
    policy = ConstantPolicy()
    objective = PlantedLogisticObjective(
        policy=policy,
        alpha=1.0,
        beta=[0.1, -0.2],
        bias=0.0,
        u_star=1.0,
    )

    # Test theta-level interface
    value = objective.value(theta, x_array)
    grad = objective.grad(theta, x_array)
    assert isinstance(value, float)
    assert isinstance(grad, np.ndarray)
    
    # Test scalar methods
    value_scalar = objective.value_scalar(x, u=1.0)
    grad_u_scalar = objective.grad_u_scalar(x, u=1.0)
    assert isinstance(value_scalar, float)
    assert isinstance(grad_u_scalar, float)
