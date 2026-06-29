from __future__ import annotations

import numpy as np

from objective import (
    ConstantPolicy,
    CubicFeatureMap,
    PreparedGLMBatch,
    PreparedGLMObjective,
    QuadraticFeatureMap,
    QuarticFeatureMap,
    PlantedLogisticObjective,
    default_rng,
    mean_acceptance_at_constant_u,
    prepare_glm_batch,
    prepare_glm_objective,
    sample_states,
    value_at_constant_u,
    value_for_reporting,
)


def test_objective_package_exports_are_importable() -> None:
    """Test that core objective module exports are importable and functional."""
    rng = default_rng(7)
    x_batch = sample_states(rng, n=1, dim=2)
    theta = np.asarray([1.0], dtype=float)
    
    policy = ConstantPolicy()
    feature_map = QuadraticFeatureMap()
    assert feature_map.output_dim(2) == 5
    assert CubicFeatureMap().output_dim(2) == 6
    assert QuarticFeatureMap().output_dim(2) == 7
    objective = PlantedLogisticObjective(
        policy=policy,
        alpha=1.0,
        beta=[0.1, -0.2],
        bias=0.0,
        u_star=1.0,
    )

    # Test theta-level interface
    value = objective.value(theta, x_batch)
    grad = objective.grad(theta, x_batch)
    assert isinstance(value, float)
    assert isinstance(grad, np.ndarray)
    
    # Test value_at_u
    value_at_u = objective.value_at_u(x_batch, u=1.0)
    assert isinstance(value_at_u, float)
    assert isinstance(value_at_constant_u(objective, x_batch, u=1.0), float)
    assert isinstance(value_for_reporting(objective, theta, x_batch), float)
    assert mean_acceptance_at_constant_u(objective, x_batch, u=1.0) is None
    assert PreparedGLMBatch is not None
    assert PreparedGLMObjective is not None
    assert prepare_glm_batch is not None
    assert prepare_glm_objective is not None
