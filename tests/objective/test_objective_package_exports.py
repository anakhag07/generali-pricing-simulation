from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import objective as objective_pkg

from objective import (
    ActionBias,
    BiasedObjective,
    BiasModification,
    ConstantPolicy,
    CubicFeatureMap,
    LinearActionBias,
    NoiseModification,
    PreparedGLMBatch,
    PreparedGLMObjective,
    ProximalThetaRegularizer,
    RegularizedObjective,
    StronglyConvexQuadratic,
    QuadraticFeatureMap,
    QuarticFeatureMap,
    PlantedLogisticObjective,
    UpperSupportHingeBias,
    compose_objective,
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
    assert BiasedObjective(objective, lambda_bias=0.1).lambda_bias == 0.1
    assert ActionBias is not None
    assert LinearActionBias(lambda_bias=0.1).lambda_bias == 0.1
    assert (
        UpperSupportHingeBias(lambda_bias=0.1, support_center=1.0, support_radius=0.2).support_upper
        == 1.2
    )
    regularized = RegularizedObjective(
        StronglyConvexQuadratic.isotropic(2),
        (ProximalThetaRegularizer(weight=0.1),),
    )
    assert regularized.theta_dim() == 2
    assert BiasModification(lambda_bias=0.1) is not None
    assert NoiseModification is not None
    assert compose_objective is not None
    assert mean_acceptance_at_constant_u(objective, x_batch, u=1.0) is None
    assert PreparedGLMBatch is not None
    assert PreparedGLMObjective is not None
    assert prepare_glm_batch is not None
    assert prepare_glm_objective is not None
    assert StronglyConvexQuadratic.isotropic(2).theta_dim() == 2
    assert "JaxPreparedGLMObjective" in objective_pkg.__all__
    assert "prepare_jax_glm_objective" in objective_pkg.__all__


def test_objective_import_does_not_eagerly_load_jax() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pythonpath = [str(repo_root / "src")]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(pythonpath),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import objective; import objective.objectives; print('jax' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
        env=env,
    )

    assert result.stdout.strip() == "False"
