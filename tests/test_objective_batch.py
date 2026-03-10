import numpy as np

from objective.base import StateVector
from objective.fixed_objective import FixedRegressionObjective
from objective.planted_logistic import PlantedLogisticObjective


def test_fixed_objective_batch_matches_scalar() -> None:
    rng = np.random.default_rng(1)
    x_array = rng.normal(size=(4, 2))
    u_array = rng.uniform(0.7, 1.3, size=4)
    x_samples = [StateVector(values=row) for row in x_array]
    objective = FixedRegressionObjective.from_parameters(
        beta_1=[0.4, 0.2],
        beta_2=-0.5,
        beta_3=[0.3, 0.1],
        beta_4=0.6,
    )

    batch_values = objective.value_batch(x_array, u_array)
    batch_grads = objective.grad_u_batch(x_array, u_array)
    scalar_values = np.array([objective.value(x, u) for x, u in zip(x_samples, u_array)])
    scalar_grads = np.array([objective.grad_u(x, u) for x, u in zip(x_samples, u_array)])

    assert np.allclose(batch_values, scalar_values)
    assert np.allclose(batch_grads, scalar_grads)


def test_planted_logistic_batch_matches_scalar() -> None:
    rng = np.random.default_rng(2)
    x_array = rng.normal(size=(5, 3))
    u_array = rng.uniform(0.6, 1.4, size=5)
    x_samples = [StateVector(values=row) for row in x_array]
    objective = PlantedLogisticObjective(
        alpha=1.1,
        beta=np.array([0.2, -0.1, 0.05], dtype=float),
        bias=-0.3,
        u_star=1.0,
    )

    batch_values = objective.value_batch(x_array, u_array)
    batch_grads = objective.grad_u_batch(x_array, u_array)
    scalar_values = np.array([objective.value(x, u) for x, u in zip(x_samples, u_array)])
    scalar_grads = np.array([objective.grad_u(x, u) for x, u in zip(x_samples, u_array)])

    assert np.allclose(batch_values, scalar_values)
    assert np.allclose(batch_grads, scalar_grads)
