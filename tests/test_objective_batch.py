import numpy as np

from objective.objectives import FixedRegressionObjective, PlantedLogisticObjective
from objective.policy import ConstantPolicy


def test_fixed_objective_value_batch_deterministic() -> None:
    """Test that _value_batch produces deterministic results."""
    rng = np.random.default_rng(1)
    x_array = rng.normal(size=(4, 2))
    u_array = rng.uniform(0.7, 1.3, size=4)
    
    policy = ConstantPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.4, 0.2],
        beta_2=-0.5,
        beta_3=[0.3, 0.1],
        beta_4=0.6,
    )

    batch_values_1 = objective._value_batch(x_array, u_array)
    batch_values_2 = objective._value_batch(x_array, u_array)

    assert np.allclose(batch_values_1, batch_values_2)


def test_fixed_objective_grad_batch_deterministic() -> None:
    """Test that _grad_u_batch produces deterministic results."""
    rng = np.random.default_rng(2)
    x_array = rng.normal(size=(4, 2))
    u_array = rng.uniform(0.7, 1.3, size=4)
    
    policy = ConstantPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.4, 0.2],
        beta_2=-0.5,
        beta_3=[0.3, 0.1],
        beta_4=0.6,
    )

    batch_grads_1 = objective._grad_u_batch(x_array, u_array)
    batch_grads_2 = objective._grad_u_batch(x_array, u_array)

    assert np.allclose(batch_grads_1, batch_grads_2)


def test_planted_logistic_batch_deterministic() -> None:
    """Test that _value_batch and _grad_u_batch produce deterministic results."""
    rng = np.random.default_rng(3)
    x_array = rng.normal(size=(5, 3))
    u_array = rng.uniform(0.6, 1.4, size=5)
    
    policy = ConstantPolicy()
    objective = PlantedLogisticObjective(
        policy=policy,
        alpha=1.1,
        beta=np.array([0.2, -0.1, 0.05], dtype=float),
        bias=-0.3,
        u_star=1.0,
    )

    batch_values_1 = objective._value_batch(x_array, u_array)
    batch_values_2 = objective._value_batch(x_array, u_array)
    assert np.allclose(batch_values_1, batch_values_2)

    batch_grads_1 = objective._grad_u_batch(x_array, u_array)
    batch_grads_2 = objective._grad_u_batch(x_array, u_array)
    assert np.allclose(batch_grads_1, batch_grads_2)


def test_fixed_objective_value_at_u() -> None:
    """Test value_at_u method."""
    rng = np.random.default_rng(4)
    x_array = rng.normal(size=(10, 2))
    
    policy = ConstantPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=[0.4, 0.2],
        beta_2=-0.5,
        beta_3=[0.3, 0.1],
        beta_4=0.6,
    )

    u = 1.0
    value = objective.value_at_u(x_array, u)
    assert isinstance(value, float)
    assert np.isfinite(value)
