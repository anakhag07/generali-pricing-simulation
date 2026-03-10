import numpy as np

from objective.base import StateVector
from model.policy import POLICY_CONSTANT, POLICY_LINEAR, POLICY_SOFTMAX, policy_u, policy_u_batch


def test_policy_u_batch_matches_scalar() -> None:
    rng = np.random.default_rng(0)
    x_array = rng.normal(size=(5, 3))
    x_samples = [StateVector(values=row) for row in x_array]

    theta_const = np.array([1.1], dtype=float)
    batch_const = policy_u_batch(theta_const, x_array, kind=POLICY_CONSTANT)
    scalar_const = np.array([policy_u(theta_const, x, kind=POLICY_CONSTANT) for x in x_samples])
    assert np.allclose(batch_const, scalar_const)

    theta_linear = np.array([0.2, -0.1, 0.3, 0.4], dtype=float)
    batch_linear = policy_u_batch(theta_linear, x_array, kind=POLICY_LINEAR)
    phi_array = np.concatenate([np.ones((x_array.shape[0], 1)), x_array], axis=1)
    batch_linear_cached = policy_u_batch(
        theta_linear,
        x_array,
        kind=POLICY_LINEAR,
        phi_array=phi_array,
    )
    scalar_linear = np.array([policy_u(theta_linear, x, kind=POLICY_LINEAR) for x in x_samples])
    assert np.allclose(batch_linear, scalar_linear)
    assert np.allclose(batch_linear_cached, scalar_linear)

    theta_softmax = np.array([0.1, -0.3, 0.5, -0.2], dtype=float)
    batch_softmax = policy_u_batch(theta_softmax, x_array, kind=POLICY_SOFTMAX)
    batch_softmax_cached = policy_u_batch(
        theta_softmax,
        x_array,
        kind=POLICY_SOFTMAX,
        phi_array=phi_array,
    )
    scalar_softmax = np.array([policy_u(theta_softmax, x, kind=POLICY_SOFTMAX) for x in x_samples])
    assert np.allclose(batch_softmax, scalar_softmax)
    assert np.allclose(batch_softmax_cached, scalar_softmax)
