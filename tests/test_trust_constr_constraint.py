from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scipy.optimize import NonlinearConstraint

from optimization import FiniteDifferenceGradient, FirstOrderGradient, Optimization


class SimpleAcceptanceObjective:
    def __init__(self, acceptance_floor: float) -> None:
        self.acceptance_floor = float(acceptance_floor)

    def value(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        del x_batch
        theta_arr = np.asarray(theta, dtype=float)
        return float((theta_arr[0] - 0.2) ** 2)

    def grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        del x_batch
        theta_arr = np.asarray(theta, dtype=float)
        return np.asarray([2.0 * (theta_arr[0] - 0.2)], dtype=float)

    def mean_acceptance(self, theta: np.ndarray, x_batch: np.ndarray) -> float:
        del x_batch
        theta_arr = np.asarray(theta, dtype=float)
        return float(theta_arr[0])

    def mean_acceptance_grad(self, theta: np.ndarray, x_batch: np.ndarray) -> np.ndarray:
        del theta, x_batch
        return np.asarray([1.0], dtype=float)


def test_trust_constr_passes_nonlinear_constraint_to_minimize() -> None:
    seen: dict[str, object] = {}

    def fake_minimize(fun, **kwargs):
        del fun
        seen.update(kwargs)
        callback = kwargs["callback"]
        callback(np.asarray([0.6], dtype=float), None)
        return SimpleNamespace(
            x=np.asarray([0.6], dtype=float),
            status=0,
            message="ok",
            v=[np.asarray([-0.4], dtype=float)],
        )

    optimizer = Optimization(
        SimpleAcceptanceObjective(acceptance_floor=0.6),
        np.zeros((4, 1), dtype=float),
        FirstOrderGradient(),
        algorithm="trust-constr",
        t_steps=5,
        n_grad_samples=1,
        sigma=1e-3,
        minimize_fn=fake_minimize,
    )

    _, trace = optimizer.solve(np.asarray([0.0], dtype=float))

    assert seen["method"] == "trust-constr"
    constraints = seen["constraints"]
    assert isinstance(constraints, list)
    assert len(constraints) == 1
    assert isinstance(constraints[0], NonlinearConstraint)
    assert trace.constraint_violation == 0.0
    assert trace.acceptance_multiplier == 0.4


def test_trust_constr_first_order_enforces_acceptance_floor() -> None:
    optimizer = Optimization(
        SimpleAcceptanceObjective(acceptance_floor=0.6),
        np.zeros((4, 1), dtype=float),
        FirstOrderGradient(),
        algorithm="trust-constr",
        t_steps=50,
        n_grad_samples=1,
        sigma=1e-3,
    )

    theta_final, trace = optimizer.solve(np.asarray([0.0], dtype=float))

    assert theta_final[0] >= 0.6 - 1e-4
    assert trace.constraint_violation is not None
    assert trace.constraint_violation <= 1e-4
    assert trace.acceptance_multiplier is not None
    assert trace.acceptance_multiplier > 0.0


def test_trust_constr_finite_difference_enforces_acceptance_floor() -> None:
    optimizer = Optimization(
        SimpleAcceptanceObjective(acceptance_floor=0.6),
        np.zeros((4, 1), dtype=float),
        FiniteDifferenceGradient(),
        algorithm="trust-constr",
        t_steps=50,
        n_grad_samples=1,
        sigma=1e-6,
    )

    theta_final, trace = optimizer.solve(np.asarray([0.0], dtype=float))

    assert theta_final[0] >= 0.6 - 1e-4
    assert trace.constraint_violation is not None
    assert trace.constraint_violation <= 1e-4
    assert trace.acceptance_multiplier is not None
    assert trace.acceptance_multiplier > 0.0
