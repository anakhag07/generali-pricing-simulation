import numpy as np
from types import SimpleNamespace

from objective.base import StateVector
from experiments.helpers import run_first_order, run_gauss_stein, run_spsa
from objective.policy import POLICY_LINEAR
import optimization.solvers as solvers


class SimpleObjective:
    def value(self, x: StateVector, u: float) -> float:
        return u**2

    def grad_u(self, x: StateVector, u: float) -> float:
        return 2.0 * u


def test_first_order_minimize_reduces_objective() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = [StateVector(values=[1.0]), StateVector(values=[-0.5])]
    objective = SimpleObjective()
    rng = np.random.default_rng(0)

    theta_final, trace = run_first_order(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        rng,
        t_steps=25,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=4,
        sigma=0.1,
    )
    assert trace.objective_values
    assert trace.objective_values[-1] <= trace.objective_values[0] + 1e-10
    assert theta_final.shape == theta_start.shape


def test_gauss_stein_minimize_is_seed_deterministic() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = [StateVector(values=[1.0]), StateVector(values=[-0.5])]
    objective = SimpleObjective()

    theta_a, trace_a = run_gauss_stein(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        np.random.default_rng(11),
        t_steps=20,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=8,
        sigma=0.1,
    )
    theta_b, trace_b = run_gauss_stein(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        np.random.default_rng(11),
        t_steps=20,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=8,
        sigma=0.1,
    )

    assert np.allclose(theta_a, theta_b)
    assert np.allclose(trace_a.objective_values, trace_b.objective_values)


def test_spsa_minimize_is_seed_deterministic() -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = [StateVector(values=[1.0]), StateVector(values=[-0.5])]
    objective = SimpleObjective()

    theta_a, trace_a = run_spsa(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        np.random.default_rng(17),
        t_steps=20,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=8,
        sigma=0.1,
    )
    theta_b, trace_b = run_spsa(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        np.random.default_rng(17),
        t_steps=20,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=8,
        sigma=0.1,
    )

    assert np.allclose(theta_a, theta_b)
    assert np.allclose(trace_a.objective_values, trace_b.objective_values)


def test_first_order_passes_ftol_to_minimize(monkeypatch) -> None:
    theta_start = np.asarray([0.2, 0.3], dtype=float)
    x_samples = [StateVector(values=[1.0]), StateVector(values=[-0.5])]
    objective = SimpleObjective()
    captured_options: dict[str, float | int] = {}

    def fake_minimize(fun, x0, jac, method, options, callback):  # type: ignore[no-untyped-def]
        del fun, jac
        assert method == "L-BFGS-B"
        captured_options.update(options)
        callback(x0)
        return SimpleNamespace(
            x=np.asarray(x0, dtype=float),
            status=0,
            message="stub",
        )

    monkeypatch.setattr(solvers, "minimize", fake_minimize)

    _, trace = run_first_order(
        theta_start,
        POLICY_LINEAR,
        x_samples,
        objective,
        np.random.default_rng(0),
        t_steps=25,
        step_rule="constant",
        step_size=0.01,
        n_grad_samples=4,
        sigma=0.1,
        grad_norm_tol=1e-6,
        ftol=1e-9,
    )

    assert captured_options["maxiter"] == 25
    assert captured_options["gtol"] == 1e-6
    assert captured_options["ftol"] == 1e-9
    assert trace.optimizer_status == 0
    assert trace.optimizer_message == "stub"
