"""Class-based optimization entry point for SciPy and manual step rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from objective.base import Objective
from objective.utils import _mean_action
from optimization.helpers import (
    objective_value_on_indices,
    sample_indices,
    scipy_method,
    x_batch,
)
from optimization.steps import (
    OPTAX_STEP_RULES,
    STEP_RULE_ARMIJO,
    STEP_RULE_CONSTANT,
    STEP_RULE_LBFGSB,
    STEP_RULE_TRUST_CONSTR,
    armijo_backtracking_step_size,
    constant_step_size,
)

if TYPE_CHECKING:
    from experiments.reporting.base import StepReporter
    from experiments.results import OptimizationTrace

TrueThetaGradFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
MinimizeFn = Callable[..., Any]


class Optimization:
    """Theta optimizer with SciPy constrained/unconstrained or manual fixed-step updates."""

    def __init__(
        self,
        objective: Objective,
        x_samples: Any,
        gradient: Any,
        *,
        algorithm: str = STEP_RULE_LBFGSB,
        t_steps: int,
        n_grad_samples: int,
        sigma: float,
        perturbation_space: str = "theta",
        step_size: float = 0.01,
        batch_size: int | None = None,
        true_grad_theta_fn: TrueThetaGradFn | None = None,
        grad_norm_tol: float | None = None,
        ftol: float | None = None,
        initial_constr_penalty: float | None = None,
        step_reporter: "StepReporter | None" = None,
        method_label: str | None = None,
        batch_rng: np.random.Generator | None = None,
        gradient_rng: np.random.Generator | None = None,
        rng: np.random.Generator | None = None,
        minimize_fn: MinimizeFn = minimize,
    ) -> None:
        """Configure SciPy-based optimization over theta.

        Args:
            objective: Theta-level objective implementing value(theta, x_batch) and grad(theta, x_batch).
            x_samples: State samples array, shape (n_samples, state_dim).
            gradient: Gradient method object with setup() and theta_grad() methods.
            algorithm: Optimization step rule (``"l-bfgs-b"``, ``"trust-constr"``,
                ``"constant"``, or ``"armijo"``).
            t_steps: Maximum number of optimization steps.
            n_grad_samples: Number of gradient samples for zeroth-order methods.
            sigma: Perturbation scale for zeroth-order methods.
            perturbation_space: Space in which zeroth-order perturbations are applied:
                ``"theta"`` (default) perturbs policy parameters directly; ``"u"`` perturbs
                actions and maps back via chain rule (requires objective.policy).
            step_size: Initial/manual step size for non-SciPy step rules.
            batch_size: Mini-batch size (None for full batch).
            true_grad_theta_fn: Optional function for computing true theta gradients.
            grad_norm_tol: Early stopping threshold on gradient norm.
            ftol: SciPy function tolerance.
            initial_constr_penalty: Initial trust-constr penalty parameter.
            step_reporter: Optional reporter for per-step metrics.
            method_label: Label for this optimization method.
            batch_rng: Random number generator for mini-batch sampling.
            gradient_rng: Random number generator for stochastic gradient perturbations.
            rng: Backward-compatible fallback RNG used when stream-specific RNGs are omitted.
            minimize_fn: SciPy minimize function (for testing).
        """
        if perturbation_space not in {"theta", "u"}:
            raise ValueError("perturbation_space must be 'theta' or 'u'.")
        self.objective = objective
        self.gradient = gradient
        self.perturbation_space = perturbation_space
        self.algorithm = algorithm
        self.t_steps = int(t_steps)
        self.n_grad_samples = int(n_grad_samples)
        self.sigma = float(sigma)
        self.step_size = float(step_size)
        self.batch_size = batch_size
        self.true_grad_theta_fn = true_grad_theta_fn
        self.grad_norm_tol = grad_norm_tol
        self.ftol = ftol
        self.initial_constr_penalty = initial_constr_penalty
        self.step_reporter = step_reporter
        self.method_label = method_label if method_label is not None else getattr(self.gradient, "name", "opt")
        fallback_rng = rng if rng is not None else np.random.default_rng(0)
        self.batch_rng = batch_rng if batch_rng is not None else fallback_rng
        self.gradient_rng = gradient_rng if gradient_rng is not None else fallback_rng
        self.rng = self.gradient_rng
        self._minimize_fn = minimize_fn

        if hasattr(x_samples, "iloc") and hasattr(x_samples, "columns"):
            x_arr = x_samples.reset_index(drop=True).copy()
            if x_arr.ndim != 2:
                raise ValueError("x_samples must be a 2D array/DataFrame.")
        else:
            x_arr = np.asarray(x_samples, dtype=float)
            if x_arr.ndim != 2:
                raise ValueError("x_samples must be a 2D array/DataFrame.")
        if x_arr.shape[0] < 1:
            raise ValueError("x_samples must contain at least one sample.")
        self.x_array = x_arr
        self.n_total = self.x_array.shape[0]
        self.batch_size_eff = self.n_total if batch_size is None else int(batch_size)
        if self.batch_size_eff <= 0 or self.batch_size_eff > self.n_total:
            raise ValueError("batch_size must be in [1, len(x_samples)].")
        if self.grad_norm_tol is not None and self.grad_norm_tol <= 0.0:
            raise ValueError("grad_norm_tol must be positive when provided.")
        if self.ftol is not None and self.ftol <= 0.0:
            raise ValueError("ftol must be positive when provided.")
        if self.initial_constr_penalty is not None and self.initial_constr_penalty <= 0.0:
            raise ValueError("initial_constr_penalty must be positive when provided.")
        if self.t_steps <= 0:
            raise ValueError("t_steps must be positive.")
        self._full_indices = np.arange(self.n_total, dtype=int)

    def solve(self, theta_start: np.ndarray) -> tuple[np.ndarray, "OptimizationTrace"]:
        """Run the configured optimizer and return final theta with trace."""
        theta0 = np.asarray(theta_start, dtype=float)
        self.gradient.setup(self, theta0)

        steps: list[int] = []
        u_values: list[float | None] = []
        values: list[float] = []
        u_grad_estimates: list[float] = []
        theta_grad_norms: list[float] = []
        true_theta_grad_norms: list[float] = []
        mean_acceptance_values: list[float] = []
        projected_loss_values: list[float] = []
        projected_revenue_values: list[float] = []
        theta_values: list[np.ndarray] = []
        step_sizes: list[float] | None = (
            []
            if self.algorithm in {STEP_RULE_CONSTANT, STEP_RULE_ARMIJO}
            or self.algorithm in OPTAX_STEP_RULES
            else None
        )
        constraint_violation: float | None = None
        acceptance_multiplier: float | None = None
        constraint_penalty: float | None = None
        optimizer_optimality: float | None = None
        optimizer_lagrangian_grad: np.ndarray | None = None
        last_optimizer_grad_key: tuple[tuple[int, ...], bytes] | None = None
        last_optimizer_grad: np.ndarray | None = None

        def theta_key(theta_vec: np.ndarray) -> tuple[tuple[int, ...], bytes]:
            theta_arr = np.ascontiguousarray(np.asarray(theta_vec, dtype=float))
            return tuple(theta_arr.shape), theta_arr.tobytes()

        def value_fn(theta_vec: np.ndarray) -> float:
            theta_arr = np.asarray(theta_vec, dtype=float)
            indices = sample_indices(self.batch_rng, self.batch_size_eff, self.n_total, self._full_indices)
            return objective_value_on_indices(self.objective, self.x_array, self.n_total, theta_arr, indices)

        def grad_fn(theta_vec: np.ndarray) -> np.ndarray:
            nonlocal last_optimizer_grad_key, last_optimizer_grad
            theta_arr = np.asarray(theta_vec, dtype=float)
            indices = sample_indices(self.batch_rng, self.batch_size_eff, self.n_total, self._full_indices)
            grad_theta = np.asarray(self.gradient.theta_grad(self, theta_arr, indices), dtype=float)
            last_optimizer_grad_key = theta_key(theta_arr)
            last_optimizer_grad = grad_theta.copy()
            return grad_theta

        def acceptance_floor() -> float | None:
            floor = getattr(self.objective, "acceptance_floor", None)
            if floor is None:
                return None
            return float(floor)

        def mean_acceptance_fn(theta_vec: np.ndarray) -> float:
            acceptance_fn = getattr(self.objective, "mean_acceptance", None)
            if not callable(acceptance_fn):
                raise ValueError(
                    "step_rule='trust-constr' requires objective.mean_acceptance(theta, x_batch)."
                )
            return float(acceptance_fn(np.asarray(theta_vec, dtype=float), self.x_array))

        def mean_acceptance_grad_fn(theta_vec: np.ndarray) -> np.ndarray:
            acceptance_grad_fn = getattr(self.objective, "mean_acceptance_grad", None)
            if not callable(acceptance_grad_fn):
                raise ValueError(
                    "step_rule='trust-constr' requires objective.mean_acceptance_grad(theta, x_batch)."
                )
            return np.asarray(acceptance_grad_fn(np.asarray(theta_vec, dtype=float), self.x_array), dtype=float)

        def constraint_margin_fn(theta_vec: np.ndarray) -> float:
            margin_fn = getattr(self.objective, "constraint_margin", None)
            if not callable(margin_fn):
                return mean_acceptance_fn(theta_vec) - float(acceptance_floor())
            return float(margin_fn(np.asarray(theta_vec, dtype=float)))

        def constraint_margin_grad_fn(theta_vec: np.ndarray) -> np.ndarray:
            margin_grad_fn = getattr(self.objective, "constraint_margin_grad", None)
            if not callable(margin_grad_fn):
                return mean_acceptance_grad_fn(theta_vec)
            return np.asarray(margin_grad_fn(np.asarray(theta_vec, dtype=float)), dtype=float)

        def trust_constr_constraint() -> NonlinearConstraint:
            floor = acceptance_floor()
            if floor is None:
                raise ValueError("step_rule='trust-constr' requires objective.acceptance_floor.")
            if callable(getattr(self.objective, "constraint_margin", None)):
                return NonlinearConstraint(
                    fun=lambda theta_vec: np.asarray([constraint_margin_fn(theta_vec)], dtype=float),
                    lb=np.asarray([0.0], dtype=float),
                    ub=np.asarray([np.inf], dtype=float),
                    jac=lambda theta_vec: np.atleast_2d(constraint_margin_grad_fn(theta_vec)),
                )
            return NonlinearConstraint(
                fun=lambda theta_vec: np.asarray([mean_acceptance_fn(theta_vec)], dtype=float),
                lb=np.asarray([floor], dtype=float),
                ub=np.asarray([np.inf], dtype=float),
                jac=lambda theta_vec: np.atleast_2d(mean_acceptance_grad_fn(theta_vec)),
            )

        def trust_constr_callback(theta_vec: np.ndarray, state: Any | None = None) -> bool:
            del state
            record(theta_vec)
            return False

        def final_constraint_violation(theta_vec: np.ndarray) -> float | None:
            floor = acceptance_floor()
            if floor is None:
                return None
            if callable(getattr(self.objective, "constraint_margin", None)):
                return max(0.0, -constraint_margin_fn(theta_vec))
            return max(0.0, floor - mean_acceptance_fn(theta_vec))

        def extract_acceptance_multiplier(result: Any) -> float | None:
            dual = getattr(result, "v", None)
            if dual is None or len(dual) == 0:
                return None
            first_dual = np.asarray(dual[0], dtype=float).reshape(-1)
            if first_dual.size == 0:
                return None
            return max(0.0, float(-first_dual[0]))

        def extract_constraint_penalty(result: Any) -> float | None:
            penalty = getattr(result, "constr_penalty", None)
            if penalty is None:
                return None
            return float(penalty)

        def record(theta_vec: np.ndarray, step_size: float | None = None) -> None:
            theta_arr = np.asarray(theta_vec, dtype=float)
            indices = sample_indices(self.batch_rng, self.batch_size_eff, self.n_total, self._full_indices)
            x_batch_arr = x_batch(self.x_array, indices, self.n_total)
            value = objective_value_on_indices(self.objective, self.x_array, self.n_total, theta_arr, indices)
            if last_optimizer_grad_key == theta_key(theta_arr) and last_optimizer_grad is not None:
                grad_theta = last_optimizer_grad.copy()
                self.gradient.advance_rng(self, theta_arr)
            else:
                grad_theta = np.asarray(self.gradient.theta_grad(self, theta_arr, indices), dtype=float)
            theta_grad_norm = float(np.linalg.norm(grad_theta))

            true_theta_grad_norm = None
            if self.true_grad_theta_fn is not None:
                grad_true = np.asarray(
                    self.true_grad_theta_fn(theta_arr, x_batch_arr),
                    dtype=float,
                )
                true_theta_grad_norm = float(np.linalg.norm(grad_true))

            step = len(steps)

            # Compute mean action if policy is available
            policy = getattr(self.objective, "policy", None)
            if policy is not None:
                mean_u = _mean_action(self.objective, theta_arr, x_batch_arr)
            else:
                mean_u = None

            mean_acceptance = None
            projected_loss = None
            projected_revenue = None
            step_metrics_fn = getattr(self.objective, "_step_metrics", None)
            if callable(step_metrics_fn):
                metrics = step_metrics_fn(theta_arr, x_batch_arr)
                mean_acceptance = metrics.get("mean_acceptance")
                projected_loss = metrics.get("projected_loss")
                projected_revenue = metrics.get("projected_revenue")

            steps.append(step)
            u_values.append(mean_u)
            values.append(value)
            u_grad_estimates.append(float("nan"))
            theta_grad_norms.append(theta_grad_norm)
            theta_values.append(theta_arr.copy())
            if step_sizes is not None:
                step_sizes.append(float("nan") if step_size is None else float(step_size))
            if true_theta_grad_norm is not None:
                true_theta_grad_norms.append(true_theta_grad_norm)
            if mean_acceptance is not None:
                mean_acceptance_values.append(float(mean_acceptance))
            if projected_loss is not None:
                projected_loss_values.append(float(projected_loss))
            if projected_revenue is not None:
                projected_revenue_values.append(float(projected_revenue))
            if self.step_reporter is not None:
                self.step_reporter.log_step(
                    self.method_label,
                    step,
                    mean_u,
                    value,
                    theta_grad_norm,
                    step_size=step_size,
                    mean_acceptance=mean_acceptance,
                    projected_loss=projected_loss,
                    projected_revenue=projected_revenue,
                )

        record(theta0)
        optimizer_success: bool
        optimizer_status: int
        optimizer_message: str

        if self.algorithm in {STEP_RULE_CONSTANT, STEP_RULE_ARMIJO}:
            theta_final = theta0.copy()
            optimizer_success = False
            optimizer_status = 1
            optimizer_message = "STOP: reached maximum iterations"
            for _ in range(self.t_steps):
                indices = sample_indices(self.batch_rng, self.batch_size_eff, self.n_total, self._full_indices)
                grad_theta = np.asarray(self.gradient.theta_grad(self, theta_final, indices), dtype=float)
                grad_norm = float(np.linalg.norm(grad_theta))
                if self.grad_norm_tol is not None and grad_norm <= self.grad_norm_tol:
                    optimizer_success = True
                    optimizer_status = 0
                    optimizer_message = "STOP: gradient norm below tolerance"
                    break

                if self.algorithm == STEP_RULE_CONSTANT:
                    step_size = constant_step_size(self.step_size)
                else:
                    step_size = armijo_backtracking_step_size(
                        theta_final,
                        grad_theta,
                        objective_fn=lambda theta_eval: objective_value_on_indices(
                            self.objective,
                            self.x_array,
                            self.n_total,
                            theta_eval,
                            indices,
                        ),
                        initial_step=self.step_size,
                    )

                theta_final = theta_final - step_size * grad_theta
                record(theta_final, step_size=step_size)
        elif self.algorithm in OPTAX_STEP_RULES:
            # Imported lazily so environments without optax can still use the
            # SciPy and manual step rules.
            from optimization.optax_loop import run_optax_minimize_loop

            theta_final, optimizer_success, optimizer_status, optimizer_message = (
                run_optax_minimize_loop(self, theta0, record)
            )
        else:
            options: dict[str, float | int] = {"maxiter": int(self.t_steps)}
            if self.grad_norm_tol is not None:
                options["gtol"] = float(self.grad_norm_tol)
            if self.ftol is not None and self.algorithm != STEP_RULE_TRUST_CONSTR:
                options["ftol"] = float(self.ftol)
            if self.algorithm == STEP_RULE_TRUST_CONSTR and self.initial_constr_penalty is not None:
                options["initial_constr_penalty"] = float(self.initial_constr_penalty)

            minimize_kwargs: dict[str, Any] = {
                "x0": theta0,
                "jac": grad_fn,
                "method": scipy_method(self.algorithm),
                "options": options,
            }
            if self.algorithm == STEP_RULE_TRUST_CONSTR:
                minimize_kwargs["constraints"] = [trust_constr_constraint()]
                minimize_kwargs["callback"] = trust_constr_callback
            else:
                minimize_kwargs["callback"] = record

            result = self._minimize_fn(value_fn, **minimize_kwargs)

            theta_final = np.asarray(result.x, dtype=float)
            optimizer_status = int(getattr(result, "status", 1))
            optimizer_success = bool(getattr(result, "success", optimizer_status == 0))
            optimizer_message = str(getattr(result, "message", ""))
            if self.algorithm == STEP_RULE_TRUST_CONSTR:
                constraint_violation = final_constraint_violation(theta_final)
                acceptance_multiplier = extract_acceptance_multiplier(result)
                constraint_penalty = extract_constraint_penalty(result)
                optimality = getattr(result, "optimality", None)
                if optimality is not None:
                    optimizer_optimality = float(optimality)
                lagrangian_grad = getattr(result, "lagrangian_grad", None)
                if lagrangian_grad is not None:
                    optimizer_lagrangian_grad = np.asarray(lagrangian_grad, dtype=float)
            if not np.allclose(theta_final, theta_values[-1]):
                record(theta_final)

        from experiments.results import OptimizationTrace

        trace = OptimizationTrace(
            steps=steps,
            u_values=u_values,
            objective_values=values,
            u_grad_estimates=u_grad_estimates,
            u_true_gradients=None,
            theta_grad_norms=theta_grad_norms,
            true_theta_grad_norms=true_theta_grad_norms if true_theta_grad_norms else None,
            step_sizes=step_sizes,
            mean_acceptance_values=mean_acceptance_values if mean_acceptance_values else None,
            projected_loss_values=projected_loss_values if projected_loss_values else None,
            projected_revenue_values=projected_revenue_values if projected_revenue_values else None,
            theta_values=theta_values,
            optimizer_success=optimizer_success,
            optimizer_optimality=optimizer_optimality,
            optimizer_lagrangian_grad=optimizer_lagrangian_grad,
            optimizer_status=optimizer_status,
            optimizer_message=optimizer_message,
            constraint_violation=constraint_violation,
            acceptance_multiplier=acceptance_multiplier,
            constraint_penalty=constraint_penalty,
        )
        return theta_final, trace


__all__ = ["Optimization", "TrueThetaGradFn"]
