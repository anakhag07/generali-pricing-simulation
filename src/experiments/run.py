"""Experiment runner for the pricing optimization demo."""

from __future__ import annotations

from typing import Optional, Tuple

from data.models import Customer, default_rng
from experiments.config import ExperimentConfig
from experiments.helpers import build_objective_fns, run_first_order, run_lbfgs, run_zeroth_order
from experiments.logging import log_summary
from experiments.visualization import plot_fixed_regression_truth, plot_gradient_norms, plot_loss_curves
from optimization.policy import apply_policy


def run_experiment(config: Optional[ExperimentConfig] = None) -> Tuple[float, float, float, float]:
    if config is None:
        config = ExperimentConfig()

    objective_model = config.objective_model
    policy_spec = config.policy_spec
    if objective_model is None or policy_spec is None:
        raise ValueError("ExperimentConfig must define objective_model and policy_spec.")

    rng = default_rng(config.seed)
    customer = Customer.sample(rng, state_dim=config.state_dim)

    objective_fn, oracle_grad_fn, grad_fn = build_objective_fns(objective_model, customer.x)

    u0 = apply_policy(policy_spec, customer.x)
    initial_value = objective_fn(u0)
    u_first, trace_first = run_first_order(
        u0,
        objective_fn,
        oracle_grad_fn,
        grad_fn,
        rng,
        config.t_steps,
        config.step_size,
        config.n_samples,
        config.sigma,
    )
    u_zero, trace_zero = run_zeroth_order(
        u0,
        objective_fn,
        grad_fn,
        rng,
        config.t_steps,
        config.step_size,
        config.n_samples,
        config.sigma,
    )
    u_lbfgs, value_lbfgs = run_lbfgs(u0, objective_fn, grad_fn, config.lbfgs_maxiter)
    value_first = objective_fn(u_first)
    value_zero = objective_fn(u_zero)

    log_summary(
        initial_value,
        u_first,
        value_first,
        u_zero,
        value_zero,
        u_lbfgs,
        value_lbfgs,
        objective_model,
    )
    if config.plot:
        plot_loss_curves(trace_first, trace_zero, config.plot_dir, u_star=u_lbfgs)
        plot_gradient_norms(trace_first, trace_zero, config.plot_dir)
        plot_fixed_regression_truth(
            customer.x,
            objective_model,
            trace_first,
            trace_zero,
            config.plot_dir,
            u_lbfgs=u_lbfgs,
        )
    return initial_value, u_first, u_zero, u_lbfgs
