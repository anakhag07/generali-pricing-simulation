from __future__ import annotations

from datetime import datetime

import numpy as np

from experiments.config import ExperimentConfig
from experiments.reporting.context import RunContext
from experiments.reporting.plots import PlotReporter
from experiments.results import EstimatorResult, ExperimentResult, OptimizationTrace
from objective import ConstantPolicy, FixedRegressionObjective


def test_plot_reporter_creates_split_plot_folders(monkeypatch, tmp_path) -> None:
    objective = FixedRegressionObjective.from_parameters(
        policy=ConstantPolicy(),
        beta_1=[0.1],
        beta_2=-0.5,
        beta_3=[0.2],
        beta_4=0.4,
    )
    config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=np.asarray([0.0], dtype=float),
        n_samples=4,
        x_fixed=np.arange(4, dtype=float).reshape(-1, 1),
        x_fixed_row_indices=np.asarray([10, 11, 12, 13], dtype=int),
        train_fraction=0.5,
        test_fraction=0.5,
        step_rule="constant",
        perturbation_space="theta",
        t_steps=1,
        plot=True,
        enabled_estimators=("first_order",),
    )
    trace = OptimizationTrace(
        steps=[0],
        u_values=[0.0],
        objective_values=[0.0],
        u_grad_estimates=[0.0],
        theta_grad_norms=[0.0],
        theta_values=[np.asarray([0.0], dtype=float)],
    )
    result = ExperimentResult(
        config=config,
        x_samples=np.asarray([[0.0], [1.0]], dtype=float),
        x_test=np.asarray([[2.0], [3.0]], dtype=float),
        train_row_indices=np.asarray([10, 11], dtype=int),
        test_row_indices=np.asarray([12, 13], dtype=int),
        initial_value=0.0,
        results={
            "first_order": EstimatorResult(
                theta=np.asarray([0.1], dtype=float),
                u=0.1,
                value=0.0,
                time=0.0,
            )
        },
        traces={"first_order": trace},
    )
    run_context = RunContext(
        experiment_name="split-plot-test",
        run_id="run",
        run_dir=tmp_path,
        plots_dir=tmp_path / "plots",
        started_at=datetime(2026, 6, 13, 0, 0, 0),
    )
    delta_plot_calls = []
    delta_histogram_calls = []
    objective_summary_calls = []

    monkeypatch.setattr(
        "experiments.reporting.plots._observed_u_reference",
        lambda result, x_samples, row_indices: np.zeros(x_samples.shape[0], dtype=float),
    )
    monkeypatch.setattr("experiments.reporting.plots._plot_policy_u_histograms", lambda *args, **kwargs: None)
    monkeypatch.setattr("experiments.reporting.plots._plot_policy_acceptance_histograms", lambda *args, **kwargs: None)
    monkeypatch.setattr("experiments.reporting.plots._plot_policy_final_summary_metrics", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "experiments.reporting.plots._plot_policy_delta_u_histograms",
        lambda *args, **kwargs: delta_histogram_calls.append(args),
    )
    monkeypatch.setattr(
        "experiments.reporting.plots._plot_policy_delta_u_by_elasticity",
        lambda *args, **kwargs: delta_plot_calls.append(args),
    )
    monkeypatch.setattr(
        "experiments.reporting.plots._plot_policy_objective_contribution_summary",
        lambda *args, **kwargs: objective_summary_calls.append(args),
    )
    monkeypatch.setattr("experiments.reporting.plots._plot_policy_u_acceptance_histograms", lambda *args, **kwargs: None)

    PlotReporter().on_end(run_context, result)

    assert (run_context.plots_dir / "optimization").is_dir()
    assert (run_context.plots_dir / "policy_train").is_dir()
    assert (run_context.plots_dir / "policy_test").is_dir()
    assert len(delta_histogram_calls) == 2
    assert len(delta_plot_calls) == 2
    assert len(objective_summary_calls) == 2
