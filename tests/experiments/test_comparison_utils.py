from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.comparison_utils import (
    ComparisonResult,
    ComparisonSpec,
    collect_comparison_final_rows,
    collect_comparison_trace_rows,
    generate_comparison_runs,
    run_preset_comparison,
    validate_comparison_estimators,
    validate_comparison_x_samples,
)
from experiments.config import ExperimentConfig
from experiments.results import EstimatorResult, ExperimentResult, OptimizationTrace
from objective import FixedRegressionObjective, LinearPolicy


def _build_result(
    *,
    estimator_names: tuple[str, ...] = ("first_order",),
    x_samples: np.ndarray | None = None,
) -> ExperimentResult:
    policy = LinearPolicy()
    objective = FixedRegressionObjective.from_parameters(
        policy=policy,
        beta_1=np.asarray([0.2], dtype=float),
        beta_2=-0.8,
        beta_3=np.asarray([0.1], dtype=float),
        beta_4=0.3,
    )
    x_arr = np.asarray(x_samples if x_samples is not None else [[0.5], [1.0]], dtype=float)
    theta0 = np.asarray([0.1, 0.2], dtype=float)
    config = ExperimentConfig(
        state_dim=1,
        objective=objective,
        theta0=theta0,
        n_samples=x_arr.shape[0],
        step_rule="constant",
        perturbation_space="theta",
        t_steps=2,
        plot=False,
        enabled_estimators=estimator_names,
    )
    traces = {
        name: OptimizationTrace(
            steps=[0, 1],
            u_values=[0.0, 0.1],
            objective_values=[1.0, 0.8],
            u_grad_estimates=[0.2, 0.1],
            theta_grad_norms=[0.4, 0.2],
            step_sizes=[0.01, 0.01],
            optimizer_status=0,
            optimizer_message="ok",
        )
        for name in estimator_names
    }
    results = {
        name: EstimatorResult(
            theta=np.asarray([0.3, -0.2], dtype=float),
            u=0.1,
            value=0.8,
            time=0.01,
            mean_acceptance=0.9,
        )
        for name in estimator_names
    }
    return ExperimentResult(
        config=config,
        x_samples=x_arr,
        initial_value=1.0,
        results=results,
        traces=traces,
    )


def test_generate_comparison_runs_merges_common_and_spec_overrides() -> None:
    runs = generate_comparison_runs(
        specs=(
            ComparisonSpec("seeded", "fixed_regression_base", {"seed": 11}),
            ComparisonSpec("sigma", "fixed_regression_base", {"sigma": 0.03}),
        ),
        common_overrides={"seed": 7, "plot": False},
    )

    assert [run.name for run in runs] == ["seeded", "sigma"]
    assert runs[0].config.seed == 11
    assert runs[0].config.plot is False
    assert runs[1].config.seed == 7
    assert runs[1].config.sigma == pytest.approx(0.03)


def test_generate_comparison_runs_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        generate_comparison_runs(
            specs=(
                ComparisonSpec("same", "fixed_regression_base"),
                ComparisonSpec("same", "planted_logistic_base"),
            )
        )


def test_collect_comparison_rows_include_labels_and_summed_objective() -> None:
    results = [ComparisonResult("constant", "preset_a", _build_result())]

    trace_rows = collect_comparison_trace_rows(results)
    final_rows = collect_comparison_final_rows(results)

    assert trace_rows[0]["comparison"] == "constant"
    assert trace_rows[0]["preset"] == "preset_a"
    assert trace_rows[1]["objective"] == pytest.approx(0.8)
    assert final_rows[0]["final_value"] == pytest.approx(0.8)
    assert final_rows[0]["final_objective_sum"] == pytest.approx(1.6)
    assert final_rows[0]["mean_acceptance"] == pytest.approx(0.9)


def test_validate_comparison_estimators_rejects_mismatch() -> None:
    results = [
        ComparisonResult("one", "preset_a", _build_result(estimator_names=("first_order",))),
        ComparisonResult("two", "preset_b", _build_result(estimator_names=("spsa",))),
    ]

    with pytest.raises(ValueError, match="share estimator sets"):
        validate_comparison_estimators(results)


def test_validate_comparison_x_samples_rejects_mismatch() -> None:
    results = [
        ComparisonResult("one", "preset_a", _build_result(x_samples=np.asarray([[0.0], [1.0]]))),
        ComparisonResult("two", "preset_b", _build_result(x_samples=np.asarray([[0.0], [2.0]]))),
    ]

    with pytest.raises(ValueError, match="share x_samples"):
        validate_comparison_x_samples(results)


def test_run_preset_comparison_writes_aggregate_outputs(monkeypatch, tmp_path: Path) -> None:
    import experiments.comparison_utils as comparison_utils

    def fake_run_experiment(config, step_reporter=None):
        return _build_result(estimator_names=tuple(config.enabled_estimators))

    monkeypatch.setattr(comparison_utils, "run_experiment", fake_run_experiment)

    results = run_preset_comparison(
        specs=(
            ComparisonSpec("one", "fixed_regression_base"),
            ComparisonSpec("two", "fixed_regression_base"),
        ),
        common_overrides={
            "enabled_estimators": ("first_order",),
            "plot": False,
            "verbose": False,
            "wandb_enabled": False,
        },
        project_name="comparison-test",
        runs_root=str(tmp_path),
        validate_shared_x=True,
    )

    comparison_dirs = list((tmp_path / "comparison-test").glob("comparison_*"))
    assert len(results) == 2
    assert len(comparison_dirs) == 1
    assert (comparison_dirs[0] / "comparison_traces.csv").exists()
    assert (comparison_dirs[0] / "comparison_finals.csv").exists()
    assert (comparison_dirs[0] / "objective_curves.png").exists()
